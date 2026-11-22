"""HTTP API: exposes the RAG pipeline to a frontend over JSON."""

import asyncio
import logging
import shutil
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import SessionAuth
from app.config import get_settings
from app.pipeline import RagPipeline

# Uvicorn only sets up its own loggers; without this, the pipeline's INFO
# logs (chunk counts, hits, citations) wouldn't show in the terminal.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# These are pydantic models, unlike app/schemas.py: this is the edge where
# untrusted input arrives, so it gets validated here.


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str
    chunks: int


class UploadResponse(BaseModel):
    """Response model for the upload endpoint."""

    file_name: str
    chunks: int


class AskRequest(BaseModel):
    """Request body for the ask endpoint."""

    question: str = Field(min_length=1)


class Source(BaseModel):
    """One cited chunk: enough for the frontend to show a citation."""

    file_name: str
    page_number: int
    text: str


class AskResponse(BaseModel):
    """Response model for the ask endpoint."""

    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline once at startup, release it once at shutdown.

    Not at module top level: that would load every model on any import,
    even a test that only imports this file.
    """
    settings = get_settings()
    # Once here, not per upload: the folder doesn't vanish between requests.
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app.state.auth = SessionAuth(
        password=settings.app_password.get_secret_value(),
        session_days=settings.session_days,
    )
    app.state.pipeline = RagPipeline.from_settings(settings)
    yield
    app.state.pipeline.close()


app = FastAPI(title="rostee", lifespan=lifespan)


SESSION_COOKIE = "rostee_session"
LOGIN_PAGE = "/login.html"
# Reachable without logging in. /health: Render pings it to know the app is
# alive. style.css: the login page needs it to look right. /logout: clicking
# "Log out" after the login already expired should still land on the login
# page, not on a raw 401.
_PUBLIC_PATHS = {"/health", "/login", "/logout", LOGIN_PAGE, "/style.css"}


# Registered before CORS on purpose: the last middleware added runs first,
# so CORS answers browser preflight checks (which never carry a cookie)
# before this check can reject them.
@app.middleware("http")
async def require_login(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Let a request through only if it's public or carries a valid login cookie."""
    token = request.cookies.get(SESSION_COOKIE, "")
    if request.url.path in _PUBLIC_PATHS or request.app.state.auth.is_valid(token):
        return await call_next(request)
    # A person opening a page gets sent to the login form. The frontend's own
    # fetch calls get a 401 instead, which app.js turns into that same redirect.
    if request.method == "GET" and "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(LOGIN_PAGE, status_code=303)
    return JSONResponse({"detail": "Not logged in."}, status_code=401)


# Browsers block a page from calling an API on a different origin unless the
# API allows it. Credentials stay off: with "*" they'd let any site send a
# logged-in user's cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/login")
async def login(request: Request, password: Annotated[str, Form()]) -> RedirectResponse:
    """Check the login form; on success set the session cookie and go to the app.

    A plain HTML form post, not JSON, so the login page needs no JavaScript.
    303 makes the browser follow up with a GET, so refreshing the next page
    doesn't resubmit the password.
    """
    token = request.app.state.auth.login(password)
    if token is None:
        # A one-second pause per wrong guess makes guessing slow. async sleep,
        # not time.sleep: it pauses only this request, not the whole server.
        await asyncio.sleep(1)
        return RedirectResponse(f"{LOGIN_PAGE}?error=1", status_code=303)

    response = RedirectResponse("/", status_code=303)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=get_settings().session_days * 24 * 60 * 60,
        # httponly: page JavaScript can't read it, so injected script can't steal it.
        httponly=True,
        # lax: other websites can't make your browser send it with a POST,
        # so they can't upload or ask in your name.
        samesite="lax",
        # secure: only ever sent over HTTPS. Render's proxy does the HTTPS and
        # tells us via x-forwarded-proto; local http://localhost stays usable.
        secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https",
    )
    return response


@app.post("/logout")
def logout() -> RedirectResponse:
    """Forget the session cookie and go back to the login page."""
    response = RedirectResponse(LOGIN_PAGE, status_code=303)
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/health")
def health(request: Request) -> HealthResponse:
    """Check the server is up and how many chunks are searchable."""
    return HealthResponse(status="ok", chunks=request.app.state.pipeline.count())


@app.post("/documents")
def upload_document(request: Request, file: UploadFile) -> UploadResponse:
    """Save an uploaded PDF, then make it searchable.

    Re-uploading a file with the same name replaces its old version.
    """
    # The name is a label the client wrote, so it can't be trusted: keep only
    # the bare file name, or "../../app/config.py" would write outside the folder.
    # `or ""` because a client may send no name at all, and Path(None) crashes.
    file_name = Path(file.filename or "").name
    if not file_name:
        raise HTTPException(status_code=400, detail="Upload has no file name.")

    # get_settings() is cached, so this returns the startup instance for free.
    settings = get_settings()
    # Checked before saving, so an oversized file never reaches the upload
    # folder or the models. It has already landed in a temp file by now,
    # though: stopping it at the door is a job for nginx in front of the app.
    # `or 0`: size is None only if the client skipped multipart, never for a browser.
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if (file.size or 0) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too big. The limit is {settings.max_upload_mb} MB.",
        )

    path = settings.upload_dir / file_name
    with path.open("wb") as out:
        # Copies in small pieces, so a big PDF never sits in memory all at once.
        shutil.copyfileobj(file.file, out)

    try:
        chunks = request.app.state.pipeline.ingest(path)
    except ValueError as e:
        # Loader errors (not a PDF, scanned, password protected) are the
        # user's to fix, so their message goes back as-is. The rejected file
        # is deleted so the upload folder doesn't fill up with junk.
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UploadResponse(file_name=file_name, chunks=chunks)


@app.post("/ask")
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question from the uploaded documents, with citations.

    POST, not GET: a question in the URL would end up in browser history and
    server logs, and it may contain private details.
    """
    try:
        answer = request.app.state.pipeline.ask(body.question)
    except RuntimeError as e:
        # 502: our server is fine, the LLM behind it failed. The real cause
        # goes to the log only: it may mention keys or quotas.
        logger.exception("LLM call failed")
        raise HTTPException(
            status_code=502, detail="The answer service is unavailable. Try again shortly."
        ) from e
    # Convert the internal dataclass into the API's pydantic shape.
    return AskResponse(
        answer=answer.text,
        sources=[
            Source(file_name=c.file_name, page_number=c.page_number, text=c.text)
            for c in answer.sources
        ],
    )


# Serves frontend/ at "/". Same origin as the API, so the page needs no CORS.
# Must stay last: a mount at "/" matches every path, and routes are tried in
# the order they were added, so anything registered after it would be hidden.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
