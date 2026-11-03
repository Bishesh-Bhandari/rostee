"""HTTP API: exposes the RAG pipeline to a frontend over JSON."""

import logging
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
    app.state.pipeline = RagPipeline.from_settings(settings)
    yield
    app.state.pipeline.close()


app = FastAPI(title="rostee", lifespan=lifespan)

# Browsers block a page from calling an API on a different origin unless the
# API allows it. Credentials stay off: with "*" they'd let any site send a
# logged-in user's cookies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    path = get_settings().upload_dir / file_name
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
