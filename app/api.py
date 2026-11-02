"""HTTP API: exposes the RAG pipeline to a frontend over JSON."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import get_settings
from app.pipeline import RagPipeline


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str
    chunks: int


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
