"""HTTP API: exposes the RAG pipeline to a frontend over JSON."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.pipeline import RagPipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the pipeline once at startup, release it once at shutdown.

    Not at module top level: that would load every model on any import,
    even a test that only imports this file.
    """
    app.state.pipeline = RagPipeline.from_settings(get_settings())
    yield
    app.state.pipeline.close()


app = FastAPI(title="rostee", lifespan=lifespan)
