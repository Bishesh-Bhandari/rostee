"""RAG pipeline: wires the ingestion and core engines into one flow."""

import logging
from pathlib import Path

from app.config import Settings
from app.core.embedder import Embedder
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import PdfLoader
from app.schemas import Embedding

logger = logging.getLogger(__name__)


class RagPipeline:
    """Runs the full RAG flow: ingest PDFs, then answer questions about them.

    Every engine is built once and reused on each call, so slow setup
    (like loading models) never happens per request.
    """

    def __init__(self, loader: PdfLoader, chunker: TextChunker, embedder: Embedder) -> None:
        """Wire up ready-made engines.

        Use `from_settings()` in the app. Pass engines directly in tests, so a
        test can use small or fake ones.

        Args:
            loader: Reads PDFs into pages.
            chunker: Splits pages into chunks.
            embedder: Turns chunk texts into dense + sparse vectors.
        """
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder

    @classmethod
    def from_settings(cls, settings: Settings) -> "RagPipeline":
        """Build a pipeline with every engine configured from settings.

        Args:
            settings: Validated app configuration.

        Returns:
            A ready-to-use pipeline.
        """
        return cls(
            loader=PdfLoader(),
            chunker=TextChunker(settings.chunk_size, settings.chunk_overlap),
            embedder=Embedder(
                dense_model=settings.dense_model,
                sparse_model=settings.sparse_model,
                dense_dim=settings.dense_dim,
                cache_dir=settings.models_cache_dir,
            ),
        )

    def ingest(self, path: Path) -> list[Embedding]:
        """Load, chunk and embed a PDF so it can be searched later.

        Args:
            path: Path to a .pdf file already saved on disk.

        Returns:
            One embedding per chunk (temporary, until storing is added).
        """
        pages = self._loader.load(path)
        chunks = self._chunker.chunk(pages)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        logger.info("%s: %d pages -> %d chunks", path.name, len(pages), len(chunks))
        return embeddings
