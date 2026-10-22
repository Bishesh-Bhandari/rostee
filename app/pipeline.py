"""RAG pipeline: wires the ingestion and core engines into one flow."""

import logging
from pathlib import Path

from app.ingestion.loader import PdfLoader
from app.schemas import Page

logger = logging.getLogger(__name__)


class RagPipeline:
    """Runs the full RAG flow: ingest PDFs, then answer questions about them.

    Every engine is built once here and reused on each call, so slow setup
    (like loading models) never happens per request.
    """

    def __init__(self) -> None:
        """Build the engines the pipeline needs."""
        self._loader = PdfLoader()

    def ingest(self, path: Path) -> list[Page]:
        """Load a PDF so it can be searched later.

        Args:
            path: Path to a .pdf file already saved on disk.

        Returns:
            The pages read from the PDF (temporary, until chunking is added).
        """
        pages = self._loader.load(path)
        logger.info("Loaded %d pages from %s", len(pages), path.name)
        return pages
