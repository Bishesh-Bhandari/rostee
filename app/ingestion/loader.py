"""Text extraction: turns a PDF file into one Page per page."""

import logging
from pathlib import Path

import pymupdf

from app.schemas import Page

logger = logging.getLogger(__name__)


class PdfLoader:
    """Extracts plain text from a PDF, page by page.

    Blank pages are skipped. Page numbers stay 1-based and match the PDF
    viewer, even when pages are skipped, so citations always point to the
    right page.
    """

    def load(self, path: Path) -> list[Page]:
        """Read a PDF and return its non-empty pages.

        Args:
            path: Path to a .pdf file.

        Returns:
            One Page per page that contains text, in document order.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file isn't a PDF, is corrupted, is password
                protected, or has no extractable text (e.g. a scanned PDF).
        """
        self._check_file(path)

        try:
            doc = pymupdf.open(path)
        except pymupdf.FileDataError as e:
            raise ValueError(
                f"'{path.name}' could not be opened as a PDF; it may be corrupted "
                "or only renamed to .pdf. Try re-exporting it."
            ) from e

        with doc:
            if doc.needs_pass:
                raise ValueError(
                    f"'{path.name}' is password protected. Remove the password "
                    "and upload it again."
                )
            pages = self._extract_pages(doc, path.name)

        if not pages:
            # A PDF with pages but zero text is almost always a scan (images of text).
            raise ValueError(
                f"No text found in '{path.name}'. It's probably a scanned document "
                "and needs OCR before it can be searched."
            )

        logger.info("Loaded %d page(s) from %s", len(pages), path.name)
        return pages

    def _check_file(self, path: Path) -> None:
        """Fail fast on problems that don't need the file to be opened."""
        if not path.is_file():
            raise FileNotFoundError(f"No file at '{path}'. Check the path is correct.")
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"'{path.name}' is not a PDF (got '{path.suffix}'). Only .pdf is supported."
            )

    def _extract_pages(self, doc: pymupdf.Document, file_name: str) -> list[Page]:
        """Pull text from each page, skipping blank ones."""
        pages: list[Page] = []
        # start=1 because PyMuPDF counts pages from 0 but people count from 1.
        for page_number, pdf_page in enumerate(doc, start=1):
            # sort=True reads blocks top-to-bottom, left-to-right, which keeps
            # multi-column layouts and tables in a sensible reading order.
            text = pdf_page.get_text("text", sort=True).strip()
            if not text:
                logger.debug("Skipping blank page %d of %s", page_number, file_name)
                continue
            pages.append(Page(file_name=file_name, page_number=page_number, text=text))
        return pages