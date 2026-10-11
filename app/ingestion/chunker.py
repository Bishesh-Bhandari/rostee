"""Chunking: splits pages into small, overlapping pieces ready for embedding."""

import logging

from app.schemas import Chunk, Page

logger = logging.getLogger(__name__)

# Preferred cut points, best first: paragraph, line, sentence, word.
_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


class TextChunker:
    """Splits each page into overlapping character-based chunks.

    Chunks never span two pages, so every chunk has exactly one page number
    to cite. Within a page, cuts land on the most natural boundary available
    (paragraph > line > sentence > word) instead of mid-word.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        """Create a chunker.

        Args:
            chunk_size: Target maximum characters per chunk.
            chunk_overlap: Characters repeated between neighbouring chunks.

        Raises:
            ValueError: If the sizes can't produce forward progress.
        """
        # Settings already checks this, but the chunker can also be built
        # directly (e.g. in tests), so it guards itself too.
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f"Need chunk_size > 0 and 0 <= chunk_overlap < chunk_size, "
                f"got size={chunk_size}, overlap={chunk_overlap}."
            )
        self._size = chunk_size
        self._overlap = chunk_overlap

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        """Split pages into chunks, keeping each chunk's source location.

        Args:
            pages: Pages from the loader, in document order.

        Returns:
            All chunks from all pages, in document order.
        """
        chunks: list[Chunk] = []
        for page in pages:
            for index, text in enumerate(self._split(page.text)):
                chunks.append(
                    Chunk(
                        file_name=page.file_name,
                        page_number=page.page_number,
                        index=index,
                        text=text,
                    )
                )
        logger.info("Split %d page(s) into %d chunk(s)", len(pages), len(chunks))
        return chunks

    def _split(self, text: str) -> list[str]:
        """Cut one page's text into overlapping pieces."""
        if len(text) <= self._size:
            return [text]

        pieces: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self._size, len(text))
            if end < len(text):
                end = self._find_break(text, start, end)

            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break

            # Step back by the overlap so a sentence cut at this edge still
            # appears whole in the next chunk. max() guarantees we always move
            # forward, even if a break was found very early in the window.
            start = max(end - self._overlap, start + 1)
        return pieces

    def _find_break(self, text: str, start: int, end: int) -> int:
        """Find the best cut point in the back half of the window.

        Only searching the back half stops a paragraph break near the start
        from producing a uselessly tiny chunk.
        """
        earliest = start + self._size // 2
        for sep in _SEPARATORS:
            pos = text.rfind(sep, earliest, end)
            if pos != -1:
                return pos + len(sep)
        return end  # No boundary at all (e.g. a long URL): hard cut.