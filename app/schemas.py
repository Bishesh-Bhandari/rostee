"""Data types passed between pipeline stages.

These are plain frozen dataclasses rather than pydantic models: the data is
created by our own code, so it is already trusted. Validation happens at the
edges (config, API requests), not on every internal hand-off.
"""

import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Page:
    """The text of one page from an uploaded document.

    Attributes:
        file_name: Original file name, e.g. "refund_policy.pdf". Shown in citations.
        page_number: 1-based page number, matching what a human sees in a PDF viewer.
        text: Extracted plain text of the page.
    """

    file_name: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A small piece of a page, the unit that gets embedded and searched.

    Attributes:
        file_name: File the chunk came from.
        page_number: 1-based page the chunk came from.
        index: Position of this chunk within its page (0, 1, 2...).
        text: The chunk's text.
    """

    file_name: str
    page_number: int
    index: int
    text: str

    @property
    def id(self) -> str:
        """Stable ID built from the chunk's location in the source document.

        The same file, page and position always produce the same ID, so
        re-uploading a document overwrites its old chunks in the vector store
        instead of storing a duplicate copy of every chunk.

        Returns:
            A UUID string (Qdrant requires UUIDs or integers as point IDs).
        """
        key = f"{self.file_name}:{self.page_number}:{self.index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A chunk returned by search, with how well it matched the question.

    Attributes:
        chunk: The matched chunk.
        score: Relevance score; higher is better. The scale depends on who set it
            (fusion score from the store, or cross-encoder score after reranking),
            so only compare scores from the same stage.
    """

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    """The final reply to a user's question.

    Attributes:
        text: The answer written by the LLM.
        sources: Chunks the answer was based on, in order of relevance. Empty for
            chitchat, out-of-scope questions, or "I don't know" replies.
    """

    text: str
    # A tuple, not a list: a frozen dataclass only stops you replacing the field,
    # it can't stop someone calling .append() on a list inside it.
    sources: tuple[Chunk, ...] = ()