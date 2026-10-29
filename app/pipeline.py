"""RAG pipeline: wires the ingestion and core engines into one flow."""

import logging
from pathlib import Path

from app.config import Settings
from app.core.embedder import Embedder
from app.core.generator import Generator
from app.core.reranker import Reranker
from app.core.store import QdrantStore
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import PdfLoader
from app.schemas import Answer

logger = logging.getLogger(__name__)


class RagPipeline:
    """Runs the full RAG flow: ingest PDFs, then answer questions about them.

    Every engine is built once and reused on each call, so slow setup
    (like loading models) never happens per request.
    """

    def __init__(
        self,
        loader: PdfLoader,
        chunker: TextChunker,
        embedder: Embedder,
        store: QdrantStore,
        reranker: Reranker,
        generator: Generator,
        retrieve_top_k: int,
        rerank_top_k: int,
    ) -> None:
        """Wire up ready-made engines.

        Use `from_settings()` in the app. Pass engines directly in tests, so a
        test can use small or fake ones.

        Args:
            loader: Reads PDFs into pages.
            chunker: Splits pages into chunks.
            embedder: Turns chunk texts into dense + sparse vectors.
            store: Saves chunks with their vectors and searches them.
            reranker: Re-scores search hits and keeps the best ones.
            generator: Writes the final answer from the best chunks.
            retrieve_top_k: How many candidates search returns.
            rerank_top_k: How many of those the reranker keeps for the answer.
        """
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._generator = generator
        self._retrieve_top_k = retrieve_top_k
        self._rerank_top_k = rerank_top_k

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
            store=QdrantStore(
                path=settings.qdrant_path,
                collection_name=settings.collection_name,
                dense_dim=settings.dense_dim,
            ),
            reranker=Reranker(
                model=settings.reranker_model,
                cache_dir=settings.models_cache_dir,
            ),
            generator=Generator(
                # Unwrap the SecretStr only here, where the real key is needed.
                api_key=settings.gemini_api_key.get_secret_value(),
                model=settings.llm_model,
            ),
            retrieve_top_k=settings.retrieve_top_k,
            rerank_top_k=settings.rerank_top_k,
        )

    def ingest(self, path: Path) -> int:
        """Make a PDF searchable: load, chunk, embed and store it.

        Re-ingesting a file with the same name replaces its old chunks.

        Args:
            path: Path to a .pdf file already saved on disk.

        Returns:
            Number of chunks stored.
        """
        pages = self._loader.load(path)
        chunks = self._chunker.chunk(pages)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])
        self._store.add(chunks, embeddings)
        logger.info("%s: %d pages -> %d chunks stored", path.name, len(pages), len(chunks))
        return len(chunks)

    def ask(self, question: str) -> Answer:
        """Answer a question from the ingested documents, with citations.

        Args:
            question: The user's question.

        Returns:
            The answer text and the chunks it cited. Sources are empty when
            the documents don't contain the answer.

        Raises:
            RuntimeError: If the LLM call fails (bad key, quota, outage).
        """
        query = self._embedder.embed_query(question)
        hits = self._store.search(query, limit=self._retrieve_top_k)
        best = self._reranker.rerank(question, hits, top_k=self._rerank_top_k)
        answer = self._generator.answer(question, best)
        # Counts only, never the question text: it may contain private data.
        logger.info(
            "Answered: %d hits -> %d reranked -> %d cited",
            len(hits), len(best), len(answer.sources),
        )
        return answer
