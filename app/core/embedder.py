"""Embedding: turns text into dense (semantic) and sparse (BM25) vectors."""

import logging
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding

from app.schemas import Embedding, SparseVector

logger = logging.getLogger(__name__)


class Embedder:
    """Produces a dense and a sparse vector for each text.

    Documents and queries go through different methods on purpose: both
    models treat a short question differently from a stored passage
    (asymmetric embedding), and mixing them up silently hurts retrieval.
    """

    def __init__(
        self,
        dense_model: str,
        sparse_model: str,
        dense_dim: int,
        cache_dir: Path,
    ) -> None:
        """Load both models (downloads them on first use).

        Args:
            dense_model: fastembed name of the semantic model.
            sparse_model: fastembed name of the keyword model.
            dense_dim: Expected dense vector length; must match the model.
            cache_dir: Folder where model files are downloaded and reused.

        Raises:
            ValueError: If the model's real dimension doesn't match dense_dim.
        """
        logger.info("Loading embedding models (first run downloads them)...")
        self._dense = TextEmbedding(model_name=dense_model, cache_dir=str(cache_dir))
        self._sparse = SparseTextEmbedding(model_name=sparse_model, cache_dir=str(cache_dir))
        self._check_dimension(dense_model, dense_dim)
        self._dim = dense_dim

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed many stored passages (e.g. chunk texts) in batches.

        Args:
            texts: Passages to embed.

        Returns:
            One Embedding per text, in the same order.
        """
        if not texts:
            return []
        dense = self._dense.embed(texts)    # generators: computed in batches
        sparse = self._sparse.embed(texts)
        embeddings = [self._to_embedding(d, s) for d, s in zip(dense, sparse, strict=True)]
        logger.info("Embedded %d text(s)", len(embeddings))
        return embeddings

    def embed_query(self, text: str) -> Embedding:
        """Embed one user question for searching.

        Args:
            text: The question.

        Returns:
            The question's Embedding, built with the query-side encoders.
        """
        dense = next(iter(self._dense.query_embed(text)))
        sparse = next(iter(self._sparse.query_embed(text)))
        return self._to_embedding(dense, sparse)

    def _check_dimension(self, model: str, expected: int) -> None:
        """Fail at startup if config and model disagree on vector size.

        Without this, the mismatch only shows up later as an insert error
        deep inside Qdrant, far from its real cause.
        """
        actual = len(next(iter(self._dense.embed(["dimension check"]))))
        if actual != expected:
            raise ValueError(
                f"{model} outputs {actual}-dim vectors but DENSE_DIM is {expected}. "
                f"Set DENSE_DIM={actual} in .env (and rebuild the collection)."
            )

    @staticmethod
    def _to_embedding(dense, sparse) -> Embedding:
        """Convert fastembed's numpy outputs into our plain, frozen types."""
        return Embedding(
            dense=tuple(dense.tolist()),
            sparse=SparseVector(
                indices=tuple(sparse.indices.tolist()),
                values=tuple(sparse.values.tolist()),
            ),
        )