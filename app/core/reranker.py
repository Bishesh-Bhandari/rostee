"""Reranking: re-scores search hits with a cross-encoder and keeps the best ones."""

import logging
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.schemas import SearchHit

logger = logging.getLogger(__name__)


class Reranker:
    """Re-orders search hits by how well each chunk actually answers the question.

    Hybrid search compares vectors made separately for the question and the
    chunk, which is fast but rough. A cross-encoder reads the question and the
    chunk together, so it is much more accurate but too slow to run over the
    whole collection. Using it only on the top search candidates gets both.
    """

    def __init__(self, model: str, cache_dir: Path) -> None:
        """Load the cross-encoder (downloads it on first use).

        Args:
            model: Hugging Face name of the cross-encoder model.
            cache_dir: Folder where model files are downloaded and reused.
        """
        logger.info("Loading reranker model (first run downloads it)...")
        self._model = CrossEncoder(model, cache_folder=str(cache_dir))

    def rerank(self, question: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        """Score each hit against the question and keep the best ones.

        Args:
            question: The user's question.
            hits: Candidates from QdrantStore.search.
            top_k: How many hits to keep.

        Returns:
            Up to top_k hits, best first, with scores replaced by the
            cross-encoder's. These scores are raw logits (can be negative),
            only comparable with each other.
        """
        if not hits:
            return []
        scores = self._model.predict([(question, hit.chunk.text) for hit in hits])
        reranked = sorted(
            (SearchHit(chunk=hit.chunk, score=float(score)) for hit, score in zip(hits, scores, strict=True)),
            key=lambda hit: hit.score,
            reverse=True,
        )
        logger.info("Reranked %d hit(s), kept %d", len(hits), min(top_k, len(reranked)))
        return reranked[:top_k]
