"""Reranking: re-scores search hits with a cross-encoder and keeps the best ones."""

import logging
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.schemas import SearchHit

logger = logging.getLogger(__name__)


class Reranker:
    """Re-orders search hits by how well each chunk actually answers the question.

    Hybrid search compares vectors made separately for the question and the
    chunk, which is fast but rough. A cross-encoder reads the question and the
    chunk together, so it is much more accurate but too slow to run over the
    whole collection. Using it only on the top search candidates gets both.

    Runs on fastembed (ONNX Runtime), the same engine as the embedder, not on
    sentence-transformers (PyTorch). Same model weights and same rankings, but
    loading it took 180 MB of RAM instead of 493 MB, and torch leaves the
    install entirely, which matters on a 512 MB cloud host.
    """

    def __init__(self, model: str, cache_dir: Path, threads: int, batch_size: int) -> None:
        """Load the cross-encoder (downloads it on first use).

        Args:
            model: fastembed name of the cross-encoder model; only the ones in
                TextCrossEncoder.list_supported_models() work.
            cache_dir: Folder where model files are downloaded and reused.
            threads: CPU threads for the model; each one needs its own memory.
            batch_size: (question, chunk) pairs scored per model call; bigger
                is faster but raises peak memory, which ONNX Runtime keeps.
        """
        logger.info("Loading reranker model (first run downloads it)...")
        # Arena off: ONNX Runtime would otherwise keep its biggest-ever memory
        # block forever, and each question grew RAM (to 1.5 GB after three).
        self._model = TextCrossEncoder(
            model_name=model,
            cache_dir=str(cache_dir),
            threads=threads,
            enable_cpu_mem_arena=False,
        )
        self._batch_size = batch_size

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
        # One score per text, in the same order as the texts went in.
        texts = [hit.chunk.text for hit in hits]
        scores = list(self._model.rerank(question, texts, batch_size=self._batch_size))
        reranked = sorted(
            (SearchHit(chunk=hit.chunk, score=float(score)) for hit, score in zip(hits, scores, strict=True)),
            key=lambda hit: hit.score,
            reverse=True,
        )
        logger.info("Reranked %d hit(s), kept %d", len(hits), min(top_k, len(reranked)))
        return reranked[:top_k]
