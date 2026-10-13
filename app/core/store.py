"""Vector store: saves chunks with their vectors in Qdrant and runs hybrid search."""

import logging
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.schemas import Chunk, Embedding, SearchHit

logger = logging.getLogger(__name__)

_DENSE = "dense"
_SPARSE = "sparse"


class QdrantStore:
    """Stores chunks as Qdrant points and retrieves them with hybrid search.

    Each point holds both vectors (dense + sparse) and a payload with the
    chunk's text and source location, so a search result already contains
    everything needed for a citation.
    """

    def __init__(self, path: Path, collection_name: str, dense_dim: int) -> None:
        """Open (or create) the local collection.

        Args:
            path: Folder where Qdrant keeps its files (local mode).
            collection_name: Name of the collection holding all chunks.
            dense_dim: Dense vector length; must match the embedder.

        Raises:
            ValueError: If an existing collection was built with a different
                dense dimension (e.g. after changing the embedding model).
        """
        self._client = QdrantClient(path=str(path))
        self._name = collection_name
        self._ensure_collection(dense_dim)

    def add(self, chunks: list[Chunk], embeddings: list[Embedding]) -> None:
        """Save chunks, replacing any older version of the same files.

        Args:
            chunks: Chunks to store.
            embeddings: One Embedding per chunk, in the same order.
        """
        if not chunks:
            return
        # Wipe old points for these files first: a shorter new version would
        # otherwise leave stale chunks behind that still show up in answers.
        self._delete_files({c.file_name for c in chunks})

        points = [
            models.PointStruct(
                id=chunk.id,
                vector={
                    _DENSE: list(emb.dense),
                    _SPARSE: models.SparseVector(
                        indices=list(emb.sparse.indices),
                        values=list(emb.sparse.values),
                    ),
                },
                payload=self._to_payload(chunk),
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        self._client.upsert(collection_name=self._name, points=points)
        logger.info("Stored %d chunk(s) in '%s'", len(points), self._name)

    def search(self, query: Embedding, limit: int) -> list[SearchHit]:
        """Hybrid search: dense and sparse separately, merged with RRF.

        Args:
            query: The question's Embedding (from Embedder.embed_query).
            limit: How many results to return.

        Returns:
            Best matches first. Scores are RRF scores, only comparable
            with each other.
        """
        result = self._client.query_points(
            collection_name=self._name,
            prefetch=[
                models.Prefetch(query=list(query.dense), using=_DENSE, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=list(query.sparse.indices),
                        values=list(query.sparse.values),
                    ),
                    using=_SPARSE,
                    limit=limit,
                ),
            ],
            # RRF ranks by position, so the two lists' different score
            # scales (cosine vs BM25) don't need to be made comparable.
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [SearchHit(chunk=self._to_chunk(p.payload), score=p.score) for p in result.points]

    def count(self) -> int:
        """Return how many chunks are stored (useful for health checks)."""
        return self._client.count(collection_name=self._name).count

    def _ensure_collection(self, dense_dim: int) -> None:
        """Create the collection if missing; check its size if it exists."""
        if self._client.collection_exists(self._name):
            existing = self._client.get_collection(self._name).config.params.vectors[_DENSE].size
            if existing != dense_dim:
                raise ValueError(
                    f"Collection '{self._name}' stores {existing}-dim vectors but the "
                    f"embedder makes {dense_dim}-dim ones. Delete the data/qdrant folder "
                    "and re-upload your documents."
                )
            return
        self._client.create_collection(
            collection_name=self._name,
            vectors_config={_DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            # IDF is computed by Qdrant across the whole collection; fastembed's
            # BM25 only supplies term frequencies, so this completes the formula.
            sparse_vectors_config={_SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        logger.info("Created collection '%s'", self._name)

    def _delete_files(self, file_names: set[str]) -> None:
        """Remove every point belonging to the given files."""
        self._client.delete(
            collection_name=self._name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="file_name", match=models.MatchAny(any=list(file_names)))]
                )
            ),
        )

    @staticmethod
    def _to_payload(chunk: Chunk) -> dict:
        return {
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "index": chunk.index,
            "text": chunk.text,
        }

    @staticmethod
    def _to_chunk(payload: dict) -> Chunk:
        return Chunk(
            file_name=payload["file_name"],
            page_number=payload["page_number"],
            index=payload["index"],
            text=payload["text"],
        )