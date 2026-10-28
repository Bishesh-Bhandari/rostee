"""Quick manual check of RagPipeline using a generated sample PDF."""

import logging
import tempfile
from pathlib import Path

import pymupdf

from app.config import get_settings
from app.core.embedder import Embedder
from app.core.store import QdrantStore
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import PdfLoader
from app.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO)  # so we see the pipeline's log line

settings = get_settings()

# A separate collection so test runs never mix with real uploaded documents.
store = QdrantStore(settings.qdrant_path, "test_pipeline", settings.dense_dim)

with tempfile.TemporaryDirectory() as tmp:
    sample = Path(tmp) / "return_policy.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Items can be returned within 14 days.")
    doc.new_page().insert_text((72, 72), "Rental gear must be cleaned before return.")
    doc.save(sample)
    doc.close()

    # Small chunk size so even this short PDF shows the split + overlap.
    # The embedder is the real one, so the first run downloads its models.
    pipeline = RagPipeline(
        loader=PdfLoader(),
        chunker=TextChunker(chunk_size=20, chunk_overlap=5),
        embedder=Embedder(
            dense_model=settings.dense_model,
            sparse_model=settings.sparse_model,
            dense_dim=settings.dense_dim,
            cache_dir=settings.models_cache_dir,
        ),
        store=store,
    )

    stored = pipeline.ingest(sample)
    assert stored == 6, f"expected 6 chunks stored, got {stored}"
    assert store.count() == 6, f"expected 6 points in the store, got {store.count()}"

    # Ingesting the same file again must replace its chunks, not duplicate them.
    pipeline.ingest(sample)
    assert store.count() == 6, f"re-ingest duplicated chunks: {store.count()} points"
    print("OK: all checks passed")
