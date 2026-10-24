"""Quick manual check of RagPipeline using a generated sample PDF."""

import logging
import tempfile
from pathlib import Path

import pymupdf

from app.ingestion.chunker import TextChunker
from app.ingestion.loader import PdfLoader
from app.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO)  # so we see the pipeline's log line

with tempfile.TemporaryDirectory() as tmp:
    sample = Path(tmp) / "return_policy.pdf"

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Items can be returned within 14 days.")
    doc.new_page().insert_text((72, 72), "Rental gear must be cleaned before return.")
    doc.save(sample)
    doc.close()

    # Small chunk size so even this short PDF shows the split + overlap.
    pipeline = RagPipeline(loader=PdfLoader(), chunker=TextChunker(chunk_size=20, chunk_overlap=5))
    for chunk in pipeline.ingest(sample):
        print(chunk)
