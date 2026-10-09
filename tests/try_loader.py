"""Quick manual check of PdfLoader using a generated sample PDF."""

import logging
import tempfile
from pathlib import Path

import pymupdf

from app.ingestion.loader import PdfLoader

logging.basicConfig(level=logging.INFO)  # so we see the loader's log line

with tempfile.TemporaryDirectory() as tmp:
    sample = Path(tmp) / "return_policy.pdf"

    # Build a 3-page PDF: page 2 is blank on purpose.
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Items can be returned within 14 days.")
    doc.new_page()
    doc.new_page().insert_text((72, 72), "Rental gear must be cleaned before return.")
    doc.save(sample)
    doc.close()

    loader = PdfLoader()
    for page in loader.load(sample):
        print(page)

    # Fail-fast checks
    for bad in [Path(tmp) / "missing.pdf", Path(tmp) / "notes.txt"]:
        bad.with_suffix(".txt").write_text("hi") if bad.suffix == ".txt" else None
        try:
            loader.load(bad)
        except (FileNotFoundError, ValueError) as e:
            print("Caught ✅", type(e).__name__, "-", e)