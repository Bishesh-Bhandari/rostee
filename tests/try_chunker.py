"""Quick manual check of TextChunker."""

import logging

from app.ingestion.chunker import TextChunker
from app.schemas import Page

logging.basicConfig(level=logging.INFO)

paragraph = (
    "Items can be returned within 14 days of purchase. "
    "The receipt is required for every return. "
    "Used boots are accepted if they show light wear only. "
)
long_page = Page(file_name="return_policy.pdf", page_number=3, text=paragraph * 6)
short_page = Page(file_name="return_policy.pdf", page_number=4, text="Rentals: see page 5.")

chunker = TextChunker(chunk_size=200, chunk_overlap=50)
chunks = chunker.chunk([long_page, short_page])

for c in chunks:
    print(f"p{c.page_number} #{c.index} ({len(c.text)} chars): {c.text[:60]}...")

# Overlap check: the end of chunk 0 should show up at the start of chunk 1.
print("\nEnd of #0:  ", repr(chunks[0].text[-40:]))
print("Start of #1:", repr(chunks[1].text[:40]))

try:
    TextChunker(chunk_size=200, chunk_overlap=300)
except ValueError as e:
    print("\nBad sizes caught ✅", e)