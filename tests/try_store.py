"""Quick manual check of QdrantStore: add, hybrid search, and clean re-upload."""

import logging

from app.config import get_settings
from app.core.embedder import Embedder
from app.core.store import QdrantStore
from app.schemas import Chunk

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

s = get_settings()
embedder = Embedder(s.dense_model, s.sparse_model, s.dense_dim, s.models_cache_dir)
store = QdrantStore(s.qdrant_path, "test_documents", s.dense_dim)

chunks = [
    Chunk("return_policy.pdf", 1, 0, "Customers can get their money back within 14 days."),
    Chunk("return_policy.pdf", 2, 0, "Used boots are accepted if they show light wear only."),
    Chunk("price_list.pdf", 1, 0, "Sleeping bag SB-15X rents for $4 per day."),
    Chunk("price_list.pdf", 1, 1, "Trekking poles rent for $1 per day."),
]
store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
print("Stored:", store.count())

for question in ["what is the refund policy?", "price of SB-15X"]:
    print(f"\nQ: {question}")
    for hit in store.search(embedder.embed_query(question), limit=3):
        c = hit.chunk
        print(f"  {hit.score:.3f}  [{c.file_name} p{c.page_number}]  {c.text}")

# Re-upload a SHORTER price list: the old trekking-pole chunk must disappear.
new_prices = [Chunk("price_list.pdf", 1, 0, "Sleeping bag SB-15X now rents for $5 per day.")]
store.add(new_prices, embedder.embed_documents([c.text for c in new_prices]))
print("\nAfter re-upload, stored:", store.count(), "(expected 3)")