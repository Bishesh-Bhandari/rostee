"""Quick manual check of Embedder: shapes, and that meaning beats exact words."""

import logging

import numpy as np

from app.config import get_settings
from app.core.embedder import Embedder

logging.basicConfig(level=logging.INFO)

s = get_settings()
embedder = Embedder(s.dense_model, s.sparse_model, s.dense_dim, s.models_cache_dir)

docs = [
    "Customers can get their money back within 14 days.",
    "Sleeping bag SB-15X rents for $4 per day.",
    "Our shop opens at 9am in Lakeside.",
]
doc_vecs = embedder.embed_documents(docs)
query = embedder.embed_query("what is the refund policy?")

print("Dense length:", len(query.dense))
print("Sparse terms in doc 1:", len(doc_vecs[1].sparse.indices))


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    return float(a_arr @ b_arr / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))


print("\nQuery: 'what is the refund policy?'")
for text, vec in zip(docs, doc_vecs):
    print(f"  {cosine(query.dense, vec.dense):.3f}  {text}")