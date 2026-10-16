"""Quick manual check of Reranker: the truly relevant chunk should move to the top."""

import logging

from app.config import get_settings
from app.core.reranker import Reranker
from app.schemas import Chunk, SearchHit

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

s = get_settings()
reranker = Reranker(s.reranker_model, s.models_cache_dir)

# Deliberately put the right answer LAST, as if search ranked it poorly.
hits = [
    SearchHit(Chunk("price_list.pdf", 1, 0, "Sleeping bag SB-15X rents for $4 per day."), 0.9),
    SearchHit(Chunk("return_policy.pdf", 2, 0, "Used boots are accepted if they show light wear only."), 0.8),
    SearchHit(Chunk("price_list.pdf", 1, 1, "Trekking poles rent for $1 per day."), 0.7),
    SearchHit(Chunk("return_policy.pdf", 1, 0, "Customers can get their money back within 14 days."), 0.6),
]

question = "how long do I have to get a refund?"
print(f"Q: {question}")
for hit in reranker.rerank(question, hits, top_k=2):
    c = hit.chunk
    print(f"  {hit.score:7.3f}  [{c.file_name} p{c.page_number}]  {c.text}")
print("Expected: the 14-day money-back chunk first.")
print("Empty input ->", reranker.rerank(question, [], top_k=2))
