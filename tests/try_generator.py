"""Quick manual check of Generator: grounded answer, out-of-scope, and small talk."""

import logging

from app.config import get_settings
from app.core.generator import Generator
from app.schemas import Chunk, SearchHit

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

s = get_settings()
generator = Generator(s.gemini_api_key.get_secret_value(), s.llm_model)

hits = [
    SearchHit(Chunk("return_policy.pdf", 1, 0, "Customers can get their money back within 14 days."), 0.0),
    SearchHit(Chunk("return_policy.pdf", 2, 0, "Used boots are accepted if they show light wear only."), 0.0),
    SearchHit(Chunk("price_list.pdf", 1, 0, "Sleeping bag SB-15X rents for $4 per day."), 0.0),
]

for question in [
    "how long do I have to get a refund?",  # expect answer citing return_policy p1
    "do you sell kayaks?",                  # expect "couldn't find", no sources
    "hi there!",                            # expect a greeting, no sources
]:
    answer = generator.answer(question, hits)
    print(f"\nQ: {question}\nA: {answer.text}")
    for c in answer.sources:
        print(f"   source: {c.file_name} p{c.page_number}")
