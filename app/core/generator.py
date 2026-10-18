"""Generation: asks the LLM to answer a question using only the retrieved chunks."""

import logging
import re

from google import genai
from google.genai import errors, types

from app.schemas import Answer, Chunk, SearchHit

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful assistant for a business. Answer the user's question using \
ONLY the numbered sources provided.

Rules:
- After every fact, cite the source it came from as [1], [2], etc.
- If the sources don't contain the answer, say you couldn't find it in the \
documents. Never guess or use outside knowledge.
- For greetings or small talk, reply briefly and politely without citations.
- Keep answers short and direct. Answer in the same language as the question.
"""

# Matches citation markers like [1] or [12].
_CITATION = re.compile(r"\[(\d+)\]")


class Generator:
    """Writes the final answer with Gemini, grounded in the reranked chunks.

    Chunks are shown to the model as numbered sources. The numbers the model
    actually cites decide which chunks are returned as sources, so the user
    only sees citations for what the answer really used.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """Create the Gemini client.

        Args:
            api_key: Gemini API key.
            model: Gemini model name, e.g. "gemini-2.5-flash".
        """
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        """Answer a question from the given chunks.

        Args:
            question: The user's question.
            hits: Best chunks from the reranker, best first.

        Returns:
            The answer text and the chunks it cited. Sources are empty when
            the model cited nothing (small talk or "not found").

        Raises:
            RuntimeError: If the Gemini API call fails (bad key, quota, outage).
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=self._build_prompt(question, hits),
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    # Low temperature: we want faithful answers, not creative ones.
                    temperature=0.2,
                ),
            )
        except errors.APIError as e:
            raise RuntimeError(
                f"Gemini request failed ({e.code}): {e.message}. Check GEMINI_API_KEY "
                "and your quota."
            ) from e

        text = (response.text or "").strip()
        if not text:
            # Happens when Gemini's safety filter blocks the reply.
            text = "Sorry, I couldn't generate an answer to that. Try rephrasing the question."
        sources = self._cited_sources(text, hits)
        logger.info("Generated answer citing %d of %d source(s)", len(sources), len(hits))
        return Answer(text=text, sources=sources)

    @staticmethod
    def _build_prompt(question: str, hits: list[SearchHit]) -> str:
        """Lay out the sources as a numbered list, then the question."""
        if not hits:
            return f"Sources: (none found)\n\nQuestion: {question}"
        sources = "\n\n".join(
            f"[{n}] ({hit.chunk.file_name}, page {hit.chunk.page_number})\n{hit.chunk.text}"
            for n, hit in enumerate(hits, start=1)
        )
        return f"Sources:\n\n{sources}\n\nQuestion: {question}"

    @staticmethod
    def _cited_sources(text: str, hits: list[SearchHit]) -> tuple[Chunk, ...]:
        """Return the chunks the answer cited, in source order, without duplicates.

        Numbers outside the source list (a model hallucinating [9] when only
        5 sources exist) are ignored.
        """
        cited = {int(n) for n in _CITATION.findall(text)}
        return tuple(hit.chunk for n, hit in enumerate(hits, start=1) if n in cited)
