"""Application settings — the single source of truth for every tunable value."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for rostee, loaded from environment variables and `.env`.

    Field names map to env vars case-insensitively, e.g. `gemini_api_key`
    reads `GEMINI_API_KEY`. Defaults are tuned for a small business document set
    on a laptop; override any of them in `.env` without touching code.

    Raises:
        pydantic.ValidationError: If a required value is missing or a value is
            out of range (e.g. overlap bigger than chunk size).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Ignore unrelated env vars so a stray variable on a client's server
        # doesn't crash startup.
        extra="ignore",
    )

    # --- LLM ---
    # SecretStr hides the key in logs/print output, so it can't leak into a
    # screenshot or a log file you send to a client.
    gemini_api_key: SecretStr
    llm_model: str = "gemini-2.5-flash"

    # --- Embeddings & reranking ---
    dense_model: str = "BAAI/bge-small-en-v1.5"
    dense_dim: int = Field(default=384, gt=0)
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Chunking (measured in characters, not tokens) ---
    # ~800 chars ≈ 200 tokens: small enough to stay well under the embedder's
    # 512-token limit, big enough to hold a full policy clause.
    chunk_size: int = Field(default=800, gt=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Retrieval ---
    retrieve_top_k: int = Field(default=20, gt=0)  # candidates from hybrid search
    rerank_top_k: int = Field(default=5, gt=0)     # best ones kept for the LLM

    # --- Storage ---
    qdrant_path: Path = Path("data/qdrant")
    collection_name: str = "documents"
    upload_dir: Path = Path("data/uploads")

    @model_validator(mode="after")
    def _check_consistency(self) -> "Settings":
        """Catch settings that are valid alone but broken together."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size}), otherwise chunking never moves "
                "forward. Fix CHUNK_OVERLAP or CHUNK_SIZE in .env."
            )
        if self.rerank_top_k > self.retrieve_top_k:
            raise ValueError(
                f"rerank_top_k ({self.rerank_top_k}) can't be larger than "
                f"retrieve_top_k ({self.retrieve_top_k}): the reranker can only "
                "keep results that search actually returned."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Build Settings once and reuse it.

    Returns:
        The shared Settings instance.

    Raises:
        pydantic.ValidationError: If `.env` is missing required values.
    """
    return Settings()