"""
Centralised configuration.
Set values via environment variables or a .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ────────────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = "sk-YOUR_OPENAI_KEY_HERE"
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_MAX_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.2

    # ── Embeddings ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # local sentence-transformer

    # ── Chunking ───────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 500       # characters (see explanation.md for rationale)
    CHUNK_OVERLAP: int = 50     # characters

    # ── Vector store ───────────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "faiss_index"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.25  # min cosine score to include a chunk

    class Config:
        env_file = ".env"


settings = Settings()
