"""
Centralised configuration via pydantic-settings.
All values read from environment / .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    openai_api_key: str = "mock-key"
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536  # 3-small default; 768 for nomic

    # ── Dev / Mock ───────────────────────────────────────────────────────────
    mock_mode: bool = False

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "rag_documents"

    # ── Structured DB ────────────────────────────────────────────────────────
    sqlite_db_path: str = "./data/structured.db"

    # ── Ingestion ────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_file_size_mb: int = 50
    upload_dir: str = "./uploads"

    # ── Agent ────────────────────────────────────────────────────────────────
    agent_max_iterations: int = 10
    retrieval_top_k: int = 10
    confidence_threshold: float = 0.25

    # ── Guardrails ───────────────────────────────────────────────────────────
    enable_pii_redaction: bool = False
    enable_injection_guard: bool = True

    # ── API ──────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    @field_validator("mock_mode", mode="before")
    @classmethod
    def parse_bool(cls, v):
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        for d in [
            self.chroma_persist_dir,
            os.path.dirname(self.sqlite_db_path),
            self.upload_dir,
            "./logs",
        ]:
            os.makedirs(d, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s