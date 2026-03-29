"""
Embedding abstraction with mock fallback.

- Real mode: OpenAI-compatible embeddings via langchain-openai
- Mock mode: Deterministic pseudo-random unit-norm vectors
"""
from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MockEmbedder:
    """
    Deterministic embedder for development.
    Same text always produces the same vector.
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return [self._embed(t) for t in texts]
        except Exception as e:
            logger.error("mock_embed_documents_error", error=str(e))
            return []

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._embed(text)
        except Exception as e:
            logger.error("mock_embed_query_error", error=str(e))
            return []

    def _embed(self, text: str) -> List[float]:
        try:
            # Hash the text to get a deterministic seed
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimensions).astype(np.float32)
            # Normalise to unit norm
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        except Exception as e:
            logger.error("mock_embed_internal_error", error=str(e))
            return [0.0] * int(self.dimensions)


class RealEmbedder:
    """Thin wrapper around langchain-openai embeddings with batching support."""

    # OpenAI embedding API has a token limit per batch; use conservative batch size
    _BATCH_SIZE = 100

    def __init__(self):
        try:
            from langchain_openai import OpenAIEmbeddings
        except Exception as e:
            logger.error("real_embedder_import_error", error=str(e))
            raise

        try:
            settings = get_settings()
            self._client = OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=settings.openai_api_key,
                openai_api_base=settings.openai_base_url,
            )
        except Exception as e:
            logger.error("real_embedder_init_error", error=str(e))
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents in batches to avoid API token limits."""
        try:
            if len(texts) <= self._BATCH_SIZE:
                return self._client.embed_documents(texts)

            # Process in batches
            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[i : i + self._BATCH_SIZE]
                batch_embeddings = self._client.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(
                    "embedding_batch_complete",
                    batch_num=i // self._BATCH_SIZE + 1,
                    batch_size=len(batch),
                    total=len(texts),
                )
            return all_embeddings
        except Exception as e:
            logger.error("real_embed_documents_error", error=str(e))
            raise

    def embed_query(self, text: str) -> List[float]:
        try:
            return self._client.embed_query(text)
        except Exception as e:
            logger.error("real_embed_query_error", error=str(e))
            raise


def get_embedder():
    try:
        settings = get_settings()
        if settings.mock_mode:
            logger.info("embedder_mode", mode="mock")
            return MockEmbedder(dimensions=settings.embedding_dimensions)
        logger.info(
            "embedder_mode",
            mode="openai",
            model=settings.embedding_model,
        )
        return RealEmbedder()
    except Exception as e:
        logger.error("get_embedder_error_fallback_to_mock", error=str(e))
        try:
            settings = get_settings()
            dims = getattr(settings, "embedding_dimensions", 1536)
        except Exception:
            dims = 1536
        return MockEmbedder(dimensions=dims)