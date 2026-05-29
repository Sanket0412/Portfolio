"""Embeddings factory.

Embeddings are pinned to OpenAI so the pgvector column dimension stays stable
(text-embedding-3-small -> 1536). Changing the model/dim requires re-ingesting.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from portfolio_api.config import get_settings


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    settings.require("openai_api_key")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
