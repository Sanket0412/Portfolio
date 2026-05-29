"""Database helpers for the Supabase Postgres + pgvector store.

`DATABASE_URL` is a standard Postgres URL (e.g. the Supabase connection string).
langchain-postgres needs a SQLAlchemy/psycopg3 URL, so we normalize the scheme.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from portfolio_api.config import get_settings


def _raw_libpq_url(url: str) -> str:
    """Plain libpq URL for psycopg.connect (strip any SQLAlchemy driver suffix)."""
    return url.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgres+psycopg://", "postgresql://"
    )


def _sqlalchemy_url(url: str) -> str:
    """SQLAlchemy URL using the psycopg (v3) driver, required by langchain-postgres."""
    if url.startswith("postgresql+psycopg://") or url.startswith("postgres+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg://", 1)
    return url


def get_database_urls() -> tuple[str, str]:
    """Return (libpq_url, sqlalchemy_url). Raises if DATABASE_URL is unset."""
    settings = get_settings()
    settings.require("database_url")
    return _raw_libpq_url(settings.database_url), _sqlalchemy_url(settings.database_url)


def ensure_pgvector() -> None:
    """Enable the pgvector extension (idempotent). Requires sufficient DB privileges."""
    import psycopg

    libpq_url, _ = get_database_urls()
    with psycopg.connect(libpq_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def pgvector_enabled() -> bool:
    """True if the pgvector extension is installed in the target database."""
    import psycopg

    libpq_url, _ = get_database_urls()
    with psycopg.connect(libpq_url) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector';"
        ).fetchone()
    return row is not None


@lru_cache(maxsize=1)
def get_vector_store(collection: str | None = None) -> Any:
    """Return a langchain-postgres PGVector store bound to the embeddings + collection.

    Cached so the SQLAlchemy engine is reused across calls within a process.
    """
    from langchain_postgres import PGVector

    from portfolio_api.llm import get_embeddings

    settings = get_settings()
    _, sa_url = get_database_urls()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection or settings.pgvector_collection,
        connection=sa_url,
        use_jsonb=True,
    )
