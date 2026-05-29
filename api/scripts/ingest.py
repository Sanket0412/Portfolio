"""Ingest portfolio sources into the Supabase pgvector store.

Pipeline:  load_profile_context() -> split_docs() -> embed -> pgvector

Idempotent: each chunk gets a deterministic ID (sha256 of source + index +
content), so ``add_documents(..., ids=...)`` upserts and re-running produces no
duplicates. Pass ``--reset`` to drop and recreate the collection for a clean rebuild.

Run:  python api/scripts/ingest.py [--reset]
Requires api/.env populated (DATABASE_URL, OPENAI_API_KEY).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from typing import List

from langchain_core.documents import Document

from portfolio_api.config import get_settings
from portfolio_api.db import get_vector_store
from portfolio_api.rag import load_profile_context, split_docs


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _doc_id(doc: Document, index: int) -> str:
    src = (doc.metadata or {}).get("source", "unknown")
    digest = hashlib.sha256(
        f"{src}::{index}::{doc.page_content}".encode("utf-8")
    ).hexdigest()
    return f"{src}-{digest[:24]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest portfolio sources into pgvector.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before ingesting (clean rebuild).",
    )
    args = parser.parse_args()

    settings = get_settings()
    settings.require("database_url", "openai_api_key")

    try:
        docs = load_profile_context()
    except FileNotFoundError as e:
        _fail(str(e))
        return 1

    chunks = split_docs(docs)
    if not chunks:
        _fail("No chunks produced from the loaded documents.")
        return 1

    by_source = Counter((c.metadata or {}).get("source", "unknown") for c in chunks)
    _info(f"Loaded {len(docs)} docs -> {len(chunks)} chunks across {len(by_source)} sources:")
    for src, n in sorted(by_source.items()):
        print(f"         {src:<32} {n:>4} chunk(s)")

    store = get_vector_store()

    if args.reset:
        store.delete_collection()
        store.create_collection()
        _ok(f"reset collection '{settings.pgvector_collection}'")

    ids: List[str] = [_doc_id(c, i) for i, c in enumerate(chunks)]
    store.add_documents(chunks, ids=ids)
    _ok(f"upserted {len(chunks)} chunks into '{settings.pgvector_collection}'")

    print()
    _ok("ingestion complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
