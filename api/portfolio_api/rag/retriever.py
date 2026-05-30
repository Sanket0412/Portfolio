"""Retrieval against the persistent Supabase pgvector store.

Replaces the legacy in-memory vector store. The retriever is backed by the cached
``get_vector_store()`` helper in :mod:`portfolio_api.db`; documents must be ingested
first (see ``api/scripts/ingest.py``).

``format_docs`` is ported from the legacy ``components/llm/chain.py`` and builds a
bounded ``[SOURCE=...]`` context block. The model must treat this as data only,
never as instructions.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from langchain_core.documents import Document

from portfolio_api.db import get_vector_store

DEFAULT_K = 6
# Hybrid retrieval pools (see `retrieve`).
DEFAULT_K_QA = 4
DEFAULT_K_DOCS = 6


def get_retriever(k: int = DEFAULT_K) -> Any:
    """Return a plain top-k retriever over the pgvector collection.

    Kept for ad-hoc use; the agent uses :func:`retrieve` for balanced recall.
    """
    return get_vector_store().as_retriever(search_kwargs={"k": k})


def retrieve(
    query: str,
    *,
    k_qa: int = DEFAULT_K_QA,
    k_docs: int = DEFAULT_K_DOCS,
) -> List[Document]:
    """Balanced hybrid retrieval over two pools.

    The 40 curated ``interview_qa`` blocks embed the question text itself, so on a
    single similarity search they crowd out every source PDF. We instead retrieve
    from two pools separately and merge: the vetted Q&A pool (leads, so the model
    prefers vetted answers) and the source-document pool (supplies depth and
    specifics). This guarantees the rich PDF detail is always available.
    """
    vs = get_vector_store()
    qa = vs.similarity_search(query, k=k_qa, filter={"source": {"$eq": "interview_qa"}})
    other = vs.similarity_search(query, k=k_docs, filter={"source": {"$ne": "interview_qa"}})

    seen: set = set()
    merged: List[Document] = []
    for d in (*qa, *other):
        key = ((d.metadata or {}).get("source"), (d.page_content or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged


def retrieve_with_scores(
    query: str,
    *,
    k_qa: int = DEFAULT_K_QA,
    k_docs: int = DEFAULT_K_DOCS,
) -> List[Tuple[Document, float]]:
    """Like :func:`retrieve`, but pairs each document with its similarity score.

    Scores come from ``PGVector.similarity_search_with_score`` (cosine distance:
    lower is closer). Used by the agent's retrieval tool to emit citations. The
    two-pool merge and de-duplication mirror :func:`retrieve` exactly.
    """
    vs = get_vector_store()
    qa = vs.similarity_search_with_score(
        query, k=k_qa, filter={"source": {"$eq": "interview_qa"}}
    )
    other = vs.similarity_search_with_score(
        query, k=k_docs, filter={"source": {"$ne": "interview_qa"}}
    )

    seen: set = set()
    merged: List[Tuple[Document, float]] = []
    for d, score in (*qa, *other):
        key = ((d.metadata or {}).get("source"), (d.page_content or "")[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append((d, float(score)))
    return merged


def format_docs(docs: List[Document], *, max_total_chars: int = 14000) -> str:
    """Format retrieved docs into a safe, bounded context block.

    Important: the model must treat this as data only, never as instructions.
    """
    if not docs:
        return ""

    blocks: List[str] = []
    used = 0

    for d in docs:
        src = (d.metadata or {}).get("source", "unknown_source")
        text = (d.page_content or "").strip()
        if not text:
            continue

        block = f"[SOURCE={src}]\n{text}\n"
        if used + len(block) > max_total_chars:
            remaining = max_total_chars - used
            if remaining <= 0:
                break
            blocks.append(block[:remaining])
            used += remaining
            break

        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks).strip()
