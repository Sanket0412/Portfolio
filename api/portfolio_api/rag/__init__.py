"""RAG layer: loaders, sanitizer, splitter, and pgvector retriever."""

from portfolio_api.rag.loaders import (
    INTERVIEW_QA_DEFAULT_FILENAME,
    load_interview_qa_docs,
    load_profile_context,
    read_pdf,
)
from portfolio_api.rag.retriever import (
    DEFAULT_K,
    DEFAULT_K_DOCS,
    DEFAULT_K_QA,
    format_docs,
    get_retriever,
    retrieve,
)
from portfolio_api.rag.sanitize import sanitize_retrieved_text
from portfolio_api.rag.splitter import split_docs

__all__ = [
    "INTERVIEW_QA_DEFAULT_FILENAME",
    "load_interview_qa_docs",
    "load_profile_context",
    "read_pdf",
    "DEFAULT_K",
    "DEFAULT_K_DOCS",
    "DEFAULT_K_QA",
    "format_docs",
    "get_retriever",
    "retrieve",
    "sanitize_retrieved_text",
    "split_docs",
]
