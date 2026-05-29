"""Chunk documents for embedding.

Ported from the legacy ``components/llm/rag.py``. Long documents are chunked with
a tiktoken-based splitter (1200 / 200), but curated interview Q&A items are kept
whole so retrieval returns a full vetted answer block.
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_docs(docs: List[Document]) -> List[Document]:
    """Split long documents into chunks, keeping interview Q&A items intact."""
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=100,
    )

    split_out: List[Document] = []
    for d in docs:
        src = (d.metadata or {}).get("source", "")
        if src == "interview_qa":
            split_out.append(d)
        else:
            split_out.extend(text_splitter.split_documents([d]))
    return split_out
