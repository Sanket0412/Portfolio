"""Load portfolio source documents for retrieval.

Ported from the legacy ``components/llm/rag.py``. Builds a list of
``Document`` objects (each tagged with a ``source`` in metadata) from:
  - curated interview Q&A JSON (one Document per item, kept whole)
  - LinkedIn PDF, resume PDF
  - personal background summary
  - the three project PDFs

Content directories are resolved from ``REPO_ROOT`` in :mod:`portfolio_api.config`.

Two fixes carried over from the legacy code:
  - the resume filename now defaults to the real file (``Sanket_Shah_Resume.pdf``),
    so the resume is actually ingested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pdfplumber
from langchain_core.documents import Document

from portfolio_api.config import REPO_ROOT
from portfolio_api.rag.sanitize import sanitize_retrieved_text

PROFILE_DIR = REPO_ROOT / "content" / "profile"
PERSONA_DIR = REPO_ROOT / "content" / "persona"
PROJECTS_DIR = REPO_ROOT / "content" / "projects"

INTERVIEW_QA_DEFAULT_FILENAME = "interview_qa.json"


def read_pdf(pdf_path: Path) -> Optional[str]:
    """Extract text from a PDF with pdfplumber, or None if it yields nothing."""
    pages: List[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                txt = txt.strip()
                if txt:
                    pages.append(txt)
    except Exception:
        return None

    out = "\n\n".join(pages).strip()
    return out if out else None


def _find_interview_qa_path(filename: str) -> Optional[Path]:
    """Locate interview_qa.json across the common repo layouts.

    Priority: content/persona/ -> content/profile/ -> repo root.
    """
    candidates = [
        PERSONA_DIR / filename,
        PROFILE_DIR / filename,
        REPO_ROOT / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_interview_qa_docs(filename: str, *, max_chars_per_doc: int) -> List[Document]:
    """Load curated interview Q&A as individual Documents.

    Each Q&A becomes one Document (tagged ``interview_qa``) so retrieval can
    return a complete vetted answer rather than forcing the LLM to invent one.
    """
    path = _find_interview_qa_path(filename)
    if not path:
        return []

    try:
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return []
        payload = json.loads(raw)
    except Exception:
        return []

    items = payload.get("items", [])
    docs: List[Document] = []

    for it in items:
        qid = str(it.get("id", "")).strip()
        question = str(it.get("question", "")).strip()
        answer_short = str(it.get("answer_short", "")).strip()
        answer_long = str(it.get("answer_long", "")).strip()
        tags = it.get("tags", []) or []
        sources = it.get("sources", []) or []

        if not question:
            continue

        body = (
            "INTERVIEW_QA\n"
            f"QA_ID: {qid}\n"
            f"Question: {question}\n\n"
            "Vetted answer (short):\n"
            f"{answer_short}\n\n"
            "Vetted answer (long):\n"
            f"{answer_long}\n\n"
            f"Tags: {', '.join([str(t) for t in tags])}\n"
            f"Origin sources: {', '.join([str(s) for s in sources])}\n"
        )

        safe = sanitize_retrieved_text(body, max_chars=max_chars_per_doc)
        if not safe:
            continue

        docs.append(
            Document(
                page_content=safe,
                metadata={
                    "source": "interview_qa",
                    "qa_id": qid,
                    "tags": tags,
                },
            )
        )

    return docs


def load_profile_context(
    linkedin_pdf_name: str = "linkedin.pdf",
    resume_pdf_name: str = "Sanket_Shah_Resume.pdf",
    persona_summary: str = "summary.txt",
    interview_qa_filename: str = INTERVIEW_QA_DEFAULT_FILENAME,
    wpp_media_projects: str = "WPP_Media_Projects.pdf",
    third_estate_ventures_projects: str = "Third_Estate_Ventures_Projects.pdf",
    cloudserve_projects: str = "Cloudserve_Projects.pdf",
    max_chars_per_doc: int = 20000,
) -> List[Document]:
    """Load portfolio docs and return Documents for retrieval.

    Includes interview Q&A, LinkedIn + resume PDFs, the persona summary, and the
    three project PDFs. Raises FileNotFoundError if nothing usable is found.
    """
    linkedin_path = PROFILE_DIR / linkedin_pdf_name
    resume_path = PROFILE_DIR / resume_pdf_name

    # Be robust: summary might be in content/persona in some repo trees
    persona_path_candidates = [
        PERSONA_DIR / persona_summary,
        PROFILE_DIR / persona_summary,
    ]

    wpp_media_projects_path = PROJECTS_DIR / wpp_media_projects
    third_estate_ventures_projects_path = PROJECTS_DIR / third_estate_ventures_projects
    cloudserve_projects_path = PROJECTS_DIR / cloudserve_projects

    docs: List[Document] = []

    # Curated interview Q&A first, so it is easy to retrieve for common interview questions
    docs.extend(
        load_interview_qa_docs(interview_qa_filename, max_chars_per_doc=max_chars_per_doc)
    )

    if linkedin_path.exists():
        linkedin_text = read_pdf(linkedin_path)
        if linkedin_text:
            safe = sanitize_retrieved_text(linkedin_text, max_chars=max_chars_per_doc)
            if safe:
                docs.append(Document(page_content=safe, metadata={"source": "linkedin_pdf"}))

    if resume_path.exists():
        resume_text = read_pdf(resume_path)
        if resume_text:
            safe = sanitize_retrieved_text(resume_text, max_chars=max_chars_per_doc)
            if safe:
                docs.append(Document(page_content=safe, metadata={"source": "resume_pdf"}))

    for persona_path in persona_path_candidates:
        if persona_path.exists():
            persona_text = persona_path.read_text(encoding="utf-8", errors="ignore").strip()
            if persona_text:
                safe = sanitize_retrieved_text(persona_text, max_chars=max_chars_per_doc)
                if safe:
                    docs.append(
                        Document(page_content=safe, metadata={"source": "persona_summary"})
                    )
            break

    if wpp_media_projects_path.exists():
        wpp_text = read_pdf(wpp_media_projects_path)
        if wpp_text:
            safe = sanitize_retrieved_text(wpp_text, max_chars=max_chars_per_doc)
            if safe:
                docs.append(Document(page_content=safe, metadata={"source": "wpp_media_projects"}))

    if third_estate_ventures_projects_path.exists():
        tev_text = read_pdf(third_estate_ventures_projects_path)
        if tev_text:
            safe = sanitize_retrieved_text(tev_text, max_chars=max_chars_per_doc)
            if safe:
                docs.append(
                    Document(
                        page_content=safe,
                        metadata={"source": "third_estate_ventures_projects"},
                    )
                )

    if cloudserve_projects_path.exists():
        cloudserve_text = read_pdf(cloudserve_projects_path)
        if cloudserve_text:
            safe = sanitize_retrieved_text(cloudserve_text, max_chars=max_chars_per_doc)
            if safe:
                docs.append(
                    Document(page_content=safe, metadata={"source": "cloudserve_projects"})
                )

    if not docs:
        raise FileNotFoundError(
            "No usable profile docs found. "
            f"Checked: {PROFILE_DIR}, {PERSONA_DIR}, {PROJECTS_DIR}. "
            f"Expected: {linkedin_pdf_name}, {resume_pdf_name}, {persona_summary}, "
            f"{interview_qa_filename}, {wpp_media_projects}, "
            f"{third_estate_ventures_projects}, {cloudserve_projects}"
        )

    return docs
