"""Light-touch sanitization of retrieved text.

Ported verbatim from the legacy ``components/llm/rag.py``. The goal is to reduce
prompt-injection risk while preserving facts: strip control characters and drop
lines that look like instruction hijacks. Retrieved context is always treated as
data, never instructions (the answer prompt enforces the same posture).
"""
from __future__ import annotations

import re
from typing import List

_INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions",
    r"system prompt",
    r"developer message",
    r"reveal (the )?hidden",
    r"you are chatgpt",
    r"do anything now",
    r"jailbreak",
    r"bypass",
]


def sanitize_retrieved_text(text: str, *, max_chars: int) -> str:
    """Sanitize retrieved text to reduce prompt injection risk while preserving facts.

    Removes control characters, normalizes whitespace, drops obvious injection
    lines, and caps the result at ``max_chars``.
    """
    if not text:
        return ""

    # Normalize whitespace and remove non-printable control characters
    text = text.replace("\x00", " ")
    text = re.sub(r"[\x01-\x08\x0B-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Remove lines that look like instruction hijacks
    lines = text.splitlines()
    kept: List[str] = []
    for ln in lines:
        low = ln.strip().lower()
        if any(re.search(p, low) for p in _INJECTION_PATTERNS):
            continue
        kept.append(ln)

    out = "\n".join(kept).strip()
    return out[:max_chars]
