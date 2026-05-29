"""Chat guardrails, ported from the legacy ``pages/4_Chat.py``.

These are pure functions with no Streamlit dependency, so the agent entrypoint
(:func:`portfolio_api.agent.answer`) can apply them centrally and the future
FastAPI ``/chat`` endpoint inherits the same behavior. Rate limiting is a
server-side concern and is intentionally NOT here (deferred to cutover).
"""
from __future__ import annotations

import re

# Canned replies (skip the LLM for these fast-paths)
GREETING_REPLY = "Hello! Feel free to ask me about my projects, experience, or publications."
MEMORY_REPLY = (
    "I cannot store personal preferences or remember details across sessions. "
    "If you have questions about my background, skills, or projects, feel free to ask."
)
BLOCKED_REPLY = (
    "I cannot help with that request. "
    "If you have questions about my background, projects, or experience, ask away."
)
EMPTY_CONTEXT_REPLY = (
    "I do not have enough information in the current context to answer that."
)

# Output length cap (characters)
MAX_ANSWER_CHARS = 3500

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good (morning|afternoon|evening))\s*[!.]*\s*$",
    re.IGNORECASE,
)

_BLOCK_PATTERNS = [
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
    r"\breveal\b.*\b(api key|secret|token|credentials)\b",
    r"\bignore (all|any|previous) instructions\b",
    r"\bjailbreak\b",
    r"\bbypass\b",
]

_SECRET_LIKE_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",                 # common OpenAI key shape
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # private keys
    r"AKIA[0-9A-Z]{16}",                    # AWS Access Key ID
]


def is_greeting(text: str) -> bool:
    return bool(_GREETING_RE.match(text or ""))


def is_memory_like(text: str) -> bool:
    low = (text or "").strip().lower()
    return low.startswith("remember that ") or low.startswith("remember ")


def should_block_user_input(text: str) -> bool:
    low = (text or "").strip().lower()
    if not low:
        return False
    return any(re.search(p, low) for p in _BLOCK_PATTERNS)


def redact_secrets(text: str) -> str:
    out = text or ""
    for p in _SECRET_LIKE_PATTERNS:
        out = re.sub(p, "[REDACTED]", out)
    return out


def strip_em_dashes(text: str) -> str:
    """Enforce the persona's no-em-dash rule (a safety net for models, e.g. Claude,
    that favor em-dashes despite the prompt). Em/en dashes become hyphens, which the
    style guide explicitly allows; spaced dashes collapse to a single ", "."""
    if not text:
        return text
    out = re.sub(r"\s*[—–]\s*", ", ", text)  # spaced em/en dash -> comma
    out = out.replace("—", "-").replace("–", "-")  # any remaining -> hyphen
    return out


def truncate_answer(text: str, *, max_chars: int = MAX_ANSWER_CHARS) -> str:
    if text and len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n\n(Truncated for length.)"
    return text
