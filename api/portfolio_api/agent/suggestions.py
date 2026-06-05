"""Follow-up question suggestions (Phase 2d).

After each real answer, propose 2-3 short follow-up questions the visitor might ask
next, grounded in Sanket's portfolio (roles, projects, skills, education,
publications). This is the recruiter-facing half of Phase 2d; per-session memory via
a LangGraph checkpointer was deliberately deferred (client-owned history is the right
pattern for this app and the planned Vercel-AI-SDK frontend - see docs/grand_plan.md).

Design notes:
  - Suggestions are **best-effort**: any failure returns ``[]`` and never breaks the
    answer. They are a separate, cheap LLM call (low temperature, small max_tokens).
  - Output is **content-policy filtered** via :func:`portfolio_api.rag.is_excluded`,
    reusing the same exclusion the ingest sanitizer and the GitHub tool honor, so an
    off-limits topic can never be surfaced as a suggested question.
  - Fast-path turns (greeting / cold start) get a fixed ``STARTER_SUGGESTIONS`` set with
    no LLM call; blocked / memory turns get none.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from portfolio_api.llm import get_chat_model
from portfolio_api.rag import is_excluded

MAX_SUGGESTIONS = 3

# Shown on a greeting / cold start, where there is no answer to branch from yet.
STARTER_SUGGESTIONS = [
    "What did you build at WPP Media?",
    "Tell me about your publications.",
    "What personal projects are you working on?",
]

_SUGGESTION_SYSTEM = (
    "You generate short follow-up questions a visitor might ask next on Sanket J Shah's "
    "portfolio. Every question must be answerable from Sanket's portfolio (his roles, "
    "projects, skills, education, and publications) and phrased in the second person, as "
    "if the visitor is asking Sanket directly. Keep each under about 12 words. Return ONLY "
    "a JSON array of 2 to 3 question strings, with no surrounding prose or code fences."
)

# Strip a leading/trailing ```json ... ``` fence some models add around JSON.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


@lru_cache(maxsize=1)
def _suggestion_model() -> BaseChatModel:
    """Cheap, low-temperature model for the suggestion call (cached)."""
    return get_chat_model(temperature=0.3, max_tokens=256)


def _build_user_prompt(question: str, answer: str, sources: List[str]) -> str:
    src = ", ".join(sources) if sources else "none"
    return (
        f"The visitor just asked: {question}\n\n"
        f"Sanket answered (portfolio sources used: {src}):\n{answer}\n\n"
        "Suggest 2 to 3 natural follow-up questions that go deeper or branch into adjacent "
        "topics in Sanket's portfolio. Do not repeat what was already answered. "
        "Return a JSON array of strings only."
    )


def _parse(raw: str) -> List[str]:
    """Parse the model output into a clean, de-duplicated, exclusion-filtered list."""
    raw = _FENCE_RE.sub("", (raw or "").strip())
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []

    seen: set = set()
    out: List[str] = []
    for item in data:
        s = str(item).strip()
        if not s or s.lower() in seen or is_excluded(s):
            continue
        seen.add(s.lower())
        out.append(s)
    return out[:MAX_SUGGESTIONS]


def generate_suggestions(
    question: str,
    answer: str,
    sources: Optional[List[str]] = None,
    *,
    max_suggestions: int = MAX_SUGGESTIONS,
) -> List[str]:
    """Return up to ``max_suggestions`` grounded follow-up questions (best-effort).

    Never raises: any error (model, parsing) yields ``[]`` so the answer is unaffected.
    """
    if not (answer or "").strip():
        return []

    # Lazy import avoids a circular import at module load (graph imports this module).
    from portfolio_api.agent.graph import message_text

    try:
        resp = _suggestion_model().invoke(
            [
                SystemMessage(content=_SUGGESTION_SYSTEM),
                HumanMessage(content=_build_user_prompt(question, answer, sources or [])),
            ]
        )
        return _parse(message_text(resp))[:max_suggestions]
    except Exception:  # best-effort: suggestions must never break the answer
        return []
