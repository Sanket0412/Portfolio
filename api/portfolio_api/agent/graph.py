"""ReAct tool-using agent (Phase 2a).

Replaces the Phase 1 linear ``rewrite -> retrieve -> answer`` chain with a LangGraph
prebuilt ``create_react_agent``. Retrieval is now a tool the model decides to call
(see :mod:`portfolio_api.agent.tools`), which lets it issue multiple focused
queries for compound questions and sets the foundation for the Phase 2b tools.

The public entrypoint ``answer(question, history, session_id) -> AgentResult`` is
unchanged so the Streamlit harness and ``parity_eval.py`` keep working. Guardrails
are still applied centrally here, and answers now carry structured ``citations``.

Per-session memory (a LangGraph checkpointer) is Phase 2d; for now chat history is
passed in by the caller and threaded through for the single turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from portfolio_api.agent.prompts import build_system_prompt
from portfolio_api.agent.tools import (
    AGENT_TOOLS,
    Citation,
    get_citations,
    reset_citations,
)
from portfolio_api.guardrails import (
    BLOCKED_REPLY,
    EMPTY_CONTEXT_REPLY,
    GREETING_REPLY,
    MEMORY_REPLY,
    is_greeting,
    is_memory_like,
    redact_secrets,
    should_block_user_input,
    strip_em_dashes,
    truncate_answer,
)
from portfolio_api.llm import get_chat_model


@dataclass
class AgentResult:
    """Result of one agent turn."""

    answer: str
    rewritten_question: str = ""
    sources: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)


def _prompt(state) -> List[BaseMessage]:
    """Prepend a freshly-built system prompt (current datetime) to the messages."""
    return [SystemMessage(content=build_system_prompt())] + list(state["messages"])


@lru_cache(maxsize=2)
def build_agent(streaming: bool = False):
    """Compile the ReAct agent (cached per streaming flag).

    ``streaming=True`` is used by the SSE endpoint so token deltas are emitted; the
    in-process ``answer()`` path uses the non-streaming agent.
    """
    model = get_chat_model(streaming=streaming)
    return create_react_agent(model, AGENT_TOOLS, prompt=_prompt)


def message_text(message: BaseMessage) -> str:
    """Extract plain text from a message whose content may be a list of blocks.

    Anthropic returns content as a list of typed blocks; OpenAI returns a string.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _fast_path_reply(question: str) -> Optional[str]:
    """Return a canned reply for guardrail fast-paths, or None to run the agent."""
    if should_block_user_input(question):
        return BLOCKED_REPLY
    if is_greeting(question):
        return GREETING_REPLY
    if is_memory_like(question):
        return MEMORY_REPLY
    return None


def _finalize(text: str) -> str:
    """Apply the output guardrails: empty fallback, redact, no em-dashes, truncate."""
    text = (text or "").strip() or EMPTY_CONTEXT_REPLY
    return truncate_answer(strip_em_dashes(redact_secrets(text)))


def answer(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    session_id: Optional[str] = None,
) -> AgentResult:
    """Answer a question as Sanket, grounded in retrieved portfolio context.

    Guardrails are applied here so any caller (Streamlit harness now, FastAPI
    ``/chat`` later) gets the same behavior. ``session_id`` is accepted for
    forward-compatibility but unused until the Phase 2d checkpointer.
    """
    question = (question or "").strip()
    history = history or []

    canned = _fast_path_reply(question)
    if canned is not None:
        return AgentResult(answer=canned)

    reset_citations()
    messages = list(history) + [HumanMessage(content=question)]
    final = build_agent().invoke({"messages": messages})

    last = final["messages"][-1]
    text = _finalize(message_text(last))
    citations = get_citations()

    return AgentResult(
        answer=text,
        sources=sorted({c.source for c in citations}),
        citations=citations,
    )
