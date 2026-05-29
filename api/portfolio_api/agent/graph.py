"""Minimal LangGraph agent: rewrite -> retrieve -> answer.

This is the Phase 1 parity port of the legacy ``build_rag_chain_with_history``.
It uses the persistent pgvector retriever instead of an in-memory store, and
centralizes the chat guardrails so the future FastAPI ``/chat`` endpoint inherits
them.

Memory / per-session checkpointing is deliberately out of scope (Phase 2): chat
history is passed in by the caller and threaded through the graph for this one turn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from portfolio_api.agent.prompts import (
    CONTEXTUALIZE_PROMPT,
    build_answer_prompt,
    build_system_prompt,
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
from portfolio_api.rag import format_docs, retrieve


@dataclass
class AgentResult:
    """Result of one agent turn."""

    answer: str
    rewritten_question: str = ""
    sources: List[str] = field(default_factory=list)


class _AgentState(TypedDict, total=False):
    question: str
    history: List[BaseMessage]
    rewritten: str
    context: str
    sources: List[str]
    answer: str


@lru_cache(maxsize=1)
def _build_graph():
    """Compile the rewrite -> retrieve -> answer graph (cached per process)."""
    llm = get_chat_model()  # provider/model from config (api/.env)
    rewrite_chain = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
    answer_chain = build_answer_prompt(build_system_prompt()) | llm | StrOutputParser()

    def rewrite_node(state: _AgentState) -> _AgentState:
        history = state.get("history", [])
        # The rewrite only exists to resolve references against prior turns. With no
        # history the question is already standalone, so skip the LLM call entirely
        # (faster, and avoids models answering the question instead of rewriting it).
        if not history:
            return {"rewritten": state["question"]}
        rewritten = rewrite_chain.invoke(
            {"input": state["question"], "chat_history": history}
        )
        return {"rewritten": (rewritten or "").strip() or state["question"]}

    def retrieve_node(state: _AgentState) -> _AgentState:
        query = state.get("rewritten") or state["question"]
        docs = retrieve(query)
        sources = sorted({(d.metadata or {}).get("source", "unknown") for d in docs})
        return {"context": format_docs(docs), "sources": sources}

    def answer_node(state: _AgentState) -> _AgentState:
        text = answer_chain.invoke(
            {
                "question": state["question"],
                "chat_history": state.get("history", []),
                "context": state.get("context", ""),
            }
        )
        return {"answer": (text or "").strip()}

    graph = StateGraph(_AgentState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def answer(
    question: str,
    history: Optional[List[BaseMessage]] = None,
    session_id: Optional[str] = None,
) -> AgentResult:
    """Answer a question as Sanket, grounded in retrieved portfolio context.

    Guardrails are applied here so any caller (Streamlit harness now, FastAPI
    ``/chat`` later) gets the same behavior. ``session_id`` is accepted for
    forward-compatibility but unused until the Phase 2 checkpointer.
    """
    question = (question or "").strip()
    history = history or []

    # Guardrail fast-paths (skip the graph entirely)
    if should_block_user_input(question):
        return AgentResult(answer=BLOCKED_REPLY)
    if is_greeting(question):
        return AgentResult(answer=GREETING_REPLY)
    if is_memory_like(question):
        return AgentResult(answer=MEMORY_REPLY)

    final = _build_graph().invoke({"question": question, "history": history})

    text = (final.get("answer") or "").strip() or EMPTY_CONTEXT_REPLY
    text = truncate_answer(strip_em_dashes(redact_secrets(text)))

    return AgentResult(
        answer=text,
        rewritten_question=final.get("rewritten", ""),
        sources=final.get("sources", []),
    )
