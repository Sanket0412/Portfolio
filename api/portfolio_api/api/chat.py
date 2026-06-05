"""Streaming ``/chat`` endpoint (Phase 2a).

Streams an agent turn as Server-Sent Events. This is the exact contract the future
Next.js BFF (Phase 4) will consume and map to AI SDK stream parts.

Event types (each ``data`` is a JSON string):
  - ``token``       incremental answer text: ``{"text": "..."}``
  - ``tool_call``   the agent invoked a tool: ``{"name": "...", "args": {...}}``
  - ``tool_result`` a tool returned: ``{"name": "..."}``
  - ``citations``   sources backing the answer: ``[{"source","snippet","score"}, ...]``
  - ``suggestions`` follow-up questions to offer next: ``["...", ...]``
  - ``final``       the fully guardrailed answer (authoritative): ``{"answer": "..."}``
  - ``done``        end of stream: ``[DONE]``
  - ``error``       something failed: ``{"message": "..."}``

Guardrails: block/greeting/memory inputs short-circuit to a single ``final`` +
``done`` without invoking the model. Streamed ``token`` deltas are best-effort and
em-dash-stripped per delta; the ``final`` event carries the fully sanitized answer
(redaction, em-dash removal, truncation) and is the source of truth for clients.
"""
from __future__ import annotations

import json
from typing import AsyncGenerator, List, Literal, Optional

from fastapi import APIRouter
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from portfolio_api.agent.graph import (
    _fast_path_reply,
    _finalize,
    build_agent,
    fast_path_suggestions,
    message_text,
)
from portfolio_api.agent.suggestions import generate_suggestions
from portfolio_api.agent.tools import get_citations, reset_citations
from portfolio_api.guardrails import strip_em_dashes

router = APIRouter()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[ChatMessage] = Field(default_factory=list)


def _to_lc_history(history: List[ChatMessage]) -> List[BaseMessage]:
    out: List[BaseMessage] = []
    for m in history:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        else:
            out.append(AIMessage(content=m.content))
    return out


def _sse(event: str, data) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


async def _event_stream(req: ChatRequest) -> AsyncGenerator[dict, None]:
    question = (req.message or "").strip()

    # Guardrail fast-paths: no model call.
    canned = _fast_path_reply(question)
    if canned is not None:
        yield _sse("final", {"answer": canned})
        yield _sse("suggestions", fast_path_suggestions(question))
        yield _sse("done", "[DONE]")
        return

    reset_citations()
    messages = _to_lc_history(req.history) + [HumanMessage(content=question)]

    buffered: List[str] = []
    try:
        async for event in build_agent(streaming=True).astream_events(
            {"messages": messages}, version="v2"
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                delta = message_text(chunk) if chunk is not None else ""
                if delta:
                    buffered.append(delta)
                    yield _sse("token", {"text": strip_em_dashes(delta)})

            elif kind == "on_tool_start":
                yield _sse(
                    "tool_call",
                    {"name": event.get("name", ""), "args": event["data"].get("input", {})},
                )

            elif kind == "on_tool_end":
                yield _sse("tool_result", {"name": event.get("name", "")})

        citations = get_citations()
        yield _sse(
            "citations",
            [{"source": c.source, "snippet": c.snippet, "score": c.score} for c in citations],
        )

        final_text = _finalize("".join(buffered))
        yield _sse("final", {"answer": final_text})
        yield _sse(
            "suggestions",
            generate_suggestions(question, final_text, sorted({c.source for c in citations})),
        )
        yield _sse("done", "[DONE]")

    except Exception as exc:  # surface a clean error event instead of a 500 mid-stream
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})
        yield _sse("done", "[DONE]")


@router.post("/chat")
async def chat(req: ChatRequest) -> EventSourceResponse:
    """Stream an agent turn as Server-Sent Events."""
    return EventSourceResponse(_event_stream(req))
