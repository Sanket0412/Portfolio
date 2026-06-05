"""Agent layer: prompts + the LangGraph ReAct tool-using agent."""

from portfolio_api.agent.graph import (
    AgentResult,
    TurnResult,
    answer,
    build_agent,
    fast_path_suggestions,
    message_text,
    stream_answer,
)
from portfolio_api.agent.suggestions import generate_suggestions
from portfolio_api.agent.tools import Citation

__all__ = [
    "AgentResult",
    "TurnResult",
    "answer",
    "stream_answer",
    "build_agent",
    "fast_path_suggestions",
    "generate_suggestions",
    "message_text",
    "Citation",
]
