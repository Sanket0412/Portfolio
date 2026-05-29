"""Provider-agnostic chat model factory (OpenAI | Anthropic).

The rest of the codebase calls `get_chat_model()` and never imports a specific
provider, so we can A/B providers or swap models from config alone.
"""
from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from portfolio_api.config import Settings, get_settings


def get_chat_model(
    provider: str | None = None,
    *,
    temperature: float | None = None,
    streaming: bool = False,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a configured chat model.

    Args:
        provider: "openai" or "anthropic". Defaults to settings.llm_provider.
        temperature: overrides settings.llm_temperature when provided.
        streaming: enable token streaming (used by the SSE endpoint in Phase 2).
        settings: injected settings (defaults to the cached global).
        kwargs: extra provider-specific constructor args.
    """
    settings = settings or get_settings()
    provider = (provider or settings.llm_provider).lower()
    temperature = settings.llm_temperature if temperature is None else temperature

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        settings.require("openai_api_key")
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
            api_key=settings.openai_api_key,
            streaming=streaming,
            **kwargs,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        settings.require("anthropic_api_key")
        return ChatAnthropic(
            model=settings.anthropic_model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
            max_tokens=kwargs.pop("max_tokens", 2048),
            streaming=streaming,
            **kwargs,
        )

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. Expected 'openai' or 'anthropic'."
    )
