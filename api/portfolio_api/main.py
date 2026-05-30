"""FastAPI application entry point.

Run locally:  uvicorn portfolio_api.main:app --reload --port 8000

Exposes /health and the Phase 2 streaming /chat endpoint.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from portfolio_api import __version__
from portfolio_api.api import chat as chat_router
from portfolio_api.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Portfolio Agent API",
    version=__version__,
    description="Agentic RAG backend for the Sanket Shah portfolio.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)


@app.get("/health")
def health() -> dict:
    """Liveness probe. Intentionally free of DB/LLM calls so it works pre-config."""
    return {
        "status": "ok",
        "version": __version__,
        "env": settings.app_env,
        "provider": settings.llm_provider,
    }


@app.get("/")
def root() -> dict:
    return {"name": "Portfolio Agent API", "version": __version__, "docs": "/docs"}
