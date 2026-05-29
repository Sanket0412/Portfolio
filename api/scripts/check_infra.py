"""Phase 0 infrastructure verification.

Checks, in order:
  1. config loads
  2. pgvector extension is enabled (enables it if privileges allow)
  3. each configured LLM provider responds to a tiny ping
  4. embeddings return a 1536-dim vector

Run:  python api/scripts/check_infra.py
Requires api/.env populated (DATABASE_URL, OPENAI_API_KEY, ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import re
import sys

from portfolio_api.config import get_settings


def _ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def _skip(msg: str) -> None:
    print(f"[SKIP] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _redact(text: str, settings) -> str:
    """Mask anything that could leak the DB password or API keys in error output.

    Handles passwords containing '@' or ':' by splitting on the *last* '@' (host
    boundary) and the *first* ':' inside userinfo (user/password boundary).
    Also masks 6-char sliding windows of the password so partial leaks are
    redacted too.
    """
    out = text
    if settings.database_url:
        url = settings.database_url
        out = out.replace(url, "***DATABASE_URL***")
        try:
            after_scheme = url.split("://", 1)[1]
            userinfo, _, _rest = after_scheme.rpartition("@")
            _user, _, password = userinfo.partition(":")
            if password:
                out = out.replace(password, "***")
                if len(password) >= 6:
                    seen: set[str] = set()
                    for i in range(len(password) - 5):
                        win = password[i : i + 6]
                        if win in seen:
                            continue
                        seen.add(win)
                        out = out.replace(win, "***")
        except Exception:
            pass
    for secret in (settings.openai_api_key, settings.anthropic_api_key):
        if secret and len(secret) > 8:
            out = out.replace(secret, "***")
    return out


def main() -> int:
    settings = get_settings()
    failures = 0

    _ok(f"config loaded (env={settings.app_env}, provider={settings.llm_provider})")

    # --- pgvector ---
    if settings.database_url:
        try:
            from portfolio_api.db import ensure_pgvector, pgvector_enabled

            ensure_pgvector()
            if pgvector_enabled():
                _ok("pgvector extension enabled")
            else:
                _fail("pgvector extension NOT found after CREATE EXTENSION")
                failures += 1
        except Exception as e:  # noqa: BLE001
            _fail(_redact(f"pgvector check: {type(e).__name__}: {e}", settings))
            failures += 1
    else:
        _skip("DATABASE_URL not set; skipping pgvector check")

    # --- providers ---
    from portfolio_api.llm import get_chat_model

    for provider, key in (("openai", settings.openai_api_key),
                          ("anthropic", settings.anthropic_api_key)):
        if not key:
            _skip(f"{provider}: no API key set")
            continue
        try:
            model = get_chat_model(provider, temperature=0)
            reply = model.invoke("Reply with the single word: pong")
            text = getattr(reply, "content", str(reply))
            _ok(f"{provider} responded: {str(text)[:40]!r}")
        except Exception as e:  # noqa: BLE001
            _fail(_redact(f"{provider} ping: {type(e).__name__}: {e}", settings))
            failures += 1

    # --- embeddings ---
    if settings.openai_api_key:
        try:
            from portfolio_api.llm import get_embeddings

            vec = get_embeddings().embed_query("hello")
            if len(vec) == settings.embedding_dim:
                _ok(f"embeddings dim = {len(vec)}")
            else:
                _fail(f"embeddings dim {len(vec)} != expected {settings.embedding_dim}")
                failures += 1
        except Exception as e:  # noqa: BLE001
            _fail(_redact(f"embeddings: {type(e).__name__}: {e}", settings))
            failures += 1
    else:
        _skip("OPENAI_API_KEY not set; skipping embeddings check")

    print()
    if failures:
        _fail(f"{failures} check(s) failed")
        return 1
    _ok("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
