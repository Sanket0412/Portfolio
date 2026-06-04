"""Structured / live data sources for the Phase 2b agent tools.

Pure helpers with no LangChain or citation coupling, so they stay easy to test:

  - :func:`load_publications` / :func:`format_publications` read the curated
    ``content/publications/publications.json`` (the same file the Streamlit
    Publications page renders), giving the agent authoritative venue, date, and
    citation-count detail that fuzzy retrieval would blur.
  - :func:`fetch_github_repos` / :func:`format_github_repos` call the live GitHub
    REST API for the owner's public repositories. The token is optional (it only
    raises the rate limit); results are cached briefly to stay polite to the API.

Both surfaces honor the content-policy exclusion (:func:`portfolio_api.rag.is_excluded`):
an excluded topic is never surfaced even though it does not flow through the ingest
sanitizer.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple

import requests

from portfolio_api.config import REPO_ROOT, get_settings
from portfolio_api.rag import is_excluded

PUBLICATIONS_PATH = REPO_ROOT / "content" / "publications" / "publications.json"

# GitHub API
_GITHUB_API = "https://api.github.com"
_GITHUB_TIMEOUT = 10  # seconds
_GITHUB_CACHE_TTL = 300  # seconds; one owner's repo list changes rarely
_MAX_REPOS = 15

# Simple per-process TTL cache: {username: (fetched_at, repos)}.
_repo_cache: Dict[str, Tuple[float, List[dict]]] = {}


# --------------------------------------------------------------------------- #
# Publications (structured, from repo JSON)
# --------------------------------------------------------------------------- #
def load_publications() -> List[dict]:
    """Load the curated publications list, or [] if the file is missing/invalid."""
    try:
        raw = PUBLICATIONS_PATH.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            return []
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def format_publications(pubs: List[dict]) -> str:
    """Render publications as a compact, grounded text block (data, not instructions)."""
    if not pubs:
        return "No publications are listed in the portfolio."

    blocks: List[str] = []
    for p in pubs:
        authors = ", ".join(str(a) for a in (p.get("authors") or []))
        lines = [
            f"Title: {p.get('title', '').strip()}",
            f"Publisher: {p.get('publisher', '')} | Venue: {p.get('venue_name', '')}",
            f"Date: {p.get('date', '')} | Citations: {p.get('citations', 'n/a')}",
        ]
        if authors:
            lines.append(f"Authors: {authors}")
        if p.get("doi"):
            lines.append(f"DOI: {p['doi']}")
        if p.get("url"):
            lines.append(f"URL: {p['url']}")
        if p.get("abstract"):
            lines.append(f"Abstract: {p['abstract'].strip()}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# GitHub (live, from the REST API)
# --------------------------------------------------------------------------- #
def _repo_is_excluded(repo: dict) -> bool:
    """Drop a repo whose name/description trips the content-policy exclusion."""
    return is_excluded(repo.get("name", "")) or is_excluded(repo.get("description", "") or "")


def fetch_github_repos(
    *,
    username: Optional[str] = None,
    token: Optional[str] = None,
    query: str = "",
) -> List[dict]:
    """Fetch the owner's public repos (cached), newest-pushed first, exclusion-filtered.

    Returns a trimmed list of dicts: name, description, language, stars, url,
    pushed_at, topics. Raises ``requests.RequestException`` on a network/API error
    so the caller can return a graceful message to the model.
    """
    settings = get_settings()
    username = username or settings.github_username
    token = settings.github_token if token is None else token

    now = time.time()
    cached = _repo_cache.get(username)
    if cached and now - cached[0] < _GITHUB_CACHE_TTL:
        repos = cached[1]
    else:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"portfolio-agent/{username}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        resp = requests.get(
            f"{_GITHUB_API}/users/{username}/repos",
            headers=headers,
            params={"per_page": 100, "sort": "pushed", "type": "owner"},
            timeout=_GITHUB_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()

        repos = []
        for r in payload if isinstance(payload, list) else []:
            if r.get("fork") or r.get("private"):
                continue
            if _repo_is_excluded(r):
                continue
            repos.append(
                {
                    "name": r.get("name", ""),
                    "description": (r.get("description") or "").strip(),
                    "language": r.get("language") or "",
                    "stars": r.get("stargazers_count", 0),
                    "url": r.get("html_url", ""),
                    "pushed_at": (r.get("pushed_at") or "")[:10],
                    "topics": r.get("topics") or [],
                }
            )
        _repo_cache[username] = (now, repos)

    q = query.strip().lower()
    if q:
        repos = [
            r
            for r in repos
            if q in r["name"].lower()
            or q in r["description"].lower()
            or q in r["language"].lower()
            or any(q in str(t).lower() for t in r["topics"])
        ]

    return repos[:_MAX_REPOS]


def format_github_repos(repos: List[dict], *, query: str = "") -> str:
    """Render repos as a grounded text block (data, not instructions)."""
    if not repos:
        scope = f" matching '{query}'" if query.strip() else ""
        return f"No public GitHub repositories{scope} were found."

    blocks: List[str] = []
    for r in repos:
        lines = [f"Repo: {r['name']}"]
        if r["description"]:
            lines.append(f"Description: {r['description']}")
        meta = []
        if r["language"]:
            meta.append(f"Language: {r['language']}")
        meta.append(f"Stars: {r['stars']}")
        if r["pushed_at"]:
            meta.append(f"Last pushed: {r['pushed_at']}")
        lines.append(" | ".join(meta))
        if r["topics"]:
            lines.append(f"Topics: {', '.join(str(t) for t in r['topics'])}")
        lines.append(f"URL: {r['url']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
