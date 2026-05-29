# Portfolio Agent API

Agentic RAG backend for the portfolio (FastAPI + LangGraph + Supabase pgvector).
This is the production backend being built to replace the Streamlit app's chat.
During development the existing Streamlit app is used as the test harness.

## Setup

From the repo root, using the existing `.venv`:

```powershell
.venv\Scripts\python.exe -m pip install -e ./api
copy api\.env.example api\.env   # then fill in DATABASE_URL + API keys
```

## Verify (Phase 0)

```powershell
# 1) API liveness (no DB/keys needed)
.venv\Scripts\python.exe -m uvicorn portfolio_api.main:app --port 8000
#    then open http://localhost:8000/health

# 2) Infra + providers (needs api/.env populated)
.venv\Scripts\python.exe api\scripts\check_infra.py
```

`check_infra.py` enables the pgvector extension, pings each configured LLM
provider, and confirms embeddings return a 1536-dim vector.

## Layout

```
api/
├── portfolio_api/
│   ├── config.py        # env-based settings
│   ├── db.py            # pgvector store + connection helpers
│   ├── llm/             # provider abstraction (openai|anthropic) + embeddings
│   └── main.py          # FastAPI app (/health; /chat in Phase 2)
├── migrations/          # 0001_enable_pgvector.sql
└── scripts/             # check_infra.py (ingest.py, parity_eval.py in Phase 1)
```

## Notes

- Package is named `portfolio_api` (not `app`) to avoid clobbering a generic
  top-level module when installed editable into the shared venv.
- Embeddings are pinned to OpenAI `text-embedding-3-small` (1536-dim); changing
  the model/dim requires re-ingesting the vector store.
- Secrets live only in `api/.env` (gitignored). Never commit them.
