-- Phase 0: enable pgvector on the Supabase Postgres database.
-- langchain-postgres (PGVector) creates and manages its own tables
-- (langchain_pg_collection, langchain_pg_embedding) on first ingest,
-- so the only schema prerequisite is the extension below.
--
-- Run via the Supabase SQL editor, or it is applied automatically by
-- `python api/scripts/check_infra.py` (which calls ensure_pgvector()).

CREATE EXTENSION IF NOT EXISTS vector;
