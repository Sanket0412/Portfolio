# TodaySponsor — Personal Project

A personal project I designed and built independently to explore AI-native influencer
marketing: matching brands with creators by audience-to-customer overlap.

## What It Is

TodaySponsor is an AI-native influencer marketing platform that matches brands with creators
based on audience-to-customer overlap. It ingests a brand's Shopify CRM, builds a statistical
customer profile, embeds that profile into the same vector space as creator audience
demographics and content, then ranks creators by cosine similarity using distribution-level
lookalike modeling rather than rules-based demographic filters or follower-count heuristics.

The core technical differentiation is that the matching signal originates from who already
buys from the brand, not from natural-language briefs or engagement proxies. I built a live
v0 that runs end to end at todaysponsor.dev.

## What I Built

- End-to-end Shopify OAuth integration with HMAC verification, CSRF protection, secure token
  storage in Supabase, and scoped access to customers, orders, and products
- Brand profile computation from CRM data covering geography, average-order-value
  distribution, product category mix, RFM segmentation, and gender inference
- A two-stage matching engine over 1,555-dimension embeddings (16 structured demographic,
  1,536 semantic, 3 behavioral): Stage 1 is a semantic category coarse filter, Stage 2 is
  decomposed cosine similarity across demographics, geography, and content-topic alignment
- Agentic outreach draft generation and a streaming campaign agent with persisted
  conversation history
- A full dashboard with match-score breakdowns, an outreach flow, and campaign chat

## Tech Stack

- Frontend: Next.js (App Router) on Vercel
- Backend: FastAPI on Railway, Python 3.11
- Database: Supabase Postgres with pgvector
- Embeddings: OpenAI text-embedding-3-large
- LLMs: GPT-4o-mini for outreach drafts, GPT-4o for the streaming campaign agent
- ML and orchestration: LangChain, LangGraph, Vertex AI
- Integrations: Shopify OAuth, creator audience data, Resend for email
