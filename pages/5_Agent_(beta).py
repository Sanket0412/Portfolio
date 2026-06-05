# pages/5_Agent_(beta).py
# Throwaway harness for the new portfolio_api agent (Phase 1 RAG parity).
# Calls portfolio_api.agent.answer() in-process against the persistent pgvector
# store. The original pages/4_Chat.py stays untouched as the parity baseline.

import streamlit as st
from uuid import uuid4

from components.config.bootstrap import *  # noqa: F401,F403
from components.navbar import render_sidebar_profile
from langchain_core.messages import AIMessage, HumanMessage

from portfolio_api.agent import (
    TurnResult,
    fast_path_suggestions,
    generate_suggestions,
    stream_answer,
)
from portfolio_api.guardrails import looks_like_job_description

PERSONA_NAME = "Sanket J Shah"
ASSISTANT_AVATAR = "https://avatars.githubusercontent.com/u/68991626?v=4"


def _avatar_for_role(role: str):
    return None if role == "user" else ASSISTANT_AVATAR


# First Streamlit call
st.set_page_config(page_title="Agent (beta) - Chat with Sanket", page_icon="🧪", layout="wide")

with st.sidebar:
    render_sidebar_profile(show_env=True)

# =========================
# Session state
# =========================
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []  # list of {"role", "content"} for rendering
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []  # list[BaseMessage] passed to the agent
if "agent_session_id" not in st.session_state:
    st.session_state.agent_session_id = str(uuid4())
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []  # follow-up chips for the latest turn

# =========================
# Header
# =========================
st.title("🧪 Agent (beta) - Chat with Sanket")
st.caption(
    "Phase 1 parity harness. Same persona and knowledge base as CloneAMA, but served "
    "by the new portfolio_api agent (LangGraph + Supabase pgvector). Compare against "
    "the Chat page."
)

# =========================
# Chat UI
# =========================
for message in st.session_state.agent_messages:
    role = message.get("role", "assistant")
    with st.chat_message(role, avatar=_avatar_for_role(role)):
        st.markdown(message.get("content", ""))

# A clicked follow-up chip (set on the previous run) takes priority over typed input.
pending = st.session_state.pop("pending_suggestion", None)
typed = st.chat_input("Ask me about my work, projects, or experience...")
prompt = pending or typed

if prompt:
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=_avatar_for_role("user")):
        st.markdown(prompt)
        if looks_like_job_description(prompt):
            st.caption("🧭 Recruiter / job-fit mode detected - matching against my experience.")

    with st.chat_message("assistant", avatar=_avatar_for_role("assistant")):
        sink = TurnResult()
        placeholder = st.empty()
        ok = True
        try:
            buffered = []
            for delta in stream_answer(
                prompt,
                history=st.session_state.agent_history,
                session_id=st.session_state.agent_session_id,
                sink=sink,
            ):
                buffered.append(delta)
                placeholder.markdown("".join(buffered))
            # Replace the streamed deltas with the fully guardrailed answer.
            reply = sink.answer or "".join(buffered)
            placeholder.markdown(reply)
        except Exception as e:
            ok = False
            reply = f"Sorry, I ran into an error while responding: {type(e).__name__}: {e}"
            placeholder.markdown(reply)

        # Citations + debug visibility for parity review
        if ok:
            citations = sink.citations or []
            if citations:
                # Retrieval hits carry a similarity score; exact sources
                # (publications, github) use 0.0 and are labelled "exact".
                def _chip_label(c):
                    return f"{c.source} (exact)" if c.score == 0.0 else f"{c.source} ({c.score:.3f})"

                chips = " ".join(
                    f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;"
                    f"border:1px solid #374151;background:#111827;color:#e5e7eb;font-size:12px;"
                    f"margin:4px 4px 0 0;'>{_chip_label(c)}</span>"
                    for c in citations
                )
                st.markdown(
                    f"<div style='margin-top:6px;'><span style='color:#9ca3af;font-size:12px;'>"
                    f"Sources:</span><br>{chips}</div>",
                    unsafe_allow_html=True,
                )
            with st.expander("Debug: retrieval"):
                st.write("Sources:", sink.sources or "(none)")
                for c in citations:
                    score_label = "exact" if c.score == 0.0 else f"{c.score:.4f}"
                    st.markdown(f"**{c.source}** · score `{score_label}`")
                    st.caption(c.snippet)

        # Suggestions are computed AFTER the answer is rendered, so they stay off the
        # answer's critical path. Greeting / blocked / memory turns use starter
        # suggestions with no LLM call.
        if not ok:
            suggestions = []
        elif sink.canned:
            suggestions = fast_path_suggestions(prompt)
        else:
            with st.spinner("Loading follow-ups..."):
                suggestions = generate_suggestions(prompt, reply, sink.sources)

    st.session_state.agent_messages.append({"role": "assistant", "content": reply})

    # Thread history for the next turn (only successful agent turns)
    if ok:
        st.session_state.agent_history.append(HumanMessage(content=prompt))
        st.session_state.agent_history.append(AIMessage(content=reply))
        st.session_state.last_suggestions = suggestions
    else:
        st.session_state.last_suggestions = []

# =========================
# Follow-up suggestion chips (Phase 2d)
# =========================
if st.session_state.last_suggestions:
    st.markdown("**You might also ask:**")
    cols = st.columns(len(st.session_state.last_suggestions))
    for i, suggestion in enumerate(st.session_state.last_suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_suggestion = suggestion
            st.rerun()

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.divider()
    st.markdown("### Chat Settings")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.agent_messages = []
        st.session_state.agent_history = []
        st.session_state.agent_session_id = str(uuid4())
        st.session_state.last_suggestions = []
        st.rerun()
