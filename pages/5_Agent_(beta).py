# pages/5_Agent_(beta).py
# Throwaway harness for the new portfolio_api agent (Phase 1 RAG parity).
# Calls portfolio_api.agent.answer() in-process against the persistent pgvector
# store. The original pages/4_Chat.py stays untouched as the parity baseline.

import streamlit as st
from uuid import uuid4

from components.config.bootstrap import *  # noqa: F401,F403
from components.navbar import render_sidebar_profile
from langchain_core.messages import AIMessage, HumanMessage

from portfolio_api.agent import answer

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

if prompt := st.chat_input("Ask me about my work, projects, or experience..."):
    st.session_state.agent_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=_avatar_for_role("user")):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=_avatar_for_role("assistant")):
        with st.spinner("Thinking..."):
            try:
                result = answer(
                    prompt,
                    history=st.session_state.agent_history,
                    session_id=st.session_state.agent_session_id,
                )
                reply = result.answer
            except Exception as e:
                result = None
                reply = f"Sorry, I ran into an error while responding: {type(e).__name__}: {e}"

            st.markdown(reply)

            # Debug visibility for parity review
            if result is not None:
                with st.expander("Debug: retrieval"):
                    st.write("Rewritten question:", result.rewritten_question or "(unchanged)")
                    st.write("Sources:", result.sources or "(none)")

    st.session_state.agent_messages.append({"role": "assistant", "content": reply})

    # Thread history for the next turn (only successful agent turns)
    if result is not None:
        st.session_state.agent_history.append(HumanMessage(content=prompt))
        st.session_state.agent_history.append(AIMessage(content=reply))

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
        st.rerun()
