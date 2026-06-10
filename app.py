"""
AI Customer Support Bot — Main Application
RAG-powered intelligent support chatbot with Groq LLM
Author: Paladugu Nandith Kumar
"""

import streamlit as st
import os
import time
from src.knowledge_base import KnowledgeBase
from src.intent_detector import IntentDetector
from src.responder import Responder
from src.conversation import ConversationManager

st.set_page_config(
    page_title="AI Support Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.brand { font-size:2rem; font-weight:800;
  background:linear-gradient(135deg,#0ea5e9,#6366f1);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub { color:#64748b; font-size:0.9rem; margin-bottom:1rem; }
.user-msg { background:#eff6ff; border-left:4px solid #3b82f6;
  border-radius:0 12px 12px 12px; padding:12px 16px; margin:6px 0; }
.bot-msg  { background:#f8fafc; border-left:4px solid #0ea5e9;
  border-radius:12px 0 12px 12px; padding:12px 16px; margin:6px 0; }
.badge { display:inline-block; background:#dbeafe; color:#1d4ed8;
  font-size:11px; font-weight:600; padding:2px 8px; border-radius:12px; margin-bottom:6px; }
.badge-escalation { background:#fee2e2; color:#dc2626; }
.online-dot { display:inline-block; width:8px; height:8px;
  background:#22c55e; border-radius:50%; margin-right:5px; }
</style>
""", unsafe_allow_html=True)


def init():
    defaults = {
        "messages": [], "kb": None, "responder": None,
        "conv": ConversationManager(), "intent_detector": IntentDetector(),
        "ready": False, "kb_name": None, "n_chunks": 0,
        "query_count": 0, "start_time": time.time()
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar():
    with st.sidebar:
        st.markdown("## 🤖 Bot Config")

        api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        model = st.selectbox("🧠 Model", [
            "llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"
        ])

        st.markdown("### 🎭 Persona")
        company = st.text_input("Company", value="TechCorp")
        bot_name = st.text_input("Bot Name", value="Alex")
        tone = st.selectbox("Tone", [
            "Professional & Helpful", "Friendly & Casual", "Formal & Precise"
        ])

        st.divider()
        st.markdown("### 📚 Knowledge Base")
        source = st.radio("Source", ["Use sample KB", "Upload documents"])

        files = None
        if source == "Upload documents":
            files = st.file_uploader("Upload FAQ/Docs", type=["pdf","txt"],
                                      accept_multiple_files=True)

        load = st.button("⚡ Load Knowledge Base", type="primary",
                          use_container_width=True, disabled=not api_key)

        if st.session_state.ready:
            st.divider()
            st.markdown("### 📊 Stats")
            uptime = int(time.time() - st.session_state.start_time)
            m, s = divmod(uptime, 60)
            c1, c2 = st.columns(2)
            c1.metric("Queries", st.session_state.query_count)
            c2.metric("Uptime",  f"{m}m{s}s")
            st.metric("KB Chunks", st.session_state.n_chunks)
            st.success("🟢 Bot is online")

        if st.session_state.messages:
            st.divider()
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conv = ConversationManager()
                st.session_state.query_count = 0
                st.rerun()

    return api_key, model, company, bot_name, tone, source, files, load


def load_kb(source, files, model, api_key, company, bot_name, tone):
    with st.spinner("📚 Loading knowledge base..."):
        kb = KnowledgeBase()
        if source == "Use sample KB":
            kb.load_sample()
            kb_name = "Sample E-commerce FAQ"
        else:
            if not files:
                st.error("Please upload at least one file.")
                return
            paths = []
            for f in files:
                p = f"/tmp/{f.name}"
                with open(p, "wb") as out:
                    out.write(f.getbuffer())
                paths.append(p)
            kb.load_files(paths)
            kb_name = f"{len(files)} file(s)"

    with st.spinner("🔢 Building vector index..."):
        kb.build_index()

    with st.spinner("🤖 Initializing bot..."):
        responder = Responder(
            kb=kb, model_name=model, api_key=api_key,
            company=company, bot_name=bot_name, tone=tone
        )

    st.session_state.kb       = kb
    st.session_state.responder = responder
    st.session_state.ready    = True
    st.session_state.kb_name  = kb_name
    st.session_state.n_chunks = kb.n_chunks
    st.session_state.messages = []
    st.session_state.conv     = ConversationManager()

    # Welcome message
    st.session_state.messages.append({
        "role": "bot", "content":
        f"Hi there! 👋 I'm **{bot_name}**, your {company} support assistant. "
        f"Knowledge base loaded with {kb.n_chunks} chunks. How can I help you today?",
        "intent": None
    })


def handle_query(query, bot_name):
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.conv.add("user", query)
    st.session_state.query_count += 1

    intent = st.session_state.intent_detector.detect(query)
    result = st.session_state.responder.respond(
        query=query,
        history=st.session_state.conv.format()
    )

    answer  = result["answer"]
    sources = result["sources"]
    st.session_state.conv.add("assistant", answer)
    st.session_state.messages.append({
        "role": "bot", "content": answer,
        "intent": intent, "sources": sources
    })


def chat_ui(bot_name):
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg"><strong>You</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            badge = ""
            if msg.get("intent"):
                cls = "badge-escalation" if msg["intent"] == "escalation" else "badge"
                badge = f'<span class="{cls}">{msg["intent"].replace("_"," ").title()}</span><br>'
            st.markdown(
                f'<div class="bot-msg">{badge}<strong>{bot_name}</strong><br>{msg["content"]}</div>',
                unsafe_allow_html=True
            )
            if msg.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**[{i}]** ...{s[:200]}...")

    # Quick suggestions
    if st.session_state.ready and len(st.session_state.messages) <= 1:
        st.markdown("**💡 Try asking:**")
        cols = st.columns(3)
        for col, q in zip(cols, [
            "How do I track my order?",
            "What is the return policy?",
            "What payment methods do you accept?"
        ]):
            if col.button(q, use_container_width=True):
                handle_query(q, bot_name)
                st.rerun()

    query = st.chat_input(f"Message {bot_name}...", disabled=not st.session_state.ready)
    if query:
        handle_query(query, bot_name)
        st.rerun()


def main():
    init()
    api_key, model, company, bot_name, tone, source, files, load = sidebar()

    st.markdown(f'<p class="brand">🤖 {company} Support Bot</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub">Powered by RAG + Groq  |  Intelligent support for {company}</p>',
                unsafe_allow_html=True)

    if load:
        load_kb(source, files, model, api_key, company, bot_name, tone)
        st.rerun()

    if not st.session_state.ready:
        c1, c2, c3 = st.columns(3)
        for col, (h, t) in zip([c1,c2,c3], [
            ("📚 Knowledge Base", "Upload your FAQs, policies, or product docs"),
            ("🧠 RAG Engine",     "Answers grounded in your documents — no hallucination"),
            ("⚡ Groq Powered",   "Ultra-fast responses on the free tier"),
        ]):
            col.markdown(
                f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center"><h3>{h}</h3><p style="color:#6b7280">{t}</p></div>',
                unsafe_allow_html=True
            )
        st.info("👈 Add your Groq API key and click Load Knowledge Base to start!")
    else:
        chat_ui(bot_name)


if __name__ == "__main__":
    main()
