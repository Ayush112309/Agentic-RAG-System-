# streamlit_app.py
"""
Agentic RAG System — Streamlit Frontend
A polished, production-quality web interface for the RAG chatbot.

Features:
  - Document upload (PDF, TXT, CSV)
  - Chat interface with conversation history
  - Retrieved sources & relevance scores display
  - Sidebar with system status
  - Clear chat button
  - Dark-mode compatible styling
"""

import logging
import sys
import time
from pathlib import Path

import streamlit as st

# ─── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Setup logging ────────────────────────────────────────────────────────────
from app.utils.logging_config import setup_logging
setup_logging(level="INFO", log_to_file=True)
logger = logging.getLogger(__name__)

# ─── Pipeline import ──────────────────────────────────────────────────────────
from app.utils.pipeline import RAGPipeline
from app.utils.config import Config


# ─── Custom CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Font import ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ── */
:root {
    --primary: #6C63FF;
    --primary-dim: #4f46e5;
    --accent: #22d3ee;
    --surface: #1e1e2e;
    --surface2: #2a2a3d;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --success: #34d399;
    --warning: #fbbf24;
    --error: #f87171;
    --border: rgba(255,255,255,0.08);
    --radius: 12px;
    --shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App header ── */
.app-header {
    background: linear-gradient(135deg, #6C63FF 0%, #22d3ee 100%);
    padding: 1.5rem 2rem;
    border-radius: var(--radius);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: white;
    margin: 0;
}
.app-header p {
    color: rgba(255,255,255,0.85);
    margin: 0;
    font-size: 0.9rem;
}

/* ── Chat bubbles ── */
.chat-bubble {
    padding: 0.85rem 1.2rem;
    border-radius: var(--radius);
    margin-bottom: 0.75rem;
    max-width: 85%;
    line-height: 1.6;
    font-size: 0.95rem;
}
.user-bubble {
    background: linear-gradient(135deg, #6C63FF22, #6C63FF11);
    border: 1px solid #6C63FF44;
    margin-left: auto;
    border-bottom-right-radius: 4px;
}
.ai-bubble {
    background: var(--surface2);
    border: 1px solid var(--border);
    margin-right: auto;
    border-bottom-left-radius: 4px;
}
.bubble-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
    opacity: 0.65;
}
.user-label { color: var(--primary); }
.ai-label { color: var(--accent); }

/* ── Source cards ── */
.source-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.chunk-content {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 0.4rem;
    white-space: pre-wrap;
    overflow-x: auto;
}

/* ── Score badge ── */
.score-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.score-high { background: #34d39922; color: #34d399; border: 1px solid #34d39944; }
.score-mid  { background: #fbbf2422; color: #fbbf24; border: 1px solid #fbbf2444; }
.score-low  { background: #f8717122; color: #f87171; border: 1px solid #f8717144; }

/* ── Stat cards ── */
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.85rem 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: var(--primary);
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Input area ── */
.stTextInput > div > div > input {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface2) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: var(--radius) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Divider ── */
.rag-divider {
    height: 1px;
    background: var(--border);
    margin: 1rem 0;
}

/* ── Tool badge ── */
.tool-badge {
    display: inline-block;
    background: #6C63FF22;
    border: 1px solid #6C63FF44;
    color: var(--primary);
    border-radius: 999px;
    padding: 0.15rem 0.6rem;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin-right: 0.3rem;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ─── Session state helpers ─────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialize all Streamlit session state variables."""
    defaults = {
        "pipeline": None,
        "messages": [],          # List of {role, content, sources, chunks, score, tools}
        "ingested_files": [],    # Names of files already ingested this session
        "doc_count": 0,          # Current embedding count
        "pipeline_ready": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_pipeline() -> RAGPipeline:
    """Get or create the RAGPipeline (cached in session state)."""
    if st.session_state.pipeline is None:
        with st.spinner("🔧 Initializing RAG pipeline..."):
            try:
                st.session_state.pipeline = RAGPipeline.from_config()
                st.session_state.pipeline_ready = True
                # Sync existing doc count
                st.session_state.doc_count = (
                    st.session_state.pipeline.vector_store.document_count()
                )
            except Exception as e:
                st.error(f"Failed to initialize pipeline: {e}")
                st.stop()
    return st.session_state.pipeline


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    """Render the left sidebar with upload and status sections."""
    pipeline = get_pipeline()

    with st.sidebar:
        st.markdown("## 🧠 Agentic RAG")
        st.markdown("<div class='rag-divider'></div>", unsafe_allow_html=True)

        # ── Stats ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class='stat-card'>
                    <div class='stat-number'>{st.session_state.doc_count}</div>
                    <div class='stat-label'>Chunks</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
            st.markdown(
                f"""<div class='stat-card'>
                    <div class='stat-number'>{msg_count}</div>
                    <div class='stat-label'>Queries</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='rag-divider'></div>", unsafe_allow_html=True)

        # ── Document Upload ──
        st.markdown("### 📄 Upload Documents")
        st.caption("Supported: PDF, TXT, CSV")

        uploaded_files = st.file_uploader(
            label="Drop files here",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.ingested_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        try:
                            chunks = pipeline.ingest_uploaded_file(
                                file_bytes=uploaded_file.read(),
                                filename=uploaded_file.name,
                            )
                            st.session_state.ingested_files.append(uploaded_file.name)
                            st.session_state.doc_count = (
                                pipeline.vector_store.document_count()
                            )
                            st.success(f"✅ {uploaded_file.name} ({chunks} chunks)")
                        except Exception as e:
                            st.error(f"❌ {uploaded_file.name}: {e}")

        st.markdown("<div class='rag-divider'></div>", unsafe_allow_html=True)

        # ── Ingested Sources ──
        sources = pipeline.vector_store.get_sources()
        if sources:
            st.markdown("### 📚 Ingested Sources")
            for src in sources:
                st.markdown(f"<div class='source-card'>📄 {src}</div>", unsafe_allow_html=True)
        else:
            st.info("No documents ingested yet.\nUpload files above to get started.")

        st.markdown("<div class='rag-divider'></div>", unsafe_allow_html=True)

        # ── Settings ──
        st.markdown("### ⚙️ Settings")
        with st.expander("Retrieval Settings"):
            new_k = st.slider("Chunks to retrieve (k)", 1, 10, pipeline.retriever.k)
            new_threshold = st.slider(
                "Relevance threshold",
                0.0, 1.0,
                pipeline.retriever.score_threshold,
                step=0.05,
            )
            if st.button("Apply Settings"):
                pipeline.retriever.update_config(k=new_k, score_threshold=new_threshold)
                st.success("Settings updated!")

        st.markdown("<div class='rag-divider'></div>", unsafe_allow_html=True)

        # ── Controls ──
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                pipeline.clear_memory()
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("⚠️ Reset All", use_container_width=True):
                pipeline.reset()
                st.session_state.messages = []
                st.session_state.ingested_files = []
                st.session_state.doc_count = 0
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("Built with LangChain · ChromaDB · Streamlit")


# ─── Main chat area ───────────────────────────────────────────────────────────

def render_score_badge(score: float) -> str:
    """Return HTML for a color-coded score badge."""
    if score >= 0.7:
        cls = "score-high"
        label = f"High ({score:.2f})"
    elif score >= 0.45:
        cls = "score-mid"
        label = f"Mid ({score:.2f})"
    else:
        cls = "score-low"
        label = f"Low ({score:.2f})"
    return f"<span class='score-badge {cls}'>{label}</span>"


def render_chat_history() -> None:
    """Render the full conversation history."""

    for msg in st.session_state.messages:

        # ───────────────── USER MESSAGE ─────────────────
        if msg["role"] == "user":

            st.markdown(
                f"""
                <div class='chat-bubble user-bubble'>
                    <div class='bubble-label user-label'>You</div>
                    {msg['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ───────────────── ASSISTANT MESSAGE ─────────────────
        else:

            # Tool badges
            tools_html = ""

            if msg.get("tools"):

                badges = "".join(
                    f"<span class='tool-badge'>⚙ {t}</span>"
                    for t in msg["tools"]
                )

                tools_html = (
                    f"<div style='margin-top:0.4rem'>{badges}</div>"
                )

            # Assistant bubble
            st.markdown(
                f"""
                <div class='chat-bubble ai-bubble'>
                    <div class='bubble-label ai-label'>Assistant</div>
                    {msg['content']}
                    {tools_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ───────────────── SOURCES SECTION ─────────────────
            if msg.get("sources"):

                score = msg.get("score", 0.0)

                score_html = render_score_badge(score)

                st.markdown(
                    f"""
                    <div style='margin-top:0.5rem; margin-bottom:0.5rem;'>
                        📎 {len(msg.get("sources", []))} source(s)
                        · Relevance: {score_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📚 Sources Used", expanded=False):

                    for src in msg["sources"]:

                        st.markdown(
                            f"""
                            <div class='source-card'>
                                📄 {src}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # ───────────────── RETRIEVED CHUNKS ─────────────────
            if msg.get("chunks"):

                with st.expander(
                    "🧩 Retrieved Context Chunks",
                    expanded=False,
                ):

                    for i, (doc, chunk_score) in enumerate(
                        msg["chunks"],
                        1,
                    ):

                        chunk_score_html = render_score_badge(
                            chunk_score
                        )

                        st.markdown(
                            f"""
                            <div style='margin-top:1rem'>
                                <strong>Chunk {i}</strong>
                                · {chunk_score_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        src = doc.metadata.get(
                            "source",
                            "Unknown",
                        )

                        page = doc.metadata.get("page", "")

                        page_info = (
                            f" · Page {page+1}"
                            if isinstance(page, int)
                            else ""
                        )

                        st.caption(
                            f"Source: {src}{page_info}"
                        )

                        st.markdown(
                            f"""
                            <div class='chunk-content'>
                                {doc.page_content[:1000]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.markdown("<hr>", unsafe_allow_html=True)

def render_main() -> None:
    """Render the main chat interface."""

    # ───────────────── HEADER ─────────────────
    st.markdown(
        """
        <div class='app-header'>
            <div>
                <h1>🧠 Agentic RAG System</h1>
                <p>Upload documents · Ask questions · Get grounded answers</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline = get_pipeline()

    # ───────────────── EMPTY STATE ─────────────────
    if st.session_state.doc_count == 0:

        st.info(
            "👈 Upload documents from the sidebar to begin chatting.",
            icon="💡",
        )

    # ───────────────── CHAT HISTORY ─────────────────
    render_chat_history()

    st.markdown(
        "<div class='rag-divider'></div>",
        unsafe_allow_html=True,
    )

    # ───────────────── INPUT AREA ─────────────────
    col_input, col_btn = st.columns([5, 1])

    with col_input:

        user_query = st.text_input(
            label="Ask a question",
            placeholder="e.g. What is RAG?",
            label_visibility="collapsed",
            key="query_input",
        )

    with col_btn:

        submit = st.button(
            "Send →",
            use_container_width=True,
            type="primary",
        )

    # ───────────────── QUERY PROCESSING ─────────────────
    if submit and user_query.strip():

        if st.session_state.doc_count == 0:

            st.warning(
                "Please upload at least one document first."
            )

            return

        # Add user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        # Run query
        with st.spinner(
            "🔍 Searching documents and generating answer..."
        ):

            try:

                start_time = time.time()

                response = pipeline.query(user_query)

                elapsed = time.time() - start_time

                # Save assistant response
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "chunks": response.retrieved_chunks,
                        "score": response.top_relevance_score,
                        "tools": response.used_tools,
                    }
                )

                logger.info(
                    f"Query completed in {elapsed:.2f}s"
                )

            except Exception as e:

                logger.exception("Query failed")

                st.error(f"Query failed: {e}")

                return

        # Refresh app
        st.rerun()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit app."""
    init_session_state()
    render_sidebar()
    render_main()


if __name__ == "__main__":
    main()