# ===========================
# RAG Chatbot - Main Application
# ===========================

import os
import concurrent.futures
from pathlib import Path

import streamlit as st
import urllib.parse

from dotenv import load_dotenv

# Import all settings
from settings import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    DISPLAY_CONTENT_LENGTH,
    ALLOWED_FILE_TYPES,
    UPLOAD_FOLDER,
    validate_settings,
)

# Import all modular functions
from modules import (
    load_embedding_model,
    build_vector_store,
    create_chunks,
    process_uploaded_file,
    build_qa_chain,
)

# Load environment variables once at import time (not inside main)
load_dotenv()


# ===========================
# Modern SVG avatar icons (data URIs)
# ===========================

_USER_AVATAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>"""
_AI_AVATAR_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04Z"/></svg>"""

USER_AVATAR = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(_USER_AVATAR_SVG)
AI_AVATAR = "data:image/svg+xml;charset=utf-8," + urllib.parse.quote(_AI_AVATAR_SVG)


# ===========================
# Cached heavy resources
# ===========================


@st.cache_resource(show_spinner=False)
def _cached_qa_chain():
    """Build QA chain once and reuse across all reruns."""
    return build_qa_chain()


@st.cache_resource(show_spinner=False)
def _cached_embedding_model():
    """Load embedding model once and reuse across all reruns."""
    return load_embedding_model()


@st.cache_data(show_spinner=False)
def _cached_validate_settings():
    """Validate settings once; result never changes at runtime."""
    return validate_settings()


# ===========================
# CSS — injected once per session
# ===========================

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-base:       #09090f;
    --bg-surface:    #111118;
    --bg-card:       #16161f;
    --bg-hover:      #1e1e2a;
    --border:        rgba(255,255,255,0.06);
    --border-glow:   rgba(99,102,241,0.35);
    --accent:        #6366f1;
    --accent-2:      #a78bfa;
    --accent-glow:   rgba(99,102,241,0.15);
    --text-primary:  #f0f0fa;
    --text-secondary:#8b8ba8;
    --text-muted:    #4a4a62;
    --success:       #10b981;
    --warning:       #f59e0b;
    --error:         #ef4444;
    --radius:        14px;
    --radius-sm:     8px;
    --font-display:  'Syne', sans-serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp {
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,102,241,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(167,139,250,0.06) 0%, transparent 55%);
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none;
    z-index: 0;
    animation: gridShift 30s linear infinite;
}
@keyframes gridShift {
    from { background-position: 0 0; }
    to   { background-position: 44px 44px; }
}

.main .block-container {
    padding: 2rem 2.5rem 6rem !important;
    max-width: 900px !important;
    position: relative;
    z-index: 1;
}

h1 {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    background: linear-gradient(135deg, var(--text-primary) 30%, var(--accent-2)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin-bottom: 0.25rem !important;
    animation: fadeSlideDown 0.6s ease both;
}
@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to   { opacity: 1; transform: translateY(0); }
}

h2, h3 {
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 2px solid var(--border-glow) !important;
    padding: 1.5rem 1rem !important;
    min-width: 280px !important;
    max-width: 350px !important;
}
[data-testid="stSidebar"] h2 {
    font-family: var(--font-display) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--accent) !important;
    margin-bottom: 1rem !important;
    margin-top: 0.5rem !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-glow) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: var(--bg-hover) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

.stButton > button {
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
    background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.4rem !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.15s, box-shadow 0.2s !important;
    box-shadow: 0 0 0 rgba(99,102,241,0) !important;
}
.stButton > button::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.12), transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover::before { opacity: 1; }
.stButton > button:active { transform: translateY(0px) !important; }

[data-testid="stChatMessage"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.75rem !important;
    animation: messageIn 0.35s cubic-bezier(0.22,1,0.36,1) both;
    position: relative;
    overflow: hidden;
}
@keyframes messageIn {
    from { opacity: 0; transform: translateY(10px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    border-radius: 4px 0 0 4px;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--success), #34d399);
    border-radius: 4px 0 0 4px;
}

[data-testid="stChatInput"] {
    background: var(--bg-card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.8rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    margin-top: 1.5rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    background: transparent !important;
    font-size: 0.95rem !important;
    min-height: 50px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-muted) !important;
}

[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden !important;
    transition: border-color 0.2s;
}
[data-testid="stExpander"]:hover { border-color: var(--border-glow) !important; }
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
    color: var(--text-secondary) !important;
    padding: 0.6rem 0.8rem !important;
}

[data-testid="stSpinner"] > div {
    border-color: var(--accent) transparent transparent transparent !important;
    filter: drop-shadow(0 0 8px var(--accent)) !important;
}
.stSpinner p {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    font-size: 0.85rem !important;
    border-left-width: 3px !important;
}
.stSuccess { background: rgba(16,185,129,0.08) !important; border-color: var(--success) !important; color: #6ee7b7 !important; }
.stInfo    { background: rgba(99,102,241,0.08) !important; border-color: var(--accent)   !important; color: #a5b4fc !important; }
.stError   { background: rgba(239,68,68,0.08)  !important; border-color: var(--error)    !important; color: #fca5a5 !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border-color: var(--warning)  !important; color: #fcd34d !important; }

hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
    margin: 1.25rem 0 !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 0.25rem 0.7rem;
    border-radius: 99px;
    margin-bottom: 1rem;
}
.status-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 6px var(--success);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.75); }
}

.upload-stat {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-muted);
    background: var(--bg-hover);
    border-radius: 6px;
    padding: 0.3rem 0.6rem;
    display: inline-block;
    margin: 0.15rem 0.15rem 0.15rem 0;
}

.section-label {
    font-family: var(--font-display);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
}

.thinking-dots span {
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    margin: 0 2px;
    animation: bounce 1.2s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40%           { transform: translateY(-5px); opacity: 1; }
}

[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] img,
[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] img {
    border-radius: 50%;
    background: var(--bg-card);
    padding: 6px;
    border: 1.5px solid var(--border);
    width: 32px !important;
    height: 32px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stChatMessage"]:hover [data-testid="chatAvatarIcon-user"] img,
[data-testid="stChatMessage"]:hover [data-testid="chatAvatarIcon-assistant"] img {
    border-color: var(--border-glow);
    box-shadow: 0 0 12px var(--accent-glow);
}

.no-doc-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-muted);
    background: var(--bg-card);
    border: 1px dashed var(--border-glow);
    border-radius: var(--radius);
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
    animation: fadeSlideDown 0.5s ease both;
}
.no-doc-banner .icon {
    font-size: 1.3rem;
    flex-shrink: 0;
}
</style>
"""


def inject_custom_css():
    """Inject CSS only once per session to avoid redundant DOM writes."""
    if "css_injected" not in st.session_state:
        st.markdown(_CSS, unsafe_allow_html=True)
        st.session_state.css_injected = True


# ===========================
# Parallel file processor
# ===========================


def _save_and_parse(file) -> list:
    """Save one uploaded file to disk and parse it. Runs in a thread."""
    save_path = os.path.join(UPLOAD_FOLDER, file.name)
    with open(save_path, "wb") as f:
        f.write(file.getbuffer())
    file_type = Path(file.name).suffix.lower()
    return process_uploaded_file(save_path, file_type)


# ===========================
# UI Functions
# ===========================


def render_sources_sidebar():
    """Render sources section in sidebar."""
    st.markdown("### Retrieved Sources")

    if st.session_state.sources:
        for i, doc in enumerate(st.session_state.sources, 1):
            with st.expander(f"**Source {i}**", expanded=(i == 1)):
                st.markdown(
                    f'<div style="font-family:var(--font-mono,monospace);font-size:0.75rem;'
                    f'color:#a5b4fc;line-height:1.5;max-height:300px;overflow-y:auto;">'
                    f"{doc.page_content[:DISPLAY_CONTENT_LENGTH]}</div>",
                    unsafe_allow_html=True,
                )
                if doc.metadata:
                    st.caption("**Metadata:**")
                    st.json(doc.metadata)
    else:
        st.info("Upload documents to retrieve sources")


def render_upload_sidebar():
    """Render file upload section in sidebar."""
    st.markdown("### Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop files or click to browse",
        type=ALLOWED_FILE_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        file_count = len(uploaded_files)
        st.markdown(
            f'<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.5rem;">'
            f'<span class="upload-stat">{file_count} file(s)</span>'
            f'<span class="upload-stat">{sum(f.size for f in uploaded_files) / 1024:.1f} KB</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.success("Ready to process")

        if st.button("Process Documents", use_container_width=True, key="process_btn"):
            progress = st.progress(0, text="Saving and parsing files...")

            all_documents = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(_save_and_parse, f): f.name for f in uploaded_files
                }
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    done_count += 1
                    fname = futures[future]
                    progress.progress(
                        done_count / (len(uploaded_files) + 2),
                        text=f"Parsed {fname} ({done_count}/{len(uploaded_files)})",
                    )
                    try:
                        all_documents.extend(future.result())
                    except Exception as exc:
                        st.warning(f"Could not parse {fname}: {exc}")

            if not all_documents:
                progress.empty()
                st.error("Failed to load documents from files.")
                return

            progress.progress(
                (len(uploaded_files) + 1) / (len(uploaded_files) + 2),
                text=f"Chunking {len(all_documents)} documents...",
            )
            chunks = create_chunks(all_documents)

            progress.progress(
                (len(uploaded_files) + 2) / (len(uploaded_files) + 2),
                text="Building vector index...",
            )
            embedding_model = _cached_embedding_model()
            if embedding_model is None:
                progress.empty()
                st.error("Failed to load embedding model")
                return

            success = build_vector_store(chunks, embedding_model)
            progress.progress(1.0, text="Done!")

            if success:
                st.cache_resource.clear()
                st.success(f"Indexed {len(chunks)} chunks")
                st.info("Reload the page to activate the new index")
            else:
                st.error("Failed to build vector store")


def render_chat_interface(qa_chain):
    """Render chat interface and handle user queries."""
    st.markdown("### Ask Your Questions")

    if qa_chain is None:
        st.markdown(
            '<div class="no-doc-banner">'
            '<span class="icon">📂</span>'
            "<span><strong>No documents loaded.</strong> Upload files in the sidebar to enable Q&A.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    prompt = st.chat_input(
        "Ask anything about your documents..."
        if qa_chain is not None
        else "Upload documents first...",
        key="chat_input_main",
        disabled=False,
    )

    if prompt:
        if qa_chain is None:
            st.toast(
                "📄 Please upload a document first to start asking questions!", icon="⚠️"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "⚠️ **No documents loaded.** Please upload and process documents in the sidebar, then ask your question again.",
                    "sources": [],
                }
            )
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown(
                '<div class="thinking-dots"><span></span><span></span><span></span></div>',
                unsafe_allow_html=True,
            )

            try:
                with st.spinner("Searching documents..."):
                    response = qa_chain.invoke({"query": prompt})
                    result = response.get("result", "No response generated")
                    source_docs = response.get("source_documents", [])
                    st.session_state.sources = source_docs

                thinking_placeholder.empty()
                st.markdown(result)

                sources_data = []
                if source_docs:
                    for doc in source_docs:
                        sources_data.append(
                            {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                            }
                        )
                    st.divider()
                    with st.expander(
                        f"{len(source_docs)} source(s) used", expanded=False
                    ):
                        for i, doc in enumerate(source_docs, 1):
                            st.caption(f"**Source {i}:**")
                            st.markdown(
                                f'<div style="font-family:var(--font-mono,monospace);font-size:0.75rem;'
                                f"color:#cbd5e1;line-height:1.4;padding:0.5rem;"
                                f'background:var(--bg-card);border-radius:6px;max-height:200px;overflow-y:auto;">'
                                f"{doc.page_content[:300]}...</div>",
                                unsafe_allow_html=True,
                            )

            except Exception as e:
                thinking_placeholder.empty()
                st.error(f"Error: {str(e)}")
                return

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result,
                "sources": sources_data,
            }
        )


def render_message_history():
    """Render chat message history."""
    if not st.session_state.messages:
        st.info("No conversation yet. Upload documents and start chatting.")
        return

    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else AI_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                sources = message["sources"]
                st.divider()
                with st.expander(f"{len(sources)} source(s) used", expanded=False):
                    for i, doc in enumerate(sources, 1):
                        st.caption(f"**Source {i}:**")
                        content = doc.get("content", "")
                        st.markdown(
                            f'<div style="font-family:var(--font-mono,monospace);font-size:0.75rem;'
                            f"color:#cbd5e1;line-height:1.4;padding:0.5rem;"
                            f'background:var(--bg-card);border-radius:6px;max-height:200px;overflow-y:auto;">'
                            f"{content[:300]}...</div>",
                            unsafe_allow_html=True,
                        )


# ===========================
# Main Application
# ===========================


def main():
    """Main application function."""

    st.set_page_config(
        page_title=STREAMLIT_PAGE_TITLE,
        page_icon=STREAMLIT_PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # CSS injected once per session (not on every rerun)
    inject_custom_css()

    # Settings validated once (cached result)
    errors = _cached_validate_settings()
    if errors:
        for error in errors:
            st.warning(error)

    # Session state init
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []

    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown('<p style="margin:0; padding:0.5rem 0;">', unsafe_allow_html=True)
        render_upload_sidebar()
        st.markdown("</p>", unsafe_allow_html=True)

    # ========== MAIN AREA ==========
    # Header
    st.markdown(
        '<div class="status-badge"><span class="dot"></span>SYSTEM READY</div>',
        unsafe_allow_html=True,
    )
    st.title(STREAMLIT_PAGE_TITLE)
    st.markdown(
        '<p style="font-family:var(--font-mono,monospace);font-size:0.8rem;'
        'color:#4a4a62;margin-top:-0.5rem;margin-bottom:1.5rem;">'
        "Retrieval-Augmented Generation · Semantic Search · Document QA</p>",
        unsafe_allow_html=True,
    )

    # QA chain — loaded once, reused across all reruns via @st.cache_resource
    qa_chain = None
    try:
        qa_chain = _cached_qa_chain()
    except Exception:
        pass

    # Render chat history
    st.markdown("---")
    render_message_history()

    # Render chat input at the bottom
    st.markdown("---")
    render_chat_interface(qa_chain)


if __name__ == "__main__":
    main()
