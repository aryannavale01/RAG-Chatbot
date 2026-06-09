# ===========================
# RAG Chatbot - Main Application
# ===========================

import os
import re
import concurrent.futures
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Import all settings
from settings import (
    STREAMLIT_PAGE_TITLE,
    STREAMLIT_PAGE_ICON,
    DISPLAY_CONTENT_LENGTH,
    ALLOWED_FILE_TYPES,
    UPLOAD_FOLDER,
    validate_settings,
    reload_settings,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    EMBEDDING_MODEL_NAME,
    FAISS_SEARCH_K,
    BM25_SEARCH_K,
    BM25_WEIGHT,
    FAISS_WEIGHT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHAIN_TYPE,
    RETURN_SOURCE_DOCUMENTS,
    MAX_FILE_SIZE_MB,
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


USER_AVATAR = "🧑"
AI_AVATAR = "🤖"


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
                st.text(doc.page_content[:DISPLAY_CONTENT_LENGTH])
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
        st.caption(
            f"{file_count} file(s) · {sum(f.size for f in uploaded_files) / 1024:.1f} KB"
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
        st.info(
            "📂 **No documents loaded.** Upload files in the sidebar to enable Q&A."
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
                "Please upload a document first to start asking questions!", icon="⚠️"
            )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "**No documents loaded.** Please upload and process documents in the sidebar, then ask your question again.",
                    "sources": [],
                }
            )
            st.rerun()

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=AI_AVATAR):
            thinking_placeholder = st.empty()
            thinking_placeholder.write("Thinking...")

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
                            st.text(doc.page_content[:300])

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
                        st.text(content[:300])


# ===========================
# Settings Page
# ===========================


def _save_settings(updated_settings: dict):
    """Write updated settings to .env, update os.environ, and reload."""
    env_path = ".env"
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    for key, value in updated_settings.items():
        pattern = rf"^{re.escape(key)}\s*=\s*.*$"
        if re.search(pattern, content, flags=re.MULTILINE):
            content = re.sub(
                pattern, f"{key}={value}", content, count=1, flags=re.MULTILINE
            )
        else:
            content += f"\n{key}={value}\n"
        os.environ[key] = str(value)

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(content)

    load_dotenv(override=True)
    reload_settings()
    st.cache_resource.clear()
    st.cache_data.clear()


def render_settings_page():
    """Render the settings management page."""
    st.title("Configuration")
    st.caption("Changes are saved to `.env` and apply after reloading the page.")

    st.markdown("---")

    with st.form("settings_form", clear_on_submit=False):
        st.markdown("### API Keys")
        api_key = st.text_input(
            "Groq API Key",
            value=os.environ.get("GROQ_API_KEY", GROQ_API_KEY),
            type="password",
            help="Get your API key from https://console.groq.com",
            placeholder="gsk_...",
        )

        st.markdown("### LLM Configuration")
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input(
                "Model Name",
                value=os.environ.get("GROQ_MODEL_NAME", GROQ_MODEL_NAME),
                help="e.g. llama-3.3-70b-versatile, mixtral-8x7b-32768",
            )
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=256,
                max_value=32768,
                step=256,
                value=int(os.environ.get("LLM_MAX_TOKENS", str(LLM_MAX_TOKENS))),
            )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            value=float(os.environ.get("LLM_TEMPERATURE", str(LLM_TEMPERATURE))),
            help="0.0 = deterministic, 2.0 = very creative",
        )

        st.markdown("### Embedding Model")
        embedding_model = st.text_input(
            "Model Name",
            value=os.environ.get("EMBEDDING_MODEL_NAME", EMBEDDING_MODEL_NAME),
            help="HuggingFace embedding model name",
        )

        st.markdown("### Retriever Configuration")
        col1, col2 = st.columns(2)
        with col1:
            faiss_k = st.number_input(
                "FAISS Search K",
                min_value=1,
                max_value=20,
                value=int(os.environ.get("FAISS_SEARCH_K", str(FAISS_SEARCH_K))),
            )
        with col2:
            bm25_k = st.number_input(
                "BM25 Search K",
                min_value=1,
                max_value=20,
                value=int(os.environ.get("BM25_SEARCH_K", str(BM25_SEARCH_K))),
            )

        st.markdown("##### Hybrid Retriever Weights")
        col1, col2 = st.columns(2)
        with col1:
            bm25_weight = st.slider(
                "BM25 Weight",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(os.environ.get("BM25_WEIGHT", str(BM25_WEIGHT))),
            )
        with col2:
            faiss_weight = st.slider(
                "FAISS Weight",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=float(os.environ.get("FAISS_WEIGHT", str(FAISS_WEIGHT))),
            )

        st.markdown("### Text Processing")
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                step=50,
                value=int(os.environ.get("CHUNK_SIZE", str(CHUNK_SIZE))),
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                step=10,
                value=int(os.environ.get("CHUNK_OVERLAP", str(CHUNK_OVERLAP))),
            )

        st.markdown("### QA Chain")
        chain_type = st.selectbox(
            "Chain Type",
            options=["stuff", "map_reduce", "refine", "map_rerank"],
            index=["stuff", "map_reduce", "refine", "map_rerank"].index(
                os.environ.get("CHAIN_TYPE", CHAIN_TYPE)
            ),
        )
        return_source_docs = st.checkbox(
            "Return Source Documents",
            value=os.environ.get(
                "RETURN_SOURCE_DOCUMENTS", str(RETURN_SOURCE_DOCUMENTS)
            ).lower()
            == "true",
        )

        st.markdown("### File Upload")
        max_file_size = st.number_input(
            "Max File Size (MB)",
            min_value=1,
            max_value=200,
            value=int(os.environ.get("MAX_FILE_SIZE_MB", str(MAX_FILE_SIZE_MB))),
        )

        st.markdown("---")
        saved = st.form_submit_button(
            "Save Settings", type="primary", use_container_width=True
        )

    if saved:
        updated = {
            "GROQ_API_KEY": api_key,
            "GROQ_MODEL_NAME": model_name,
            "LLM_TEMPERATURE": str(temperature),
            "LLM_MAX_TOKENS": str(max_tokens),
            "EMBEDDING_MODEL_NAME": embedding_model,
            "FAISS_SEARCH_K": str(faiss_k),
            "BM25_SEARCH_K": str(bm25_k),
            "BM25_WEIGHT": str(bm25_weight),
            "FAISS_WEIGHT": str(faiss_weight),
            "CHUNK_SIZE": str(chunk_size),
            "CHUNK_OVERLAP": str(chunk_overlap),
            "CHAIN_TYPE": chain_type,
            "RETURN_SOURCE_DOCUMENTS": str(return_source_docs).lower(),
            "MAX_FILE_SIZE_MB": str(max_file_size),
        }
        _save_settings(updated)
        st.toast("Settings saved. Reloading...")
        st.session_state.page = "chat"
        st.rerun()


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

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "page" not in st.session_state:
        st.session_state.page = "chat"

    errors = _cached_validate_settings()
    has_api_key = bool(os.environ.get("GROQ_API_KEY", GROQ_API_KEY))

    # ========== SIDEBAR ==========
    with st.sidebar:
        on_chat = st.session_state.page == "chat"
        target_page = "settings" if on_chat else "chat"
        target_label = "Settings" if on_chat else "Chat"
        if st.button(
            target_label,
            use_container_width=True,
            type="primary" if on_chat else "secondary",
        ):
            st.session_state.page = target_page
            st.rerun()

        if errors:
            for error in errors:
                st.warning(error)

        st.markdown("---")
        render_upload_sidebar()

    # ========== MAIN AREA ==========
    if st.session_state.page == "settings":
        render_settings_page()
        return

    # ========== API KEY CHECK ==========
    if not has_api_key:
        st.warning(
            "**Groq API Key Required** — Set your Groq API key in Settings to enable the chatbot."
        )
        if st.button("Open Settings", type="primary", use_container_width=False):
            st.session_state.page = "settings"
            st.rerun()

    # ========== CHAT AREA ==========
    if has_api_key:
        st.caption("System Ready")
    st.title(STREAMLIT_PAGE_TITLE)
    st.caption("Retrieval-Augmented Generation · Semantic Search · Document QA")

    qa_chain = None
    if has_api_key:
        try:
            qa_chain = _cached_qa_chain()
        except Exception:
            pass

    st.markdown("---")
    render_message_history()

    st.markdown("---")
    render_chat_interface(qa_chain)


if __name__ == "__main__":
    main()
