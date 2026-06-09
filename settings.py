# ===========================
# Settings Configuration
# ===========================

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===========================
# Paths Configuration
# ===========================

# Base directories
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", "data"))
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "data/uploaded_files"))
VECTORSTORE_FOLDER = Path(os.getenv("VECTORSTORE_FOLDER", "vectorstore"))

# Database paths
DB_FAISS_PATH = VECTORSTORE_FOLDER / os.getenv("DB_FAISS_NAME", "db_faiss")
CHUNKS_PATH = VECTORSTORE_FOLDER / os.getenv("CHUNKS_FILE", "chunks.pkl")
DOCS_PATH = VECTORSTORE_FOLDER / os.getenv("DOCS_FILE", "docs.pkl")

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, VECTORSTORE_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# ===========================
# LLM Configuration
# ===========================

# Groq API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ===========================
# Embedding Configuration
# ===========================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# ===========================
# Retriever Configuration
# ===========================

# FAISS retriever
FAISS_SEARCH_K = int(os.getenv("FAISS_SEARCH_K", "3"))

# BM25 retriever
BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "5"))

# Hybrid retriever weights (BM25, FAISS)
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.7"))

# ===========================
# Text Processing Configuration
# ===========================

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ===========================
# QA Chain Configuration
# ===========================

CHAIN_TYPE = os.getenv("CHAIN_TYPE", "stuff")
RETURN_SOURCE_DOCUMENTS = os.getenv("RETURN_SOURCE_DOCUMENTS", "true").lower() == "true"

# ===========================
# Streamlit Configuration
# ===========================

STREAMLIT_PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "Ask Chatbot")
STREAMLIT_PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", " ")  # blank to avoid emoji
MAX_SOURCE_DOCS_DISPLAY = int(os.getenv("MAX_SOURCE_DOCS_DISPLAY", "5"))
DISPLAY_CONTENT_LENGTH = int(os.getenv("DISPLAY_CONTENT_LENGTH", "500"))

# ===========================
# File Upload Configuration
# ===========================

ALLOWED_FILE_TYPES = [
    "pdf",
    "txt",
    "csv",
    "doc",
    "docx",
    "json",
    "html",
    "md",
    "xml",
    "xlsx",
    "xls",
    "pptx",
    "ppt",
]

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# ===========================
# Prompt Templates
# ===========================

CUSTOM_RAG_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.

If you don't know the answer, just say you don't know.
Do not try to make up an answer.

Context:
{context}

Question:
{question}

Start the answer directly.
"""

CUSTOM_RAG2_PROMPT_TEMPLATE = """
You are an intelligent RAG assistant.

Your primary job is to answer using the provided context documents.

Rules:

1. If the answer is clearly available in the context:
   - Answer using the context.
   - Be accurate and concise.

2. If the context contains partial information:
   - Use the context first.
   - Then complete the answer using your own general knowledge.
   - Clearly mention which part was not found in the documents.

3. If the answer is NOT available in the context at all:
   - Say:
     "The uploaded documents do not contain information about this."
   - Then provide a helpful answer using your general knowledge.

4. Never pretend the context contains information that it does not contain.

5. Do not hallucinate citations, page numbers, or facts from the documents.

Context:
{context}

Question:
{question}

Answer:
"""

# ===========================
# Validation
# ===========================


def reload_settings():
    """Re-read all settings from environment variables (after .env changes)."""
    global DATA_FOLDER, UPLOAD_FOLDER, VECTORSTORE_FOLDER
    global DB_FAISS_PATH, CHUNKS_PATH, DOCS_PATH
    global GROQ_API_KEY, GROQ_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS
    global EMBEDDING_MODEL_NAME
    global FAISS_SEARCH_K, BM25_SEARCH_K, BM25_WEIGHT, FAISS_WEIGHT
    global CHUNK_SIZE, CHUNK_OVERLAP
    global CHAIN_TYPE, RETURN_SOURCE_DOCUMENTS
    global \
        STREAMLIT_PAGE_TITLE, \
        STREAMLIT_PAGE_ICON, \
        MAX_SOURCE_DOCS_DISPLAY, \
        DISPLAY_CONTENT_LENGTH
    global MAX_FILE_SIZE_MB

    DATA_FOLDER = Path(os.getenv("DATA_FOLDER", "data"))
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "data/uploaded_files"))
    VECTORSTORE_FOLDER = Path(os.getenv("VECTORSTORE_FOLDER", "vectorstore"))
    DB_FAISS_PATH = VECTORSTORE_FOLDER / os.getenv("DB_FAISS_NAME", "db_faiss")
    CHUNKS_PATH = VECTORSTORE_FOLDER / os.getenv("CHUNKS_FILE", "chunks.pkl")
    DOCS_PATH = VECTORSTORE_FOLDER / os.getenv("DOCS_FILE", "docs.pkl")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    EMBEDDING_MODEL_NAME = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    FAISS_SEARCH_K = int(os.getenv("FAISS_SEARCH_K", "3"))
    BM25_SEARCH_K = int(os.getenv("BM25_SEARCH_K", "5"))
    BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
    FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", "0.7"))

    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

    CHAIN_TYPE = os.getenv("CHAIN_TYPE", "stuff")
    RETURN_SOURCE_DOCUMENTS = (
        os.getenv("RETURN_SOURCE_DOCUMENTS", "true").lower() == "true"
    )

    STREAMLIT_PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "Ask Chatbot")
    STREAMLIT_PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", " ")
    MAX_SOURCE_DOCS_DISPLAY = int(os.getenv("MAX_SOURCE_DOCS_DISPLAY", "5"))
    DISPLAY_CONTENT_LENGTH = int(os.getenv("DISPLAY_CONTENT_LENGTH", "500"))

    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))


def validate_settings():
    """Validate that required settings are configured"""
    errors = []

    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        errors.append("GROQ_API_KEY not set in .env")

    if not UPLOAD_FOLDER.exists():
        try:
            UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create UPLOAD_FOLDER: {e}")

    if not VECTORSTORE_FOLDER.exists():
        try:
            VECTORSTORE_FOLDER.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create VECTORSTORE_FOLDER: {e}")

    if LLM_TEMPERATURE < 0 or LLM_TEMPERATURE > 2:
        errors.append(f"LLM_TEMPERATURE must be between 0 and 2, got {LLM_TEMPERATURE}")

    if FAISS_SEARCH_K < 1:
        errors.append(f"FAISS_SEARCH_K must be >= 1, got {FAISS_SEARCH_K}")

    if BM25_SEARCH_K < 1:
        errors.append(f"BM25_SEARCH_K must be >= 1, got {BM25_SEARCH_K}")

    if CHUNK_SIZE < 100:
        errors.append(f"CHUNK_SIZE should be >= 100, got {CHUNK_SIZE}")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_SIZE ({CHUNK_SIZE})"
        )

    return errors


# ===========================
# Debug Info
# ===========================


def print_settings():
    """Print all settings (useful for debugging)"""
    print("=" * 60)
    print("RAG CHATBOT - SETTINGS")
    print("=" * 60)

    print("\n[PATHS]")
    print(f"  UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"  VECTORSTORE_FOLDER: {VECTORSTORE_FOLDER}")
    print(f"  DB_FAISS_PATH: {DB_FAISS_PATH}")
    print(f"  CHUNKS_PATH: {CHUNKS_PATH}")

    print("\n[LLM]")
    print(f"  GROQ_MODEL_NAME: {GROQ_MODEL_NAME}")
    print(f"  LLM_TEMPERATURE: {LLM_TEMPERATURE}")
    print(f"  LLM_MAX_TOKENS: {LLM_MAX_TOKENS}")

    print("\n[EMBEDDING]")
    print(f"  EMBEDDING_MODEL_NAME: {EMBEDDING_MODEL_NAME}")

    print("\n[RETRIEVER]")
    print(f"  FAISS_SEARCH_K: {FAISS_SEARCH_K}")
    print(f"  BM25_SEARCH_K: {BM25_SEARCH_K}")
    print(f"  BM25_WEIGHT: {BM25_WEIGHT}")
    print(f"  FAISS_WEIGHT: {FAISS_WEIGHT}")

    print("\n[TEXT PROCESSING]")
    print(f"  CHUNK_SIZE: {CHUNK_SIZE}")
    print(f"  CHUNK_OVERLAP: {CHUNK_OVERLAP}")

    print("\n[CHAIN]")
    print(f"  CHAIN_TYPE: {CHAIN_TYPE}")
    print(f"  RETURN_SOURCE_DOCUMENTS: {RETURN_SOURCE_DOCUMENTS}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_settings()
    errors = validate_settings()
    if errors:
        print("\n[VALIDATION ERRORS]")
        for error in errors:
            print(f"  ✗ {error}")
    else:
        print("\n✓ All settings validated successfully")
