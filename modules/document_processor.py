# ===========================
# Document Processing Module
# ===========================

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

from modules.loader import (
    load_pdf, load_txt, load_csv, load_docx, load_doc,
    load_xlsx, load_pptx, load_md, load_html, load_xml, load_json
)


def create_chunks(documents):
    """
    Split documents into chunks using RecursiveCharacterTextSplitter
    """
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(documents)
        return chunks
    except Exception as e:
        st.error(f"Error creating chunks: {str(e)}")
        return []


def process_uploaded_file(file_path, file_name):
    """
    Process uploaded file and extract documents
    Supports multiple file formats: PDF, TXT, CSV, DOC, DOCX, JSON, HTML, MD, XML, XLSX, PPTX
    """
    documents = []
    file_name_lower = file_name.lower()

    try:
        # PDF
        if file_name_lower.endswith(".pdf"):
            docs = load_pdf(file_path)
            documents.extend(docs)

        # TXT
        elif file_name_lower.endswith(".txt"):
            docs = load_txt(file_path)
            documents.extend(docs)

        # CSV
        elif file_name_lower.endswith(".csv"):
            docs = load_csv(file_path)
            documents.extend(docs)

        # DOC (but not DOCX)
        elif file_name_lower.endswith(".doc") and not file_name_lower.endswith(".docx"):
            docs = load_doc(file_path)
            documents.extend(docs)

        # DOCX
        elif file_name_lower.endswith(".docx"):
            docs = load_docx(file_path)
            documents.extend(docs)

        # JSON
        elif file_name_lower.endswith(".json"):
            docs = load_json(file_path)
            documents.extend(docs)

        # HTML
        elif file_name_lower.endswith(".html") or file_name_lower.endswith(".htm"):
            docs = load_html(file_path)
            documents.extend(docs)

        # Markdown
        elif file_name_lower.endswith(".md"):
            docs = load_md(file_path)
            documents.extend(docs)

        # XML
        elif file_name_lower.endswith(".xml"):
            docs = load_xml(file_path)
            documents.extend(docs)

        # Excel
        elif file_name_lower.endswith(".xlsx") or file_name_lower.endswith(".xls"):
            docs = load_xlsx(file_path)
            documents.extend(docs)

        # PowerPoint
        elif file_name_lower.endswith(".pptx") or file_name_lower.endswith(".ppt"):
            docs = load_pptx(file_path)
            documents.extend(docs)

        else:
            st.warning(f"{file_name} type not supported")
            return documents

    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")
        return documents

    if documents:
        st.success(f"✓ Loaded {len(documents)} documents from {file_name}")
    
    return documents
