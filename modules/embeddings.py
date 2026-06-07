# ===========================
# Embeddings Module
# ===========================

import streamlit as st
from pathlib import Path
import pickle

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from settings import (
    EMBEDDING_MODEL_NAME,
    DB_FAISS_PATH,
    CHUNKS_PATH,
)


@st.cache_resource
def load_embedding_model():
    """Load HuggingFace embedding model"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )
        return embedding_model
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None


@st.cache_resource
def get_vectorstore():
    """Load FAISS vector store from disk"""
    try:
        embedding_model = load_embedding_model()
        
        if embedding_model is None:
            st.error("Cannot load embedding model")
            return None

        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    
    except FileNotFoundError:
        st.warning("Vector store not found. Please upload and process documents first.")
        return None
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None


def build_vector_store(chunks, embedding_model):
    """Build and save FAISS vector store"""
    try:
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        
        # Save chunks for BM25 retriever
        save_chunks(chunks)
        
        return True
    except Exception as e:
        st.error(f"Error building vector store: {str(e)}")
        return False


def save_chunks(chunks):
    """Save document chunks to pickle file"""
    try:
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(chunks, f)
    except Exception as e:
        st.error(f"Error saving chunks: {str(e)}")


def load_chunks():
    """Load document chunks from pickle file"""
    try:
        if CHUNKS_PATH.exists():
            with open(CHUNKS_PATH, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading chunks: {str(e)}")
    return None
