# ===========================
# Retrievers Module
# ===========================

import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from settings import (
    FAISS_SEARCH_K,
    BM25_SEARCH_K,
    BM25_WEIGHT,
    FAISS_WEIGHT,
)

from modules.embeddings import (
    get_vectorstore,
    load_chunks,
)


def get_bm25_retriever():
    """Build BM25 retriever from saved chunks"""
    chunks = load_chunks()
    
    if chunks is None:
        return None
    
    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = BM25_SEARCH_K
        return bm25_retriever
    except Exception as e:
        st.error(f"Error building BM25 retriever: {str(e)}")
        return None


@st.cache_resource
def get_hybrid_retriever():
    """
    Combine FAISS vector search with BM25 keyword search
    Creates an ensemble retriever with weighted combination
    """
    try:
        db = get_vectorstore()
        
        if db is None:
            return None
            
        faiss_retriever = db.as_retriever(search_kwargs={"k": FAISS_SEARCH_K})
        
        bm25_retriever = get_bm25_retriever()
        
        if bm25_retriever is None:
            # Fall back to FAISS only if BM25 fails
            st.info("Using FAISS retriever only (BM25 not available)")
            return faiss_retriever
        
        # Combine retrievers with weights from settings
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[BM25_WEIGHT, FAISS_WEIGHT]
        )
        
        return hybrid_retriever
    
    except Exception as e:
        st.error(f"Error creating hybrid retriever: {str(e)}")
        return None
