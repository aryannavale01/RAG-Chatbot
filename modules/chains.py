# ===========================
# Chains Module
# ===========================

import streamlit as st
from langchain.chains import RetrievalQA

from settings import (
    CHAIN_TYPE,
    RETURN_SOURCE_DOCUMENTS,
)

from modules.llm import load_llm, set_custom_prompt
from modules.retrievers import get_hybrid_retriever


@st.cache_resource
def build_qa_chain():
    """
    Build QA chain with hybrid retriever
    Uses settings for chain type and document return options
    """
    try:
        retriever = get_hybrid_retriever()
        
        if retriever is None:
            st.error("Failed to initialize retriever")
            return None
        
        llm = load_llm()
        
        if llm is None:
            st.error("Failed to initialize LLM")
            return None
        
        prompt = set_custom_prompt()
        
        if prompt is None:
            st.error("Failed to set prompt")
            return None

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type=CHAIN_TYPE,
            retriever=retriever,
            return_source_documents=RETURN_SOURCE_DOCUMENTS,
            chain_type_kwargs={
                "prompt": prompt
            }
        )

        return qa_chain
    
    except Exception as e:
        st.error(f"Error building QA chain: {str(e)}")
        return None
