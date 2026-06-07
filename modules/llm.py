# ===========================
# LLM Module
# ===========================

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from settings import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    CUSTOM_RAG2_PROMPT_TEMPLATE,
)


@st.cache_resource
def load_llm():
    """Load Groq LLM with configured settings"""
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None


def set_custom_prompt():
    """Set custom prompt template for RAG"""
    try:
        prompt = PromptTemplate(
            template=CUSTOM_RAG2_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
        return prompt
    except Exception as e:
        st.error(f"Error setting prompt: {str(e)}")
        return None
