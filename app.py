# ===========================
# Imports
# ===========================

import os
import streamlit as st
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from config import CUSTOM_RAG2_PROMPT_TEMPLATE

# imports for file processing 
from loader import *

from data_pipeline import(
    create_chunks,
    
)

# ===========================
# Load ENV
# ===========================

load_dotenv()

UPLOAD_FOLDER = Path("data/uploaded_files")
DB_FAISS_PATH = Path("vectorstore/db_faiss")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
DB_FAISS_PATH.mkdir(parents=True, exist_ok=True)
# ===========================
# LLM
# ===========================

@st.cache_resource
def load_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )

    return llm


# ===========================
# Prompt
# ===========================

def set_custom_prompt():

    prompt = PromptTemplate(
        template=CUSTOM_RAG2_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt


# ===========================
# Embedding Model
# ===========================

@st.cache_resource
def load_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


# ===========================
# Vector Store
# ===========================

@st.cache_resource
def get_vectorstore():

    embedding_model = load_embedding_model()

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return db


# ===========================
# QA Chain
# ===========================

@st.cache_resource
def build_qa_chain():

    db = get_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": set_custom_prompt()
        }
    )

    return qa_chain

# ===========================
# For Uploaded Documnets
# ===========================

# function to build vectorstore again
def build_vector_store(chunks,embedding_model):
    db = FAISS.from_documents(chunks,embedding_model)
    db.save_local(DB_FAISS_PATH)

# function to select the loder for uploaded files 
def process_uploaded_file(file_path, file_name):

    documents = []

    file_name = file_name.lower()

    try:

        # PDF
        if file_name.endswith(".pdf"):
            docs = load_pdf(file_path)
            documents.extend(docs)

        # TXT
        elif file_name.endswith(".txt"):
            docs = load_txt(file_path)
            documents.extend(docs)

        # CSV
        elif file_name.endswith(".csv"):
            docs = load_csv(file_path)
            documents.extend(docs)

        # DOC
        elif file_name.endswith(".doc"):
            docs = load_doc(file_path)
            documents.extend(docs)

        # DOCX
        elif file_name.endswith(".docx"):
            docs = load_docx(file_path)
            documents.extend(docs)

        # JSON
        elif file_name.endswith(".json"):
            docs = load_json(file_path)
            documents.extend(docs)

        # HTML
        elif file_name.endswith(".html") or file_name.endswith(".htm"):
            docs = load_html(file_path)
            documents.extend(docs)

        # Markdown
        elif file_name.endswith(".md"):
            docs = load_md(file_path)
            documents.extend(docs)

        # XML
        elif file_name.endswith(".xml"):
            docs = load_xml(file_path)
            documents.extend(docs)

        # Excel
        elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            docs = load_xlsx(file_path)
            documents.extend(docs)

        # PowerPoint
        elif file_name.endswith(".pptx") or file_name.endswith(".ppt"):
            docs = load_pptx(file_path)
            documents.extend(docs)

        else:
            st.warning(f"{file_name} type not supported")

    except Exception as e:
        st.error(f"Error processing {file_name}: {str(e)}")

    return documents


# ===========================
# Streamlit UI
# ===========================

def main():
    # chat initalization
    st.title("Ask Chatbot")
    try:
        qa_chain = build_qa_chain()
    except:
        qa_chain = None
    
    if qa_chain is None:
        st.warning("Please upload and process documents first.")

    # message memorry management
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # sources memorry management
    if "sources" not in st.session_state:
        st.session_state.sources = []

    # sidebar
    with st.sidebar:

        st.header("Upload PDFs")

        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf", "txt", "csv", "doc", "docx"],
            accept_multiple_files=True
        )

        if uploaded_files:

            all_documents = []
            st.success("Files Uploaded Successfully!")
            
            # buiiding vectorstore for document
            with st.spinner("Building Vector Store..."):
                if st.button("Process PDFs"):

                    for file in uploaded_files:

                        save_path = os.path.join(
                            UPLOAD_FOLDER,
                            file.name
                        )

                        with open(save_path, "wb") as f:
                            f.write(file.getbuffer())

                        st.write(f"Saved: {file.name}")
                        file_type = Path(file.name).suffix.lower()
                        documet = process_uploaded_file(save_path,file_type)
                        all_documents.extend(documet)

                    # st.info("Creating chunks...")
                    chunks = create_chunks(all_documents)
                    # geting embidding model 
                    # st.info("Loading embedding model...")
                    embedding_model = load_embedding_model()
                    # vectorstore builded 
                    # st.info("Creating vectorstore...")
                    build_vector_store(chunks,embedding_model)
                    st.cache_resource.clear()
                    st.info("Done...")
        
        st.header("Sources")

        if st.session_state.sources:

            for i, doc in enumerate(st.session_state.sources):

                with st.expander(f"Source {i+1}"):

                    st.write(doc.page_content[:500])

                    st.write(doc.metadata)

        else:
            st.info("No sources available")

    # tacking prompt from user 
    prompt = st.chat_input("Ask your question")

    if prompt:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        # shows the promopt in chat 
        with st.chat_message("user"):
            st.markdown(prompt)

        # shows thinking spinner 
        with st.spinner("Thinking..."):

            response = qa_chain.invoke({
                "query": prompt
            })

            result = response["result"]
            source_docs = response["source_documents"]
            # SAVE SOURCES
            st.session_state.sources = source_docs

        # shows assistant message on chat
        with st.chat_message("assistant"):
            st.markdown(result)
            

        st.session_state.messages.append({
            "role": "assistant",
            "content": result
        })




if __name__ == "__main__":
    main()