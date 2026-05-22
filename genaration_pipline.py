# ===========================
# Imports
# ===========================

from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from config import CUSTOM_RAG_PROMPT_TEMPLATE

from dotenv import load_dotenv
import os

# ===========================
# Load ENV
# ===========================

load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# ===========================
# Load LLM
# ===========================

# function to load llm 
def load_llm():

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b",
        temperature=0.5
    )

    return llm

# ===========================
# Custom Prompt
# ===========================


# function to set custom prompt 
def set_custom_prompt():

    prompt = PromptTemplate(
        template=CUSTOM_RAGPROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    return prompt

# ===========================
# Embedding Model
# ===========================
# function to load embedding model 
def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


# ===========================
# Load FAISS DB
# ===========================

# function to load vectorstore 
def get_vectorstore(embedding_model):

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# ===========================
# QA Chain
# ===========================

def build_qa_chain(db):
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
# Ask Query
# ===========================
embedding_model = load_embedding_model()
db = get_vectorstore(embedding_model)

user_query = input("Write Query Here: ")
qa_chain = build_qa_chain(db)
response = qa_chain.invoke({
    "query": user_query
})

# # ===========================
# # Output
# # ===========================

print("==="*30)
print("\nUser:")
print(user_query)
print("\nRESULT:")
print(response["result"])
print("==="*30)

# print("\nSOURCE DOCUMENTS:\n")
# print(response["source_documents"])