# Steps to Create data for llm 

from loader import load_pdf_files_directory # importing load PDF function from loader.py
from langchain_text_splitters import RecursiveCharacterTextSplitter # use to make chunks 
from langchain_community.embeddings import HuggingFaceEmbeddings # hugging face embidding model
from langchain_community.vectorstores import FAISS #for vector storage 



# Step 2 : Create Chunks 

def create_chunks(data):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
    text_chunks = splitter.split_documents(data)

    return text_chunks



# Step 3 : Create Vector Embiddings

# loading embedding model 
def get_embedding_model():

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model  












# Step 1 : Load raw Pdf 

# while giving the path give directory path only 
# Data_path = "data/pdf/"
# document = load_pdf_files_directory(Data_path)
# print(f"Lenths of Document Pages : {len(document)}\n")
# print("Data Loaded Succesfully")

# # does chunkking of document
# chunks = create_chunks(document)
# print(f"Lenth of Text Chunk {len(chunks)}\n")
# print("Chunking Done\n")

# # loads the embedding model
# embidding_model = get_embedding_model()


# Step 4 : Store Embiddings in FAISS
# DB_FAISS_PATH = "vectorstore/db_faiss"
# db = FAISS.from_documents(chunks,embidding_model)
# db.save_local(DB_FAISS_PATH)
# print("Succesfully Created Vector Storage")