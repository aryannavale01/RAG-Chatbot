
# Step 1 : Load Row Data 
# In this module we implimented some functions wich are used to load row data and this function 
# further call in data pipeline file 
# ===========================================================================================


# importing modules 
from langchain_community.document_loaders import DirectoryLoader # use to load directory 
from langchain_community.document_loaders import PyMuPDFLoader  # use to load pdf from directory or single file 

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    JSONLoader,
    UnstructuredXMLLoader,
)



# this function returns document data structure of langchain 
def load_pdf_files_directory(data):

    return DirectoryLoader(
        path=data,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
        use_multithreading=True
    ).load()



# ===========================
# PDF
# ===========================

def load_pdf(file_path):

    return PyMuPDFLoader(
        str(file_path)
    ).load()


# ===========================
# TXT
# ===========================

def load_txt(file_path):

    return TextLoader(
        str(file_path),
        encoding="utf-8"
    ).load()


# ===========================
# CSV
# ===========================

def load_csv(file_path):

    return CSVLoader(
        str(file_path)
    ).load()


# ===========================
# DOCX
# ===========================

def load_docx(file_path):

    return Docx2txtLoader(
        str(file_path)
    ).load()


# ===========================
# DOC
# ===========================

def load_doc(file_path):

    return UnstructuredWordDocumentLoader(
        str(file_path)
    ).load()


# ===========================
# XLSX
# ===========================

def load_xlsx(file_path):

    return UnstructuredExcelLoader(
        str(file_path)
    ).load()


# ===========================
# PPTX
# ===========================

def load_pptx(file_path):

    return UnstructuredPowerPointLoader(
        str(file_path)
    ).load()


# ===========================
# MARKDOWN
# ===========================

def load_md(file_path):

    return UnstructuredMarkdownLoader(
        str(file_path)
    ).load()


# ===========================
# HTML
# ===========================

def load_html(file_path):

    return UnstructuredHTMLLoader(
        str(file_path)
    ).load()


# ===========================
# XML
# ===========================

def load_xml(file_path):

    return UnstructuredXMLLoader(
        str(file_path)
    ).load()


# ===========================
# JSON
# ===========================

def load_json(file_path):

    return JSONLoader(
        file_path=str(file_path),
        jq_schema=".",
        text_content=False
    ).load()