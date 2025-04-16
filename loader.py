from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdf_loader(directory_path):
    loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents(documents, chunk_size=300, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
