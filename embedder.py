from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vectorstore(documents, embeddings):
    return FAISS.from_documents(documents, embeddings)


