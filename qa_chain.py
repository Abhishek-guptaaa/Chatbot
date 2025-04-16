from langchain.chains import RetrievalQA
from groq import Groq
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

def initialize_model():
    api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(model="llama3-8b-8192", api_key=api_key)
    return model

def initialize_qa_chain(vectorstore, model):
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=model, retriever=retriever)
