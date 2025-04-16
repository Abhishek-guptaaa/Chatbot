import streamlit as st
import os
from bs4 import BeautifulSoup
import tempfile
import requests
from loader import pdf_loader, split_documents
from embedder import create_embeddings, create_vectorstore
from qa_chain import initialize_model, initialize_qa_chain
from tqdm import tqdm
from embedder import create_embeddings, create_vectorstore



st.set_page_config(page_title="📄 PDF Question Answering App", layout="centered")
st.title("📄 PDF Question Answering using LangChain + Groq")

# Sidebar Inputs
st.sidebar.title("📁 Load your PDF")

load_option = st.sidebar.radio("Choose input method:", ["Upload File", "Website URL"])


uploaded_files = None
pdf_url = None

#if load_option == "Upload File":
uploaded_files = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
#elif load_option == "Website URL":
   # pdf_url = st.sidebar.text_input("Enter URL:")


# Process PDFs
if (uploaded_files or pdf_url) and st.sidebar.button("Process PDFs"):
    with st.spinner("Preparing your document..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            if uploaded_files:
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
            elif pdf_url:
                try:
                    file_data = requests.get(pdf_url)
                    file_name = os.path.basename(pdf_url)
                    file_path = os.path.join(temp_dir, file_name)
                    with open(file_path, "wb") as f:
                        f.write(file_data.content)
                except Exception as e:
                    st.error(f"Failed to download PDF from URL: {e}")
                    st.stop()

            # Load and process documents
            docs = pdf_loader(temp_dir)
            chunks = split_documents(docs)

            with st.spinner("Generating embeddings and creating vector store..."):
                embeddings = create_embeddings()
                chunks = list(tqdm(chunks, desc="Processing documents"))
                vectorstore = create_vectorstore(chunks, embeddings)

            with st.spinner("Initializing model..."):
                model = initialize_model()
                qa_chain = initialize_qa_chain(vectorstore, model)
                st.session_state.qa_chain = qa_chain
                st.success("✅ Model and RetrievalQA chain initialized")

# Question Answering Section
if "qa_chain" in st.session_state:
    st.subheader("💬 Ask a question from the document")
    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Getting answer..."):
            response = st.session_state.qa_chain.run(query)
        st.write("📢 **Answer:**", response)


