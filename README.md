How to Run this file

create conda env 

conda create -p <env_name> python==3.9 -y

conda activate suppose that your your env name venv 

command: conda activate venv/

install dependency

command: pip install -r requirements.txt
 
 
 
 
 
 
 1. loader.py – PDF Loader and Splitter
 2. embedder.py – Embedding & FAISS Vector Store
 3. qa_chain.py – Initialize RetrievalQA Chain
 4. app.py - Streamlit ui
 you can : streamlit run app.py 