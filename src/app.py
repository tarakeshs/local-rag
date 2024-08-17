import os
import torch

# Force PyTorch to use the CPU instead of MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Alternatively, disable MPS altogether
torch.device("cpu")


import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from utils.faiss_utils import process_and_store_documents, retrieve_documents
from ollama_client import query_ollama  # Import the Ollama client

st.title("Local RAG: Langchain + FAISS + Ollama + Streamlit")

uploaded_file = st.file_uploader("Upload a CSV or PDF", type=["csv", "pdf"])

if uploaded_file is not None:
    try:
        if uploaded_file.type == "text/csv":
            df = pd.read_csv(uploaded_file)
            st.write("CSV Loaded")
            st.dataframe(df)
            text = df.to_string()

        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                if i % 10 == 0:  # Process 10 pages at a time
                    st.write(f"Processing page {i+1}")
                text += page.extract_text()
            st.write("PDF Loaded")
            st.text_area("PDF Content", text, height=300)

        # Process the text and store in FAISS
        faiss_index = process_and_store_documents(text)

        # User Input
        user_query = st.text_input("Ask a question about the document")

        if user_query:
            # Retrieve relevant documents and generate response
            context = retrieve_documents(faiss_index, user_query)
            st.write("Retrieved Context:", context)

            # Generate response using Ollama
            prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {user_query}"
            response = query_ollama(prompt)
            st.write("Generated Response:", response)

    except Exception as e:
        st.error(f"An error occurred: {e}")
