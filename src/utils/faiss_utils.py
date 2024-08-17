import faiss
import numpy as np
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Text Splitter for document chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

def process_and_store_documents(text):
    # Split the document into chunks
    documents = [
        Document(page_content=chunk, metadata={})  # Ensure metadata is an empty dictionary
        for chunk in text_splitter.split_text(text)
    ]
    
    # Generate embeddings for the document chunks
    embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])
    
    # Convert embeddings to a numpy array
    embedding_vectors = np.array(embeddings)
    
    # Create an empty FAISS index
    index = faiss.IndexFlatL2(embedding_vectors.shape[1])  # Using L2 distance

    # Create an InMemoryDocstore
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(documents)})
    index_to_docstore_id = {i: i for i in range(len(documents))}

    # Initialize FAISS vector store
    faiss_index = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    faiss_index.add_texts([doc.page_content for doc in documents], embeddings)

    return faiss_index

def retrieve_documents(faiss_index, query):
    query_embedding = embedding_model.embed_query(query)
    results = faiss_index.similarity_search_by_vector(query_embedding, k=3)
    return " ".join([doc.page_content for doc in results])
