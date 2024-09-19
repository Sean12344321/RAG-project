import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from preprocess import all_splits

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_docs_embeddings(documents):
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(texts)
    return np.array(embeddings)

def get_query_embedding(query):
    query_embedding = embedding_model.encode([query])  
    return np.array(query_embedding)

# Store embeddings in FAISS
embeddings = get_docs_embeddings(all_splits)
embedding_dim = embeddings.shape[1] 
index = faiss.IndexFlatL2(embedding_dim)  
index.add(embeddings)