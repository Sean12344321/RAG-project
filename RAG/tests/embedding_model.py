from sentence_transformers import SentenceTransformer
import numpy as np

def test_embedding_model():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    documents = [
        "Hello, world! This is a test document.",
        "image processing is an very important topic, we should master it to become a good image processing engineer",
        "CNN and RNN are popular neural network architectures used in deep learning.",
        "Breakthroughs in AI have been seen in natural language processing.",
        "The Transformer architecture has revolutionized the field of NLP.",
        "Chain of thought is a new startup that is making waves in the AI industry."
    ]
    print("Testing similarity between input sentence and documents")
    for i, doc in enumerate(documents):
        print(f"document {i+1} : {doc}")
    input_sentence = input("Enter a sentence: ")
    
    input_embedding = embedding_model.encode([input_sentence])
    document_embeddings = embedding_model.encode(documents)
    
    similarities = []
    for doc_embedding in document_embeddings:
        similarity = np.dot(input_embedding, doc_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(doc_embedding))
        similarities.append(similarity)
    
    print("Similarities:")
    for i, similarity in enumerate(similarities):
        print(f"Document {i + 1}: {similarity}")

if __name__ == "__main__":
    test_embedding_model()
