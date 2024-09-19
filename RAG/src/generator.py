from retriever import retrieve
from dotenv import load_dotenv
import cohere
import os 

load_dotenv()
qa_model = cohere.Client(os.getenv('COHERE_API_KEY'))

def generate_response(question, top_k):
    retrieved_documents, distances = retrieve(question, top_k=top_k)

    response  = qa_model.chat(
        message=question, 
        documents=retrieved_documents,
        max_tokens=4000, 
    )
    return [response.text, retrieved_documents, distances]

if __name__ == "__main__":
    while True:
        question = input('Ask a question (or type "exit" to quit): ')
        if question.lower() == 'exit':
            break
        question += 'please response long as possible'
        generated_response, retrieved_documents, distances = generate_response(question, top_k=5)
        print(f"\nTop {len(retrieved_documents)} Retrieved Documents:")
        for i, (doc, distance) in enumerate(zip(retrieved_documents, distances)):
            print(f"\nResult {i+1} (Distance: {distance}):\n{doc['snippet']}")
        
        print("\n\nGenerated Response:")
        print(generated_response)