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

def show_both(question):
    question += ' please respond with a long explanation'
    generated_response, retrieved_documents, distances = generate_response(question, top_k=5)
    
    print(f"\n{'='*30} Retrieved Documents {'='*30}")
    for i, (doc, distance) in enumerate(zip(retrieved_documents, distances)):
        print(f"\nResult {i+1} (Distance: {distance}):\n{doc['snippet']}")
    
    print(f"\n{'='*30} Generated Response {'='*30}")
    print(generated_response)

def show_response_only(question):
    question += ' please respond with a long explanation'
    generated_response, _, _ = generate_response(question, top_k=5)
    
    print(f"\n{'='*30} Generated Response {'='*30}")
    print(generated_response)


if __name__ == "__main__":
    while True:
        question = input('Ask a question (or type "exit" to quit): ')
        if question.lower() == 'exit':
            break
        mode = input('Type "both" to see both documents and response, or "response" to see only the generated response: ').lower()
        if mode == 'both':
            show_both(question)
        elif mode == 'response':
            show_response_only(question)
        else:
            print("Invalid input. Please type 'both' or 'response'.")
