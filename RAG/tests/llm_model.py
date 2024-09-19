import cohere
import os 
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY'))

while True:
	question = input('Ask a question (or type "exit" to quit): ')
	if question.lower() == 'exit':
		break
	question += 'please response long as possible'
	response = co.chat(
		message=question,
		max_tokens=4000, 
	)
	print(response.text)