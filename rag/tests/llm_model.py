import ollama

while True:
    user_input = input("Type anything (or exit to leave): ")
    if user_input.lower() == "exit":
        break
    user_input += ' please respond as long as possible'
    response = ollama.chat(model='gemma2:2b', messages=[{
        'role': 'user',
        'content': user_input,
    }])
    print("AI: " + response['message']['content'])
