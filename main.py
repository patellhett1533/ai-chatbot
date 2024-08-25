from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="llama3", n_predict=1)

template = '''
Answer then question below.

Here is the converstion history: {context}

Question: {question}

Answer: 
'''

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model


def handle_conversation():
    context = ""
    print("Welcome to the chatbot. Type 'exit' to exit.")
    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break
        result = chain.invoke({"context": context, "question": question})
        print("Bot: ", result)
        context += f"\nUser: {question}\nAI: {result}"


if __name__ == "__main__":
    handle_conversation()
