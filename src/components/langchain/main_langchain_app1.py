from config_langchain import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from config_langchain import OPENAI_API_KEY
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from router import route_input  # Use  logic engine

def main():
    print("Welcome to the Insurance Chatbot!")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = route_input(user_input)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
