from config_langchain import OPENAI_API_KEY
# pip install -U langchain-openai
# from langchain.chat_models import ChatOpenAI
# pip install -U langchain-community
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferMemory
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)


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
