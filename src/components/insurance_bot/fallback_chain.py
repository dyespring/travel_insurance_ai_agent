from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from config_langchain import OPENAI_API_KEY

def get_fallback_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are a helpful assistant for an insurance company."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.5)
    memory = ConversationBufferMemory(return_messages=True)

    return LLMChain(llm=llm, prompt=prompt, memory=memory)
