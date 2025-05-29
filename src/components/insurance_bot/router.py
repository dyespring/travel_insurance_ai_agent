from fixed_responses import EXACT_RESPONSES
from fallback_chain import get_fallback_chain
from langchain.chat_models import ChatOpenAI

chain = get_fallback_chain()

def route_input(user_input: str) -> str:
    if user_input in EXACT_RESPONSES:
        return EXACT_RESPONSES[user_input]
    return chain.run(input=user_input)
