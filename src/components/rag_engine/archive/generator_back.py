from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any

class PolicyResponseGenerator:
    """Generates responses using retrieved context and LLM"""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template(
            """You are a travel insurance policy expert. Answer the user's question 
            based only on the following context. Be concise and accurate:

            Context: {context}

            Question: {question}

            Answer:"""
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        """Create the RAG chain"""
        return (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
        )

    def generate_response(self, question: str, context: str) -> Dict[str, Any]:
        """Generate a response using the provided context"""
        print("\n--- LLM Context (DEBUG) ---")  # Debugging
        print(context)
        print("-" * 20)
        return self.chain.invoke({"question": question, "context": context})