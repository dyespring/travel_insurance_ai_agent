from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional

class PolicyRetriever:
    def __init__(self, vector_store, llm: Optional[BaseLLM] = None):
        self.base_retriever = vector_store.as_retriever(
            search_type="similarity",  # Keep similarity search
            search_kwargs={
                "k": 2,
                "score_threshold": 0.5
                # "metric": "cos"  # Add cosine similarity metric
            }
        )
        self.compressor = LLMChainExtractor.from_llm(llm) if llm else None

    def invoke(self, query: str) -> List[Document]:  # This is the main method now
        if self.compressor:
            retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.base_retriever
            )
        else:
            retriever = self.base_retriever

        docs = retriever.invoke(query)  # Get the documents
        print("\n--- Retrieved Documents (DEBUG) ---")  # Debugging
        if docs:
            for doc in docs:
                print(doc.page_content)
        else:
            print("No documents retrieved.")
        return docs

    # For backward compatibility, you can add this:
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Deprecated method, use invoke() instead"""
        return self.invoke(query)