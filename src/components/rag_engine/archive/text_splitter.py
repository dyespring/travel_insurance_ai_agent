from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

class PolicyTextSplitter:
    """Handles splitting of policy documents into chunks"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", ", ", " "]  # Added separators
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.splitter.split_documents(documents)