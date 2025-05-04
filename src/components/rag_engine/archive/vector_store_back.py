from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os
from typing import Optional, List


class PolicyVectorStore:
    """Manages the FAISS vector store for policy documents"""

    def __init__(self, embedding_model: Embeddings, index_dir: str = "data/policies/vector_store"):
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.vector_store = None

    def create_vector_store(self, documents: list[Document]) -> None:
        """Create a new FAISS vector store from documents"""
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

    def save_vector_store(self) -> None:
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(self.index_dir)

    def load_vector_store(self) -> None:
        """Load vector store from disk"""
        self.vector_store = FAISS.load_local(
            self.index_dir,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query, k=k)

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the vector store for debugging.
        This is a workaround using similarity_search with an empty query.
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search(query="", k=self.vector_store.index.ntotal)