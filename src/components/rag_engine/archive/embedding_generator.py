from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import numpy as np

class PolicyEmbeddingGenerator:
    """Generates embeddings for policy document chunks"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        return self.embedding_model.embed_documents(texts)