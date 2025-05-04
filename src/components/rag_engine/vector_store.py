import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, dimension: int, documents: List[str] = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings
            documents: Optional list of documents to initialize the store with
        """
        self.dimension = dimension
        self.documents = documents if documents else []
        self.index = None
        
    def build_index(self, embeddings: np.ndarray, n_clusters: int = 100):
        """
        Build and train the FAISS index.
        
        Args:
            embeddings: Numpy array of document embeddings
            n_clusters: Number of clusters for IVF index
        """
        nlist = min(n_clusters, len(self.documents))
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        self.index.train(embeddings)
        self.index.add(embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int = 2) -> List[Tuple[str, float]]:
        """
        Search the index for similar documents.
        
        Args:
            query_embedding: Embedding of the query
            k: Number of results to return
            
        Returns:
            List of (document, similarity_score) tuples
        """
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            results.append((self.documents[idx], 1 - score))
        return results
    
    def add_documents(self, documents: List[str], embeddings: np.ndarray):
        """
        Add new documents to the store.
        
        Args:
            documents: List of new document texts
            embeddings: Numpy array of corresponding embeddings
        """
        if self.index is None:
            self.build_index(embeddings)
        else:
            self.index.add(embeddings)
        self.documents.extend(documents)