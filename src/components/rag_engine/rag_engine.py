from typing import List, Tuple, Optional, Union
from pathlib import Path
from .embedder import DocumentEmbedder
from .vector_store import VectorStore
from .generator import ResponseGenerator

class TravelInsuranceRAGEngine:
    def __init__(self, policy_source: Union[List[str], str, Path] = "components/rag_engine/policy_docs"): # add file path here
        """
        Initialize the RAG Engine with travel insurance policy documents.
        
        Args:
            policy_source: Either:
                - List of policy document texts
                - Path to directory containing policy files
                - Path to single policy file
        """
        self.embedder = DocumentEmbedder()
        
        # Handle different input types
        if isinstance(policy_source, (str, Path)):
            # Process files from directory or single file
            self.chunks, embeddings = self.embedder.process_policy_files(policy_source)
            self.documents = self.chunks  # Using chunks as documents
        else:
            # Process list of texts directly
            self.documents = policy_source
            self.chunks = self.embedder.chunk_documents(self.documents)
            embeddings = self.embedder.embed_documents(self.documents)
        
        # Initialize vector store with the embeddings
        self.vector_store = VectorStore(
            dimension=embeddings.shape[1], 
            documents=self.chunks  # Store chunks for retrieval
        )
        self.vector_store.build_index(embeddings)
        
        self.generator = ResponseGenerator()
        
        # Prompt template
        self.prompt_template = """You are a travel insurance expert assistant. 
        Use the following policy excerpts to answer the question concisely and accurately.
        Always use exact policy terminology and cite relevant clauses when possible.

        Policy Excerpts:
        {context}

        Question: {question}
        
        Answer:"""
    
    def retrieve_relevant_documents(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant policy documents for a query.
        """
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k=k)
    
    def generate_response(self, query: str, max_length: int = 200) -> str:
        """
        Generate a response to a travel insurance query using RAG.
        """
        relevant_docs = self.retrieve_relevant_documents(query)
        context = "\n\n".join([f"Excerpt {i+1}: {doc}" for i, (doc, score) in enumerate(relevant_docs)])
        prompt = self.prompt_template.format(context=context, question=query)
        return self.generator.generate(prompt, max_length=max_length)
    
    def add_policy_documents(self, new_documents: Union[List[str], str, Path]):
        """
        Add new policy documents to the RAG system.
        
        Args:
            new_documents: Either:
                - List of policy texts
                - Path to directory
                - Path to single file
        """
        if isinstance(new_documents, (str, Path)):
            new_chunks, new_embeddings = self.embedder.process_policy_files(new_documents)
        else:
            new_chunks = self.embedder.chunk_documents(new_documents)
            new_embeddings = self.embedder.embed_documents(new_documents)
        
        # Update vector store
        self.vector_store.add_documents(new_chunks, new_embeddings)
        self.chunks.extend(new_chunks)
        self.documents.extend(new_documents if isinstance(new_documents, list) else [])