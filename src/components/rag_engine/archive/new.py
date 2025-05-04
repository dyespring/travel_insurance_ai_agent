import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Tuple
from src.config.hf_auth import authenticate_huggingface

class TravelInsuranceRAGEngine:
    def __init__(self, policy_documents: List[str]):
        """
        Initialize the RAG Engine with travel insurance policy documents.
        
        Args:
            policy_documents: List of policy document texts to be indexed
        """
        # Authenticate first
        # authenticate_huggingface()

        # Proper device initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store documents FIRST
        self.documents = policy_documents  # Initialize before building index
        
        # Initialize embedding model (as specified in the report)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index (vector database)
        self.index = None
        self._build_faiss_index(policy_documents)
        
        # Initialize fine-tuned Mistral-7B generator (with LoRA)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator_model_name = "google/gemma-2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_model_name)
        self.generator = AutoModelForCausalLM.from_pretrained(
            self.generator_model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(self.device)
        
        # Add policy-specific prompt template
        self.prompt_template = """You are a travel insurance expert assistant. 
        Use the following policy excerpts to answer the question concisely and accurately.
        Always use exact policy terminology and cite relevant clauses when possible.

        Policy Excerpts:
        {context}

        Question: {question}
        
        Answer:"""
    
    def _build_faiss_index(self, documents: List[str]):
        """Create FAISS index from policy documents"""
        # Generate embeddings for all documents
        embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        
        # Create and train FAISS index (using IVF for efficiency as mentioned in report)
        dimension = embeddings_np.shape[1]
        nlist = min(100,len(self.documents))  # Number of clusters
        quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        
        # Train the index and add vectors
        self.index.train(embeddings_np)
        self.index.add(embeddings_np)
        self.documents = documents
    
    def retrieve_relevant_documents(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most relevant policy documents for a query
        
        Args:
            query: User question about travel insurance
            k: Number of documents to retrieve
            
        Returns:
            List of (document_text, similarity_score) tuples
        """
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding_np, k)
        
        # Return documents with their similarity scores
        results = []
        for idx, score in zip(indices[0], distances[0]):
            results.append((self.documents[idx], 1 - score))  # Convert distance to similarity
        
        return results
    
    def generate_response(self, query: str, max_length: int = 200) -> str:
        """
        Generate a response to a travel insurance query using RAG
        
        Args:
            query: User question about travel insurance
            max_length: Maximum length of generated response
            
        Returns:
            Generated answer based on policy documents
        """
        # Retrieve relevant documents
        relevant_docs = self.retrieve_relevant_documents(query)
        context = "\n\n".join([f"Excerpt {i+1}: {doc}" for i, (doc, score) in enumerate(relevant_docs)])
        
        # Format prompt with context and question
        prompt = self.prompt_template.format(context=context, question=query)
        
        # Generate response (using the fine-tuned Mistral-7B as specified)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.generator.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        # Decode and clean the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove the prompt part
        
        return response

# Example usage
if __name__ == "__main__":
    # Sample policy documents (in practice, these would be full policy texts)
    policies = [
        "Standard travel insurance covers medical expenses up to $100,000 for accidents occurring during the trip.",
        "Flight delay coverage is activated after 6 hours of delay and provides $50 per 6-hour block up to $300 total.",
        "Adventure sports like skydiving require an additional premium of 15% and are not covered under basic plans.",
        "Cancellation coverage applies if the trip is cancelled due to illness with a doctor's certificate.",
        "Lost baggage coverage provides up to $1,000 reimbursement with proper documentation from the airline."
    ]
    
    # Initialize RAG engine
    rag_engine = TravelInsuranceRAGEngine(policies)
    
    # Example queries
    queries = [
        "Does my policy cover delayed flights?",
        "What's the coverage limit for medical expenses?",
        "What's the capital city for China?",
        "Are adventure sports included in basic coverage?"
    ]
    
    # Get responses
    for query in queries:
        print(f"\nQuestion: {query}")
        response = rag_engine.generate_response(query)
        print(f"Answer: {response}")