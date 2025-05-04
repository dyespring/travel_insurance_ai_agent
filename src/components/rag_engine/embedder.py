import os
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the document embedder with text splitting capabilities.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.supported_extensions = ('.pdf', '.txt', '.md')

    def load_documents(self, file_paths: Union[str, List[str], Path, List[Path]]) -> List[str]:
        """
        Load and extract text from documents (supports PDFs and text files).
        
        Args:
            file_paths: Single path or list of paths to documents
            
        Returns:
            List of extracted texts
            
        Raises:
            FileNotFoundError: If no supported files are found
        """
        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]
        
        documents = []
        for path in file_paths:
            path = Path(path)
            if not path.exists():
                continue
                
            try:
                if path.suffix.lower() == '.pdf':
                    with pdfplumber.open(path) as pdf:
                        text = "\n".join([page.extract_text() for page in pdf.pages])
                        documents.append(text)
                elif path.suffix.lower() in ('.txt', '.md'):
                    with open(path, 'r', encoding='utf-8') as f:
                        documents.append(f.read())
            except Exception as e:
                print(f"Error processing {path.name}: {str(e)}")
                continue
                
        if not documents:
            raise FileNotFoundError("No supported documents could be loaded")
        return documents

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """
        Split documents into semantically meaningful chunks.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of text chunks
        """
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))
        return chunks

    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of documents after automatic chunking.
        
        Args:
            documents: List of document texts
            
        Returns:
            Numpy array of document embeddings (shape: [num_chunks, embedding_dim])
        """
        chunks = self.chunk_documents(documents)
        return self.model.encode(chunks, convert_to_tensor=False)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: Input query/text to embed
            
        Returns:
            Numpy array of query embedding (shape: [1, embedding_dim])
        """
        return self.model.encode(query, convert_to_tensor=False).reshape(1, -1)


    def find_policy_files(self, base_dir: Union[str, Path] = "components/rag_engine") -> List[Path]: # add the file path
        """
        Find all supported policy files in the specified directory.
        
        Args:
            base_dir: Directory to search for policy files
            
        Returns:
            List of Path objects for found policy files
        """
        policy_dir = Path(base_dir)
        policy_dir.mkdir(parents=True, exist_ok=True)
        
        policy_files = []
        for ext in self.supported_extensions:
            policy_files.extend(list(policy_dir.glob(f"*{ext}")))
            
        return policy_files

    def process_policy_files(self, base_dir: Union[str, Path] = "") -> Tuple[List[str], np.ndarray]:
        """
        Complete pipeline: find, load, chunk and embed policy documents.
        
        Args:
            base_dir: Directory containing policy files
            
        Returns:
            Tuple of (chunks, embeddings)
            
        Raises:
            FileNotFoundError: If no supported files are found
        """
        policy_files = self.find_policy_files(base_dir)
        if not policy_files:
            raise FileNotFoundError(
                f"No supported policy files found in {base_dir}\n"
                f"Supported extensions: {', '.join(self.supported_extensions)}"
            )
            
        print(f"Processing {len(policy_files)} policy files:")
        for i, file in enumerate(policy_files, 1):
            print(f"{i}. {file.name}")
            
        documents = self.load_documents(policy_files)
        chunks = self.chunk_documents(documents)
        embeddings = self.embed_documents(chunks)
        
        print("\nProcessing results:")
        print(f"- Total chunks generated: {len(chunks)}")
        print(f"- Embeddings shape: {embeddings.shape}")
        print(f"- Average chunk length: {sum(len(c) for c in chunks)/len(chunks):.0f} characters")
        
        return chunks, embeddings