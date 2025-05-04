import os
from typing import List, Optional, Union
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class PolicyDocumentProcessor:
    def __init__(
        self,
        policy_dir: str = "data/policies/raw",
        processed_dir: str = "data/policies/processed",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        use_unstructured: bool = False,
    ):
        """
        Initialize the policy document processor

        Args:
            policy_dir: Directory containing raw policy documents
            processed_dir: Directory to store processed documents and FAISS index
            embedding_model: HuggingFace model name for embeddings
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            use_unstructured: Whether to use UnstructuredFileLoader for more file types
        """
        self.policy_dir = policy_dir
        self.processed_dir = processed_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Create directories if they don't exist
        os.makedirs(self.policy_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def _get_loader(self):
        """Get appropriate loader based on configuration"""
        if self.use_unstructured:
            return DirectoryLoader(
                self.policy_dir,
                glob="**/*.*",  # All files
                loader_cls=UnstructuredFileLoader,
                show_progress=True,
                use_multithreading=True,
            )
        else:
            return DirectoryLoader(
                self.policy_dir,
                glob="**/*.*",  # All files
                loader_kwargs={".pdf": PyPDFLoader, ".txt": TextLoader},
                show_progress=True,
                loader_cls=lambda path: self._select_loader(path),
            )

    def _select_loader(self, path: str):
        """Select loader based on file extension"""
        if path.lower().endswith(".pdf"):
            return PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            return TextLoader(path)
        else:
            # logger.warning(f"No loader available for file: {path}")
            return None

    def load_policy_documents(self) -> List[Document]:
        """
        Load and split policy documents from files

        Returns:
            List of document chunks with metadata
        """
        # logger.info(f"Loading policy documents from {self.policy_dir}")

        loader = self._get_loader()
        documents = []

        try:
            documents = loader.load()
            # Filter out None results from unsupported files
            documents = [doc for doc in documents if doc is not None]
            print(f"Loaded {len(documents)} raw documents.")  # DEBUG
        except Exception as e:
            # logger.error(f"Error loading documents: {str(e)}")
            raise

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")  # DEBUG

        return chunks

    def generate_embeddings(self, documents: Optional[List[dict]] = None) -> FAISS:
        """
        Generate embeddings for policy documents and create FAISS index

        Args:
            documents: List of documents to process (if None, loads fresh)

        Returns:
            FAISS vector store with document embeddings
        """
        if documents is None:
            documents = self.load_policy_documents()

        if not documents:
            raise ValueError("No documents found to process")

        # logger.info("Generating embeddings for policy documents")

        # Create FAISS index
        vector_store = FAISS.from_documents(documents=documents, embedding=self.embedding_model)

        # Save the index
        index_path = os.path.join(self.processed_dir, "faiss_index")
        vector_store.save_local(index_path)
        # logger.info(f"Saved FAISS index to {index_path}")

        return vector_store

    def update_policy_index(self, new_document_path: str):
        """
        Update the FAISS index with a new policy document

        Args:
            new_document_path: Path to new policy document to add
        """
        # Load existing index if it exists
        index_path = os.path.join(self.processed_dir, "faiss_index")

        if os.path.exists(index_path):
            vector_store = FAISS.load_local(
                index_path, self.embedding_model, allow_dangerous_deserialization=True
            )
        else:
            vector_store = None

        # Process new document
        if self.use_unstructured:
            loader = UnstructuredFileLoader(new_document_path)
        else:
            if new_document_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(new_document_path)
            elif new_document_path.lower().endswith(".txt"):
                loader = TextLoader(new_document_path)
            else:
                # logger.warning(f"Unsupported file type: {new_document_path}")
                return

        try:
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)

            # Add to existing index or create new
            if vector_store:
                vector_store.add_documents(chunks)
                # logger.info(f"Added {len(chunks)} chunks to existing index")
            else:
                vector_store = FAISS.from_documents(
                    documents=chunks, embedding=self.embedding_model
                )
                # logger.info("Created new index with document")

            # Save updated index
            vector_store.save_local(index_path)
            # logger.info(f"Updated FAISS index saved to {index_path}")
        except Exception as e:
            # logger.error(f"Error processing new document: {str(e)}")
            raise

    # def get_supported_extensions(self) -> List[str]:
    #     """Return list of supported file extensions"""
    #     if self.use_unstructured:
    #         return ["All files supported by Unstructured"]
    #     return [".pdf", ".txt"]