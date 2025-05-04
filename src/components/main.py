from dotenv import load_dotenv
from src.components.rag_engine.archive.document_loader1112132 import PolicyDocumentLoader
from src.components.rag_engine.archive.text_splitter import PolicyTextSplitter
from src.components.rag_engine.archive.embedding_generator import PolicyEmbeddingGenerator
from src.components.rag_engine.vector_store import PolicyVectorStore
from src.components.rag_engine.archive.retriever import PolicyRetriever
from src.components.rag_engine.archive.generator_back import PolicyResponseGenerator
from src.components.rag_engine.rag_engine import RAGOrchestrator
from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline
from transformers import pipeline


def initialize_rag_system():
    # Load environment variables
    load_dotenv()

    # 1. Load documents
    loader = PolicyDocumentLoader()
    documents = loader.load_all_documents()

    # 2. Split documents
    splitter = PolicyTextSplitter()
    chunks = splitter.split_documents(documents)

    # 3. Initialize embedding model
    embedding_model = PolicyEmbeddingGenerator()

    # 4. Create and save vector store
    vector_store = PolicyVectorStore(embedding_model.embedding_model)
    vector_store.create_vector_store(chunks)
    vector_store.save_vector_store()

    # Initialize the correct pipeline with a text-generation specific model
    pipe = pipeline(
        "text-generation",
        model="google/gemma-2b-it",  # Or try "google/flan-t5-large" if you have the resources
        tokenizer="google/gemma-2b-it",
        device="cpu",
        max_length=1000,
        do_sample=True,
        temperature=0.3,
        truncation=True,
    )

    # Create LangChain interface
    llm = HuggingFacePipeline(pipeline=pipe)

    # 5. Initialize LLM
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-large",  # 3GB model
    #     huggingfacehub_api_token="hf_your_token_here",
    #     model_kwargs={"temperature": 0.2, "max_length": 1000}
    # )

    # 6. Create retriever and generator
    # Load the vector store here, after it's been saved
    vector_store.load_vector_store()

    # **DEBUG: Check vector store contents**
    print("\n--- Vector Store Contents (DEBUG) ---")
    try:
        all_docs = vector_store.get_all_documents()
        print(f"Found {len(all_docs)} documents in the vector store.")
        for doc in all_docs:
            print(f"  - {doc.page_content[:50]}...")  # Print the first 50 characters
    except Exception as e:
        print(f"Error accessing vector store documents: {e}")

    retriever = PolicyRetriever(vector_store.vector_store, llm)
    generator = PolicyResponseGenerator(llm)

    # 7. Create orchestrator
    return RAGOrchestrator(retriever, generator)


if __name__ == "__main__":
    # 1. Initialize system
    rag_system = initialize_rag_system()

    # 2. Verify vector store
    print("Vector store test:")
    test_docs = rag_system.retriever.get_relevant_documents("flight delay")
    print("Test retrieval:", test_docs[0].page_content if test_docs else "No documents found")

    # 3. Run query
    result = rag_system.query_policy("Does my policy cover delayed flights?")

    # 4. Print formatted output
    print("\nFinal Answer:")
    print("Question:", result["question"])
    print("Answer:", result["response"])
    print("\nSource Documents:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"{i}. {doc.page_content[:100]}... (from {doc.metadata.get('source', 'unknown')})")