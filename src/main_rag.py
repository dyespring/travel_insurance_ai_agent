from components.rag_engine.rag_engine import TravelInsuranceRAGEngine

def main():
 
    # Initialize RAG engine
    rag_engine = TravelInsuranceRAGEngine()
    
    # Example queries
    queries = [
        # "Does my policy cover delayed flights?",
        "What's the coverage limit for medical expenses?",
        "Are adventure sports included in basic coverage?"
    ]
    
    # Get responses
    for query in queries:
        print(f"\nQuestion: {query}")
        response = rag_engine.generate_response(query)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()