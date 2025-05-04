from rag_engine import TravelInsuranceRAGEngine

def main():
    # Sample policy documents
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
        "Are adventure sports included in basic coverage?"
    ]
    
    # Get responses
    for query in queries:
        print(f"\nQuestion: {query}")
        response = rag_engine.generate_response(query)
        print(f"Answer: {response}")

if __name__ == "__main__":
    main()