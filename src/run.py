from components.orchestrator import InsuranceAgentOrchestrator

# pip install numpy torch transformers sentence-transformers langchain faiss-cpu serpapi flask
# pip install pdfplumber

if __name__ == "__main__":
    agent = InsuranceAgentOrchestrator()

    print("\n=== RAG Query Test ===")
    response = agent.handle_rag_query("Does the policy cover lost baggage?")
    print("RAG Response:", response)

    print("\n=== Recommendation Test ===")
    sample_form = {
        "age": 35,
        "gender": "F",
        "duration": 2,
        "destination": "Italy"
    }
    print("Recommendation:", agent.handle_recommendation(sample_form))

    print("\n=== Sentiment Test ===")
    reviews = [
        "Great insurance coverage and support!",
        "The claim process was terrible and slow.",
        "It was okay, nothing special."
    ]
    sentiment = agent.handle_sentiment(reviews)
    print("Sentiment Result:", sentiment)

    print("\n=== Logging Test ===")
    agent.log_all("Does the policy cover lost baggage?", response, sentiment, "Test Recommendation")
