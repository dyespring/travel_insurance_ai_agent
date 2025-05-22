from components.rag_engine.rag_engine import TravelInsuranceRAGEngine
from components.sentiment.sentiment import analyze_sentiment, summarize_reviews
from database.response_table_redis import RedisResponseTable
from database.db_connector_sql_1 import insert_query_sql  
from components.rec_system import recommend_policy  

class InsuranceAgentOrchestrator:
    def __init__(self):
        self.rag_engine = TravelInsuranceRAGEngine()
        self.redis = RedisResponseTable()

    def handle_rag_query(self, question: str) -> str:
        answer = self.rag_engine.generate_response(question)
        return answer

    def handle_sentiment(self, reviews: list[str]) -> dict:
        labels, _ = analyze_sentiment(reviews)
        summary = summarize_reviews(reviews)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        distribution = {label_map[l]: labels.count(l) for l in set(labels)}
        return {
            "distribution": distribution,
            "summary": summary
        }

    def handle_recommendation(self, user_input: dict) -> str:
        return f"Recommended plan for age {user_input['age']}, visiting {user_input['destination']} for {user_input['duration']} weeks."

    def store_all_to_redis(self, query: str, rag: str, sentiment: dict, recommendation: str):
        data = {
            "query": query,
            "rag_aw": rag,
            "gpt_aw": rag,  # Same as rag for now
            "sentiment_aw": sentiment["distribution"],
            "summary_aw": sentiment["summary"],
            "recommendation": recommendation
        }
        uid = self.redis.insert_response(data)
        print(f"[Redis] Stored under key: response:{uid}")


# === CLI TEST ===
if __name__ == "__main__":
    orchestrator = InsuranceAgentOrchestrator()

    print("\n=== RAG Query ===")
    query = "Does my insurance cover delayed flights?"
    rag_result = orchestrator.handle_rag_query(query)
    print("RAG Answer:", rag_result)

    print("\n=== Sentiment Analysis ===")
    dummy_reviews = ["Great support", "Terrible refund process", "Reasonable coverage"]
    sentiment_result = orchestrator.handle_sentiment(dummy_reviews)
    print("Sentiment Result:", sentiment_result)

    print("\n=== Recommendation ===")
    form = {"age": 30, "gender": "F", "duration": 3, "destination": "Japan"}
    recommendation = orchestrator.handle_recommendation(form)
    print("Recommendation:", recommendation)

    print("\n=== Store All to Redis ===")
    orchestrator.store_all_to_redis(query, rag_result, sentiment_result, recommendation)
