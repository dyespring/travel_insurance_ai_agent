# saved response for quickly reference
import os
import uuid
import redis
from dotenv import load_dotenv

load_dotenv()

class RedisResponseTable:
    def __init__(self):
        self.client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=0
        )
        self.index_key = "response:index"
        self._ensure_index()

    def _ensure_index(self):
        if not self.client.exists(self.index_key):
            self.client.sadd(self.index_key, "__init__")
            self.client.srem(self.index_key, "__init__")
            print(" Redis table 'response' initialized.")

    def insert_response(self, data: dict):
        record_id = str(uuid.uuid4())
        redis_key = f"response:{record_id}"
        self.client.hset(redis_key, mapping=data)
        self.client.sadd(self.index_key, redis_key)
        print(f" Inserted: {redis_key}")
        return redis_key

    def get_response(self, redis_key: str):
        raw = self.client.hgetall(redis_key)
        return {k.decode(): v.decode() for k, v in raw.items()}

    def list_all_responses(self):
        return [key.decode() for key in self.client.smembers(self.index_key)]

# if __name__ == "__main__":
#     table = RedisResponseTable()

#     sample_data = {
#         "query": "What travel insurance covers skiing?",
#         "rag_aw": "Winter sports are covered under Plan B.",
#         "gpt_aw": "You should look for policies with skiing add-ons.",
#         "sentiment_aw": "positive",
#         "summary_aw": "Plan B offers full coverage for ski-related travel risks."
#     }

#     key = table.insert_response(sample_data)
#     print("\n Retrieved:")
#     print(table.get_response(key))

#     print("\n All keys:")
#     print(table.list_all_responses())