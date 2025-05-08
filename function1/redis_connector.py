import redis
'pip install redis'

redis_client = redis.Redis(host='localhost', port=6379, db=0)
redis_client.set('test_key', 'Hello Redis!')
print(redis_client.get('test_key').decode())
import os
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

if __name__ == "__main__":
    r = get_redis_client()
    r.set("test", "hello world")
    print("Test value:", r.get("test").decode())
