from response_table_redis import RedisResponseTable

if __name__ == "__main__":
    table = RedisResponseTable()

    sample_data = {
        "query": "Does travel insurance cover cancelled flights?",
        "rag_aw": "Yes, if cancellation is due to illness and supported by documentation.",
        "gpt_aw": "Yes, provided you submit a doctor’s certificate.",
        "sentiment_aw": "positive",
        "summary_aw": "Coverage is available for illness-related cancellations."
    }

    print("\n=== Inserting sample response ===")
    redis_key = table.insert_response(sample_data)

    print("\n=== Retrieving stored response ===")
    stored_data = table.get_response(redis_key)
    for k, v in stored_data.items():
        print(f"{k}: {v}")

    print("\n=== Listing all Redis response keys ===")
    keys = table.list_all_responses()
    for k in keys:
        print(k)
