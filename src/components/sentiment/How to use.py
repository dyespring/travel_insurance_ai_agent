# other_module.py
from sentiment import sentiment

# instantiate once (loads models into memory)
analyzer = sentiment()

def do_reviews(company_name: str):
    try:
        result = analyzer.analyze(company_name)
    except ValueError as e:
        print("⚠️", e)
        return

    # unpack
    avg_rating = result["average_rating"]
    counts     = result["sentiment_counts"]
    summary    = result["summary"]

    # now you can log it, return it from an API, save to DB, etc.
    print(f"{company_name}: {avg_rating} ⭐ — {counts}")
    print("Summary:", summary)

# example call
if __name__ == "__main__":
    do_reviews("CBH Group Metro Grain Centre")
