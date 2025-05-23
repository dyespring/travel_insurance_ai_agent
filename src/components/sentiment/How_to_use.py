import json
# other_module.py
from sentiment import sentiment

# instantiate once (loads models into memory)
analyzer = sentiment()

def do_reviews(company_names: [str]):
    results_list = []
    for company_name in company_names:
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

        # Prepare structured result
        entry = {
            "company_name": company_name,
            "average_rating": f"{avg_rating} ⭐",
            "positive": counts.get("Positive", 0),
            "neutral": counts.get("Neutral", 0),
            "negative": counts.get("Negative", 0),
            "summary": summary
        }

        results_list.append(entry)
    
    # Save all results to a JSON file
    # with open("reviews-8.json", "w", encoding="utf-8") as f:
    #     json.dump(results_list, f, indent=4)

# example call
if __name__ == "__main__":
    do_reviews(["Statue of Liberty"])
