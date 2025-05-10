from serpapi import GoogleSearch
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from torch.nn.functional import softmax
from collections import Counter

# === STEP 1: Get data_id and reviews from Google Maps ===
def get_data_id(query, api_key):
    params = {
        "engine": "google_maps",
        "q": query,
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()

        # Extract business info from place_results
        place = results.get("place_results", {})
        data_id = place.get("data_id")
        title = place.get("title")
        address = place.get("address")

        # Display business info
        if data_id:
            print(f"\n✅ Found business: {title}")
            print(f"📍 Address: {address}")
            print(f"🆔 data_id: {data_id}")
        else:
            print("❌ No provider_id found.")

        # Extract most relevant reviews
        reviews = place.get("user_reviews", {}).get("most_relevant", [])
        if not reviews:
            print("❌ No user reviews found.")
        else:
            print(f"\n🗒️ {len(reviews)} most relevant reviews found:")

        return data_id, reviews

    except Exception as e:
        print(f"❌ Error during lookup: {e}")
        return None, []
def fetch_all_reviews(data_id, api_key, max_pages=3):
    all_reviews = []
    page_token = None

    for _ in range(max_pages):
        params = {
            "engine": "google_maps_reviews",
            "api_key": api_key,
            "data_id": data_id
        }
        if page_token:
            params["next_page_token"] = page_token

        search = GoogleSearch(params)
        results = search.get_dict()

        reviews = results.get("reviews", [])
        if not reviews:
            break

        all_reviews.extend(reviews)
        page_token = results.get("serpapi_pagination", {}).get("next_page_token")
        if not page_token:
            break

    print(f"🗒️ Total reviews fetched: {len(all_reviews)}")
    return all_reviews
# === STEP 2: Print formatted reviews ===
def print_reviews(reviews):
    for i, r in enumerate(reviews, 1):
        print(f"\n--- Review #{i} ---")
        print(f"⭐ Rating: {r.get('rating')}")
        print(f"👤 User: {r.get('username')}")
        print(f"🗓️ Date: {r.get('date')}")
        print(f"📝 Review: {r.get('description')}")
# Add this function before main
def extract_cleaned_descriptions(reviews):
    def clean(text):
        return text.lower().strip()

    return [clean(r["description"]) for r in reviews if "description" in r and r["description"].strip()]

# Step 3: Sentiment Analysis
def analyze_sentiment(texts, tokenizer, model):
    if not texts:
        print("❌ No valid reviews for sentiment analysis.")
        return [], []

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    logits = model(**encoded).logits
    probs = softmax(logits, dim=1)
    labels = torch.argmax(probs, dim=1).tolist()
    
    return labels, probs.tolist()
# Step 4: Summarize Reviews
def summarize_reviews(texts, summarizer, max_chunk_length=1000):
    if not texts:
        return "No valid reviews to summarize."

    combined_text = " ".join(texts)[:max_chunk_length]
    summary = summarizer(combined_text, max_length=45, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# === MAIN ===
def main():
    SERP_API_KEY = "e236059e8ddc733ce7e2957365020e879106a4bbe69a22a5e65a4c6cc31dd726"
    BUSINESS_QUERY = "Tick Travel Insurance"
    ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    T5_MODEL = "t5-small"

    data_id, reviews = get_data_id(BUSINESS_QUERY, SERP_API_KEY)
    all_reviews = fetch_all_reviews(data_id,SERP_API_KEY, 3)

    if not (data_id and reviews):
        print("\n❌ No data_id or reviews to show.")
        return

    print_reviews(all_reviews)
    # cleaned_texts = extract_cleaned_descriptions(all_reviews)

    # if not cleaned_texts:
    #     print("❌ No valid cleaned review texts to analyze.")
    #     return

    # print("\n🧹 Cleaned review texts for RoBERTa:")
    # for i, text in enumerate(cleaned_texts, 1):
    #     print(f"{i}. {text}")

    # print("\n📦 Loading models...")
    # roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    # roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
    # summarizer = pipeline("summarization", model=T5_MODEL)

    # print("🔍 Analyzing sentiment...")
    # sentiments, _ = analyze_sentiment(cleaned_texts, roberta_tokenizer, roberta_model)

    # print("🧠 Summarizing reviews...")
    # summary = summarize_reviews(cleaned_texts, summarizer)

    # print("\n📊 Sentiment Distribution:")
    # sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    # counts = Counter(sentiments)
    # for label, count in counts.items():
    #     print(f"{sentiment_map.get(label, label)}: {count}")

    # print("\n📝 Summary:")
    # print(summary)

# === Run it ===
if __name__ == "__main__":
    main()
    