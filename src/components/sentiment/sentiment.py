from serpapi import GoogleSearch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.nn.functional import softmax
from collections import Counter
import torch

# === STEP 1: Get provider_id and summary metadata ===
def get_data_id(query, api_key):
    params = {
        "engine": "google_maps",
        "q": query,
        "api_key": api_key
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        place = results.get("place_results", {})
        provider_id = place.get("provider_id")
        title = place.get("title")
        address = place.get("address")

        if provider_id:
            print(f"\n✅ Found business: {title}")
            print(f"📍 Address: {address}")
            print(f"🆔 provider_id: {provider_id}")
        else:
            print("❌ No provider_id found.")

        return provider_id, place
    except Exception as e:
        print(f"❌ Error during lookup: {e}")
        return None, {}

# === STEP 2: Fetch all review pages ===
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

# === STEP 3: Clean and prepare reviews ===
def extract_cleaned_descriptions(reviews):
    def clean(text):
        return text.lower().strip()

    return [clean(r["text"]) for r in reviews if "text" in r and r["text"].strip()]

# === STEP 4: Sentiment Analysis ===
def analyze_sentiment(texts, tokenizer, model):
    if not texts:
        print("❌ No valid reviews for sentiment analysis.")
        return [], []

    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    logits = model(**encoded).logits
    probs = softmax(logits, dim=1)
    labels = torch.argmax(probs, dim=1).tolist()
    
    return labels, probs.tolist()

# === STEP 5: Summarization ===
def summarize_reviews(texts, summarizer, max_chunk_length=1000):
    if not texts:
        return "No valid reviews to summarize."

    combined_text = " ".join(texts)[:max_chunk_length]
    summary = summarizer(combined_text, max_length=45, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# === MAIN ===
def main():
    SERP_API_KEY = "e236059e8ddc733ce7e2957365020e879106a4bbe69a22a5e65a4c6cc31dd726"
    BUSINESS_QUERY = "InsureandGo Insurance Sydney"
    ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    T5_MODEL = "t5-small"

    data_id, place = get_data_id(BUSINESS_QUERY, SERP_API_KEY)
    if not data_id:
        return

    reviews = fetch_all_reviews(data_id, SERP_API_KEY, max_pages=3)
    if not reviews:
        print("❌ No reviews to process.")
        return

    cleaned_texts = extract_cleaned_descriptions(reviews)
    if not cleaned_texts:
        print("❌ No cleaned review text to analyze.")
        return

    print("\n📦 Loading models...")
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
    summarizer = pipeline("summarization", model=T5_MODEL)

    print("🔍 Analyzing sentiment...")
    sentiments, _ = analyze_sentiment(cleaned_texts, roberta_tokenizer, roberta_model)

    print("🧠 Summarizing reviews...")
    summary = summarize_reviews(cleaned_texts, summarizer)

    print("\n📊 Sentiment Distribution:")
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    counts = Counter(sentiments)
    for label, count in counts.items():
        print(f"{sentiment_map.get(label, label)}: {count}")

    print("\n📝 Summary:")
    print(summary)

# === Run it ===
if __name__ == "__main__":
    main()
