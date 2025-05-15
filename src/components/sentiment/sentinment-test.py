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
def print_reviews_and_average(reviews):
    ratings = []

    for i, r in enumerate(reviews, 1):
        rating = r.get('rating')
        if isinstance(rating, (int, float)):
            ratings.append(rating)

        print(f"\n--- Review #{i} ---")
        print(f"⭐ Rating: {rating}")
        print(f"👤 User: {r.get('username') or r.get('user', {}).get('name', 'Unknown')}")
        print(f"🗓️ Date: {r.get('date')}")
        print(f"📝 Review: {r.get('extracted_snippet', {}).get('original') or r.get('text')}")

    if ratings:
        average = round(sum(ratings) / len(ratings), 2)
        print(f"\n📊 Average Rating: {average} ⭐ (based on {len(ratings)} reviews)")
    else:
        print("\n⚠️ No valid ratings to calculate average.")

# Add this function before main
def extract_cleaned_descriptions(reviews):
    def clean(text):
        return text.lower().strip()

    descriptions = []
    for r in reviews:
        # Try to get the full original review text
        text = r.get("extracted_snippet", {}).get("original") or r.get("text")
        if text and text.strip():
            descriptions.append(clean(text))
    return descriptions


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

def summarize_reviews_chunked(texts, summarizer, chunk_char_limit=800, summary_max_len=80):
    chunks = []
    current_chunk = ""

    for text in texts:
        if len(current_chunk) + len(text) <= chunk_char_limit:
            current_chunk += text + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = text + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"📦 Splitting into {len(chunks)} chunks for summarization...")

    summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"📝 Summarizing chunk #{i}...")
        result = summarizer(
            chunk,
            max_length=summary_max_len,
            min_length=30,
            do_sample=False
        )
        summaries.append(result[0]["summary_text"])

    # Optionally, summarize the summaries:
    final_input = " ".join(summaries)
    final_summary = summarizer(
        final_input,
        max_length=100,
        min_length=40,
        do_sample=False
    )[0]["summary_text"]

    return final_summary

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

    print_reviews_and_average(all_reviews)
    cleaned_texts = extract_cleaned_descriptions(all_reviews)

    if not cleaned_texts:
        print("❌ No valid cleaned review texts to analyze.")
        return

    print("\n🧹 Cleaned review texts for RoBERTa:")
    for i, text in enumerate(cleaned_texts, 1):
        print(f"{i}. {text}")

    print("\n📦 Loading models...")
    roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
    summarizer = pipeline("summarization", model=T5_MODEL)

    print("🔍 Analyzing sentiment...")
    sentiments, _ = analyze_sentiment(cleaned_texts, roberta_tokenizer, roberta_model)

    print("🧠 Summarizing reviews...")
    summary = summarize_reviews_chunked(cleaned_texts, summarizer)

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
    