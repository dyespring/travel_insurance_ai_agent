from serpapi import GoogleSearch
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from torch.nn.functional import softmax
from collections import Counter

class sentiment:
    # ⚙️ Embed your API key here once:
    SERP_API_KEY = "e236059e8ddc733ce7e2957365020e879106a4bbe69a22a5e65a4c6cc31dd726"

    def __init__(
        self,
        sentiment_model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        summarizer_model_name: str = "t5-small",
        max_pages: int = 3
    ):
        """
        sentiment_model_name: HuggingFace model for sentiment
        summarizer_model_name: HuggingFace model for summarization
        max_pages: how many pages of reviews to fetch
        """
        self.max_pages = max_pages

        # load sentiment model/tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

        # load summarizer pipeline once
        self.summarizer = pipeline("summarization", model=summarizer_model_name)

    def _get_data_id(self, query: str):
        params = {
            "engine": "google_maps",
            "q": query,
            "api_key": self.SERP_API_KEY
        }
        results = GoogleSearch(params).get_dict()
        return results.get("place_results", {}).get("data_id")

    def _fetch_reviews(self, data_id: str):
        all_reviews = []
        token = None
        for _ in range(self.max_pages):
            params = {
                "engine": "google_maps_reviews",
                "api_key": self.SERP_API_KEY,
                "data_id": data_id
            }
            if token:
                params["next_page_token"] = token

            res = GoogleSearch(params).get_dict()
            reviews = res.get("reviews", [])
            if not reviews:
                break

            all_reviews.extend(reviews)
            token = res.get("serpapi_pagination", {}).get("next_page_token")
            if not token:
                break

        return all_reviews

    @staticmethod
    def _clean_texts(reviews):
        cleaned = []
        for r in reviews:
            text = r.get("extracted_snippet", {}).get("original") or r.get("text", "")
            if text.strip():
                cleaned.append(text.lower().strip())
        return cleaned

    def _analyze_sentiment(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        logits = self.sentiment_model(**enc).logits
        probs = softmax(logits, dim=1)
        return torch.argmax(probs, dim=1).tolist()

    def _summarize(self, texts):
        # build chunks of ≤800 chars
        chunks, curr = [], ""
        for t in texts:
            if len(curr) + len(t) <= 800:
                curr += t + " "
            else:
                chunks.append(curr.strip()); curr = t + " "
        if curr: chunks.append(curr.strip())

        # summarise each chunk
        partials = [self.summarizer(c, max_length=80, min_length=30, do_sample=False)[0]["summary_text"]
                    for c in chunks]
        # final summary
        combined = " ".join(partials)
        return self.summarizer(combined, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]

    def analyze(self, company_name: str):
        """
        Returns:
          average_rating: float
          summary: str
          sentiment_counts: dict
        """
        data_id = self._get_data_id(company_name)
        if not data_id:
            raise ValueError(f"No data_id found for '{company_name}'")

        reviews = self._fetch_reviews(data_id)
        if not reviews:
            raise ValueError(f"No reviews found for '{company_name}'")

        # compute average rating
        ratings = [r["rating"] for r in reviews if isinstance(r.get("rating"), (int, float))]
        avg = round(sum(ratings) / len(ratings), 2) if ratings else 0.0

        # sentiment analysis
        texts = self._clean_texts(reviews)
        labels = self._analyze_sentiment(texts)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        counts = Counter(label_map[l] for l in labels)

        # summarization
        summary = self._summarize(texts)

        return {
            "average_rating": avg,
            "summary": summary,
            "sentiment_counts": {
                "Positive": counts.get("Positive", 0),
                "Neutral": counts.get("Neutral", 0),
                "Negative": counts.get("Negative", 0)
            }
        }
