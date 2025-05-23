import os
import json
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify
from src.components.rec_system.recommend import recommend
from src.components.rec_system.preprocessing import getDestination

# from chat import get_response

# Load trained model
# with open("chatbot_model.pkl", "rb") as f:
#     model = pickle.load(f)
# print(model)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


app = Flask(__name__)
load_dotenv()

@app.route("/")
def home():
    options = getDestination()
    return render_template("index.html", options=options)

@app.route("/predict", methods=["POST"])
def predict():
    destination = request.form.get("destination")
    duration = request.form.get("duration")
    gender = request.form.get("gender")
    age = request.form.get("age")
   
    result = recommend(gender, destination, duration, age)

    # now get reviews
   
    with open("reviews.json", "r") as f:
        review_data = json.load(f)

    review_lookup = {entry["company_name"]: entry for entry in review_data}
    for r in result:
        agency = r.get("Agency")
        review = review_lookup.get(agency)
        if review:
            r.update({
                "average_rating": review["average_rating"],
                "positive": review["positive"],
                "neutral": review["neutral"],
                "negative": review["negative"],
                "summary": review["summary"]
            })
        else:
            # fallback if review not found
            r.update({
                "average_rating": "N/A",
                "positive": 0,
                "neutral": 0,
                "negative": 0,
                "summary": "No reviews available."
            })
    
    return render_template('predict.html', result=result) 


# @app.route("/chat", methods=["POST"])
# def chat():
#     text = request.get_json().get("message");
#     response = get_response(text);
#     return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(port=8000, debug=True)
