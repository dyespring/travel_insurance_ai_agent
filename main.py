import os
import json
from dotenv import load_dotenv

from flask import Flask, render_template, request, jsonify
# from src.travel_insurance_agent.tools.vector_search_tools import WeaviateVectorSearchTool

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
    return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     load_dotenv()
#     tool = WeaviateVectorSearchTool(
#         weaviate_cluster_url=os.getenv("WEAVIATE_URL"),
#         weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
#         collection_name="insurance_data_system_2",
#     )
#     destination = request.form.get("destination")
#     duration = request.form.get("duration")
#     gender = request.form.get("gender")
#     age = request.form.get("age")
#     result = tool.run("Find me similar insurance package with destination: " + destination +
#     ", Duration: " + duration + ", gender: " + gender + ", age: " + age)
#     return render_template('predict.html', result=result) 


# @app.route("/chat", methods=["POST"])
# def chat():
#     text = request.get_json().get("message");
#     response = get_response(text);
#     return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
