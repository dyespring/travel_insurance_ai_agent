from flask import Flask, render_template, request, jsonify
import sys
import os
from langchain_openai import ChatOpenAI


# add module path to path
sys.path.append("..")
from router import route_input

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form["message"]
    bot_response = route_input(user_input)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)

# python app.py