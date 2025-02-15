from flask import Flask, jsonify
from query import fetch_milk_production

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Milk Production API!"

@app.route("/milk_production", methods=["GET"])
def get_milk_production():
    data = fetch_milk_production()
    return jsonify(data)  # Convert result to JSON response

if __name__ == "__main__":
    app.run(debug=True)