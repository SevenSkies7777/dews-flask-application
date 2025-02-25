from flask import Flask, jsonify

from OutliersDetectionService import fetch_outliers

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Milk Production API!"

@app.route("/service-api/v1/outliers-detection/process-outliers", methods=["GET"])
def fetch_responses_outliers():
    data = fetch_outliers()
    return jsonify(data)  # Convert result to JSON response

if __name__ == "__main__":
    app.run(debug=True)