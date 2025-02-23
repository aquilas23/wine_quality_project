from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        data = request.get_json()

        # Extract features from request JSON (instead of looking for a "features" key)
        features = np.array([[
            data["fixed acidity"], data["volatile acidity"], data["citric acid"],
            data["residual sugar"], data["chlorides"], data["free sulfur dioxide"],
            data["total sulfur dioxide"], data["density"], data["pH"],
            data["sulfates"], data["alcohol"]
        ]])

        # Make a prediction
        prediction = model.predict(features)[0]

        # Return the prediction result
        return jsonify({"wine_quality": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
