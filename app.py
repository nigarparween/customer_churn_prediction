import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run train_model.py first."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
categories = artifact["categories"]

app = Flask(__name__)

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "message": "Customer Churn Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send customer details as a JSON object."
        }), 400

    missing = [
        col for col in REQUIRED_COLUMNS if col not in data
    ]

    if missing:
        return jsonify({
            "error": "Required fields are missing.",
            "missing_fields": missing
        }), 400

    customer = pd.DataFrame(
        [{col: data[col] for col in REQUIRED_COLUMNS}]
    )

    # Convert numeric inputs
    for col in NUMERIC_COLUMNS:
        customer[col] = pd.to_numeric(
            customer[col], errors="coerce"
        )

    if customer[NUMERIC_COLUMNS].isnull().any().any():
        return jsonify({
            "error": "Numeric fields must contain valid numbers."
        }), 400

    # Check categorical values
    for col, valid_values in categories.items():
        if customer[col].iloc[0] not in valid_values:
            return jsonify({
                "error": f"Invalid value for {col}.",
                "allowed_values": valid_values
            }), 400

        customer[col] = pd.Categorical(
            customer[col],
            categories=valid_values
        )

    # Apply the same encoding structure as training
    encoded = pd.get_dummies(
        customer, drop_first=True, dtype=int
    )
    encoded = encoded.reindex(
        columns=feature_columns, fill_value=0
    )

    prediction = int(model.predict(encoded)[0])
    probabilities = model.predict_proba(encoded)[0]
    churn_probability = float(probabilities[1])

    return jsonify({
        "prediction": "Yes" if prediction == 1 else "No",
        "churn_probability_percent": round(
            churn_probability * 100, 2
        ),
        "message": (
            "Customer is predicted to churn."
            if prediction == 1
            else "Customer is predicted to stay."
        )
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
