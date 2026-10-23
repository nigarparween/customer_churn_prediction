
# Customer Churn Prediction

A Machine Learning project that predicts whether a telecom customer is likely to leave a service.

## Live API

**API URL:** https://customer-churn-prediction-pkre.onrender.com

Health check: https://customer-churn-prediction-pkre.onrender.com/health

## Features

- Customer churn prediction using Machine Learning
- Data cleaning and preprocessing with Pandas
- Feature encoding for categorical data
- Flask REST API
- Input validation and JSON responses
- Deployed online using Render

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Flask
- Joblib
- Gunicorn
- Render

## API Endpoints

### 1. Health Check

**Method:** GET

**Endpoint:** `/health`

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy",
  "message": "Customer Churn Prediction API is running"
}
```

### 2. Predict Customer Churn

**Method:** POST

**Endpoint:** `/predict`

Send customer details as a JSON object. The API returns a prediction (`Yes` or `No`) and an estimated churn probability.

Required fields:

- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

Example response:

```json
{
  "prediction": "No",
  "churn_probability_percent": 27.0,
  "message": "Customer is predicted to stay."
}
```

*This is an example result from one test request. Actual predictions depend on the input data.*

## Run Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure the trained model exists at `models/churn_model.pkl`.
5. Start the API:

```bash
python app.py
```

The API will run locally at `http://127.0.0.1:5000`.

## Project Structure

```text
customer_churn_prediction/
├── app.py
├── train_model.py
├── requirements.txt
├── data/
│   └── Telco-Customer-Churn.csv
└── models/
    └── churn_model.pkl
```

## Notes

The live API is hosted on Render's free instance. It may take around 50 seconds or more to respond after a period of inactivity.

This project was built for learning and portfolio demonstration.

