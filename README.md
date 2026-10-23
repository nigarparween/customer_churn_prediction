
# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave a service.

## Features
- Data cleaning and preprocessing
- Categorical feature encoding
- Random Forest classification model
- Model evaluation using accuracy, precision, recall, and F1-score
- Flask REST API for churn predictions
- Churn probability returned as JSON

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib

## Model Performance
- Accuracy: 76.83% on the test dataset

## Project Structure
```text
customer_churn_prediction/
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   └── churn_model.pkl
├── train_model.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

1. Install the dependencies:
   `pip install -r requirements.txt`

2. Train and save the model:
   `python train_model.py`

3. Start the API:
   `python app.py`

4. Health check:
   `http://127.0.0.1:5000/health`

5. Send customer details as JSON to the `/predict` endpoint using a POST request.

## Dataset
Telco Customer Churn dataset. Data cleaning and preprocessing are performed before training.

## Disclaimer
This project is for educational purposes. Predictions are estimates and are not guarantees of customer behaviour.
