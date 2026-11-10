# Customer Churn Prediction API

A machine learning application that predicts customer churn
using Logistic Regression and FastAPI.

## Project Features

- Machine Learning model using Scikit-learn
- Numerical and categorical preprocessing
- FastAPI REST API
- Pydantic input validation
- Churn prediction and probability
- Interactive API documentation

## Project Structure

customer-churn-prediction-api/
│
├── main.py
├── churn_pipeline.joblib
├── requirements.txt
└── README.md

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Joblib

## Installation

Clone the repository:

    git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-api.git

Install dependencies:

    pip install -r requirements.txt

## Run the API

    uvicorn main:app --reload

Open the API documentation:

    http://127.0.0.1:8000/docs

## Prediction Endpoint

POST /predict

Example input:

    {
        "tenure": 3,
        "monthly_charges": 85.5,
        "support_calls": 6,
        "contract_type": "Month-to-month"
    }

The API returns a churn prediction and churn probability.

## Model

The saved Joblib artifact contains the trained
Scikit-learn preprocessing and classification pipeline.

## Disclaimer

This project uses synthetic customer data for educational purposes.
The model is a demonstration and is not intended for real business decisions.
