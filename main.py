
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# Load the trained pipeline
model = joblib.load("churn_pipeline.joblib")

# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn using Machine Learning",
    version="1.0"
)

# Define input validation
class CustomerData(BaseModel):
    tenure: int = Field(ge=1, le=100)
    monthly_charges: float = Field(gt=0)
    support_calls: int = Field(ge=0)
    contract_type: str

# Health-check endpoint
@app.get("/")
def home():
    return {
        "message": "Customer Churn API is running"
    }

# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    input_data = pd.DataFrame([{
        "tenure": customer.tenure,
        "monthly_charges": customer.monthly_charges,
        "support_calls": customer.support_calls,
        "contract_type": customer.contract_type
    }])

    prediction = int(model.predict(input_data)[0])

    probabilities = model.predict_proba(input_data)[0]

    churn_index = list(model.classes_).index(1)

    churn_probability = float(probabilities[churn_index])

    return {
        "prediction": (
            "Churn" if prediction == 1 else "No Churn"
        ),
        "churn_probability": round(churn_probability, 4)
    }
