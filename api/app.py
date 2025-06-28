from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all pickle files
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

# Initialize FastAPI app
app = FastAPI(title="Customer Churn Prediction API")

# Define input schema
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str      
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "churn prediction API is running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])

    y_proba = model.predict_proba(input_df)[0][1]
    y_pred = int(y_proba > 0.55)

    return {
        "churn prediction": y_pred,
        "churn probability": round(float(y_proba), 3)
    }