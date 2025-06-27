from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all pickle files
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

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

    input_df_encoded = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    input_df_encoded = input_df_encoded[feature_names]

    input_scaled = scaler.transform(input_df_encoded)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    return {
        "churn prediction": int(prediction),
        "churn probability": round(float(probability), 3)
    }