"""
Customer Churn Prediction — REST API
======================================
A lightweight FastAPI wrapper around the serialised sklearn pipeline.
Loads the trained model from disk and exposes a POST /predict endpoint
that accepts a single customer's data and returns the churn prediction.

Usage:
    uvicorn api:app --reload

Docs:
    http://127.0.0.1:8000/docs   (Swagger UI)
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums for constrained fields
# ──────────────────────────────────────────────
class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Gender(str, Enum):
    male = "Male"
    female = "Female"


class InternetServiceType(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class ContractType(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodType(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


# ──────────────────────────────────────────────
# Pydantic request schema
# ──────────────────────────────────────────────
class CustomerData(BaseModel):
    """
    Schema representing a single telecom customer.
    Field names and allowed values match the original CSV exactly.
    """

    gender: Gender = Field(..., example="Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: YesNo = Field(..., example="Yes")
    Dependents: YesNo = Field(..., example="No")
    tenure: int = Field(..., ge=0, example=12)
    PhoneService: YesNo = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: InternetServiceType = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: ContractType = Field(..., example="Month-to-month")
    PaperlessBilling: YesNo = Field(..., example="Yes")
    PaymentMethod: PaymentMethodType = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., ge=0, example=70.35)
    TotalCharges: float = Field(..., ge=0, example=840.50)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 840.50,
            }
        }


# ──────────────────────────────────────────────
# Pydantic response schema
# ──────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """JSON response returned by the /predict endpoint."""

    churn_prediction: int = Field(..., description="1 = will churn, 0 = will not churn")
    churn_probability: float = Field(..., description="Probability of churn (0.0 to 1.0)")
    risk_level: str = Field(..., description="Human-readable risk label")


# ──────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join("models", "logistic_regression_pipeline.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict whether a telecom customer will churn based on their "
        "account attributes. Powered by a scikit-learn Logistic Regression "
        "pipeline trained on the Telco Customer Churn dataset."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
pipeline = None


@app.on_event("startup")
def load_model():
    """Load the serialised sklearn pipeline into memory at server start."""
    global pipeline
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run `python main.py` first to train and save the model."
        )
    pipeline = joblib.load(MODEL_PATH)
    print(f"[API] Model loaded from {MODEL_PATH}")


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    """Root health-check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "docs": "/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerData):
    """
    Accept a single customer's data and return the churn prediction.

    The incoming JSON is converted to a DataFrame row, pushed through
    the pre-trained sklearn Pipeline (imputation + encoding + scaling +
    classifier), and the result is returned as structured JSON.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Convert Pydantic model → dict → single-row DataFrame
    row = customer.model_dump()

    # Convert enum values back to their string representation
    for key, value in row.items():
        if isinstance(value, str):
            pass  # already a string from enum .value
        elif hasattr(value, "value"):
            row[key] = value.value

    df = pd.DataFrame([row])

    # Predict
    prediction = int(pipeline.predict(df)[0])
    probability = float(pipeline.predict_proba(df)[0][1])

    # Human-readable risk label
    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return PredictionResponse(
        churn_prediction=prediction,
        churn_probability=round(probability, 4),
        risk_level=risk,
    )
