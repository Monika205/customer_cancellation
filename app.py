
import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer is likely to churn.",
    version="1.0.0"
)

# Load the trained model from the repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "customer_churn_model.joblib")

model = joblib.load(MODEL_PATH)


# Input fields received by the API
class CustomerData(BaseModel):
    Territory: str
    DeviceType: str
    Promotion: str
    PlanType: str

    BaseFee: float = Field(ge=0)
    AdditionalFeatureSpend: float = Field(ge=0)
    NumberOfPenalties: int = Field(ge=0)
    CurrentPeriodUsage: float = Field(ge=0)
    PriorPeriodUsage: float = Field(ge=0)
    ServiceRating: float = Field(ge=0)
    ServiceTickets: int = Field(ge=0)
    CustomerTenure: int = Field(ge=0)

    UsageDifference: float
    AdditionalSpendRatio: float = Field(ge=0)
    TotalSpend: float = Field(ge=0)
    StartMonthNumber: int = Field(ge=1, le=12)
    StartWeekNumber: int = Field(ge=1, le=53)


@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Step 1: Convert input into a dictionary
    data = customer.model_dump()

    # Step 2: Create the engineered features
    data["PenaltyLevel"] = (
        "High (4+)"
        if data["NumberOfPenalties"] > 3
        else "Low (0-3)"
    )

    fee = data["BaseFee"]

    if 0 < fee <= 20:
        data["FeeBand"] = "<=20"
    elif 20 < fee <= 30:
        data["FeeBand"] = "20-30"
    elif 30 < fee <= 38:
        data["FeeBand"] = "30-38"
    elif fee > 38:
        data["FeeBand"] = ">38"
    else:
        data["FeeBand"] = None

    tenure = data["CustomerTenure"]

    if tenure <= 60:
        data["TenureGroup"] = "New"
    elif tenure <= 90:
        data["TenureGroup"] = "Mid"
    elif tenure <= 120:
        data["TenureGroup"] = "Established"
    else:
        data["TenureGroup"] = "Long-term"

    data["RedPlan_x_Penalties"] = (
        int(data["PlanType"] == "Red Plan")
        * data["NumberOfPenalties"]
    )

    data["Penalties_per_Tenure"] = (
        data["NumberOfPenalties"] / (tenure + 1)
    )

    data["Tickets_per_Tenure"] = (
        data["ServiceTickets"] / (tenure + 1)
    )

    # Step 3: Convert input into a DataFrame
    input_data = pd.DataFrame([data])

    # Step 4: Use the exact feature order expected by the model
    expected_features = list(model.feature_names_in_)
    input_data = input_data[expected_features]

    # Step 5: Predict churn
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)
    churn_index = classes.index(1)
    probability = float(probabilities[churn_index])

    # Step 6: Return the prediction
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(
            probability * 100, 2
        )
    }
