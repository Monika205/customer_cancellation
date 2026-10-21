
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Find the saved model
model_candidates = [
    "customer_churn_model.joblib",
    "customer_churn_model_v1.joblib",
    "saved_models/customer_churn_model.joblib",
    "saved_models/customer_churn_model_v1.joblib"
]

MODEL_PATH = next(
    (
        os.path.join(BASE_DIR, p)
        for p in model_candidates
        if os.path.isfile(os.path.join(BASE_DIR, p))
    ),
    None
)

# Find the training dataset
data_candidates = [
    "Customer_Cancellations_Training.csv",
    "Customer_Cancellations.csv",
    "customer_cancellations.csv",
    "customer_churn.csv",
    "data/Customer_Cancellations_Training.csv"
]

DATA_PATH = next(
    (
        os.path.join(BASE_DIR, p)
        for p in data_candidates
        if os.path.isfile(os.path.join(BASE_DIR, p))
    ),
    None
)

st.title("📊 Customer Churn Intelligence")
st.write("Machine Learning Based Customer Retention Dashboard")
st.divider()

if MODEL_PATH is None:
    st.error("Model file not found. Check its filename and location in GitHub.")
    st.stop()

if DATA_PATH is None:
    st.error("Training CSV not found. Check its filename and location in GitHub.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
except Exception as error:
    st.error(f"Could not load model or dataset: {error}")
    st.stop()

required_columns = [
    "Territory", "DeviceType", "PlanType", "Promotion",
    "StartMonth", "StartWeek", "CustomerTenure", "BaseFee",
    "AdditionalFeatureSpend", "CurrentPeriodUsage",
    "PriorPeriodUsage", "NumberOfPenalties", "ServiceTickets",
    "ServiceRating"
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    st.error(f"Dataset is missing these columns: {missing}")
    st.stop()

def median_value(column):
    return float(data[column].median())

st.sidebar.header("About this Dashboard")
st.sidebar.write(
    "Enter customer and subscription details to estimate "
    "the probability of customer churn."
)
st.sidebar.success("Model and dataset loaded")

st.subheader("1. Customer and Subscription Details")

c1, c2, c3 = st.columns(3)

with c1:
    territory = st.selectbox(
        "Territory", sorted(data["Territory"].dropna().unique())
    )
    device = st.selectbox(
        "Device Type", sorted(data["DeviceType"].dropna().unique())
    )
    plan = st.selectbox(
        "Plan Type", sorted(data["PlanType"].dropna().unique())
    )

with c2:
    promotion = st.selectbox(
        "Promotion", sorted(data["Promotion"].dropna().unique())
    )
    months = sorted(
        data["StartMonth"].dropna().unique(),
        key=lambda x: pd.to_datetime(str(x), format="%b").month
    )
    month = st.selectbox("Start Month", months)

    weeks = sorted(
        data["StartWeek"].dropna().unique(),
        key=lambda x: int(str(x)[1:])
    )
    week = st.selectbox("Start Week", weeks)

with c3:
    tenure = st.number_input(
        "Customer Tenure",
        min_value=0,
        value=max(0, int(median_value("CustomerTenure")))
    )
    base_fee = st.number_input(
        "Base Fee",
        min_value=0.0,
        value=max(0.0, median_value("BaseFee"))
    )
    extra_spend = st.number_input(
        "Additional Feature Spend",
        min_value=0.0,
        value=max(0.0, median_value("AdditionalFeatureSpend"))
    )

st.subheader("2. Usage and Service Experience")

a1, a2, a3 = st.columns(3)

with a1:
    current_usage = st.number_input(
        "Current Period Usage",
        min_value=0.0,
        value=max(0.0, median_value("CurrentPeriodUsage"))
    )
    prior_usage = st.number_input(
        "Prior Period Usage",
        min_value=0.0,
        value=max(0.0, median_value("PriorPeriodUsage"))
    )

with a2:
    penalties = st.number_input(
        "Number of Penalties",
        min_value=0,
        value=max(0, int(median_value("NumberOfPenalties")))
    )
    tickets = st.number_input(
        "Service Tickets",
        min_value=0,
        value=max(0, int(median_value("ServiceTickets")))
    )

with a3:
    rating = st.number_input(
        "Service Rating",
        min_value=0.0,
        value=max(0.0, median_value("ServiceRating")),
        step=0.5
    )

if st.button("Analyze Customer Churn", use_container_width=True):
    try:
        month_number = pd.to_datetime(
            str(month), format="%b"
        ).month
        week_number = int(str(week)[1:])

        usage_difference = current_usage - prior_usage

        # Avoid NaN when Base Fee is zero
        additional_spend_ratio = (
            extra_spend / base_fee if base_fee != 0 else 0.0
        )
        total_spend = base_fee + extra_spend

        penalty_level = (
            "Low (0-3)" if penalties <= 3 else "High (4+)"
        )

        if 0 < base_fee <= 20:
            fee_band = "<=20"
        elif 20 < base_fee <= 30:
            fee_band = "20-30"
        elif 30 < base_fee <= 38:
            fee_band = "30-38"
        elif base_fee > 38:
            fee_band = ">38"
        else:
            fee_band = None

        if tenure <= 60:
            tenure_group = "New"
        elif tenure <= 90:
            tenure_group = "Mid"
        elif tenure <= 120:
            tenure_group = "Established"
        else:
            tenure_group = "Long-term"

        customer = pd.DataFrame([{
            "Territory": territory,
            "DeviceType": device,
            "Promotion": promotion,
            "PlanType": plan,
            "BaseFee": base_fee,
            "AdditionalFeatureSpend": extra_spend,
            "NumberOfPenalties": penalties,
            "CurrentPeriodUsage": current_usage,
            "PriorPeriodUsage": prior_usage,
            "ServiceRating": rating,
            "ServiceTickets": tickets,
            "CustomerTenure": tenure,
            "UsageDifference": usage_difference,
            "AdditionalSpendRatio": additional_spend_ratio,
            "TotalSpend": total_spend,
            "StartMonthNumber": month_number,
            "StartWeekNumber": week_number,
            "PenaltyLevel": penalty_level,
            "FeeBand": fee_band,
            "TenureGroup": tenure_group,
            "RedPlan_x_Penalties": int(plan == "Red Plan") * penalties,
            "Penalties_per_Tenure": penalties / (tenure + 1),
            "Tickets_per_Tenure": tickets / (tenure + 1)
        }])

        # Match the exact feature order used during model training
        expected_features = list(model.feature_names_in_)
        customer = customer[expected_features]

        prediction = int(model.predict(customer)[0])
        probabilities = model.predict_proba(customer)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            st.error("The model does not contain churn class 1.")
            st.stop()

        probability = float(probabilities[classes.index(1)])
        outcome = "Churn" if prediction == 1 else "No Churn"

        st.divider()
        st.subheader("Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Predicted Outcome", outcome)
        r2.metric("Churn Probability", f"{probability:.2%}")
        r3.metric("Customer Tenure", str(tenure))

        st.write("Churn Probability")
        st.progress(float(np.clip(probability, 0, 1)))

        if prediction == 1:
            st.warning(
                "The model predicts churn. Review the customer's "
                "service experience and possible retention options."
            )
        else:
            st.success(
                "The model predicts no churn. Continue normal "
                "customer monitoring and service support."
            )

        report = pd.DataFrame([{
            "Prediction": outcome,
            "Churn Probability (%)": round(probability * 100, 2),
            "Territory": territory,
            "DeviceType": device,
            "PlanType": plan,
            "Promotion": promotion,
            "CustomerTenure": tenure,
            "BaseFee": base_fee,
            "AdditionalFeatureSpend": extra_spend
        }])

        st.subheader("Customer Summary")
        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction Report (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_prediction.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the saved model and feature values match "
            "the training notebook."
        )

st.divider()
st.caption(
    "Customer Churn Intelligence | Machine Learning Assessment Project. "
    "Predictions are estimates, not guarantees."
)
