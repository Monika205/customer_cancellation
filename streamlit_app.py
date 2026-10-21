
#Step 39

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Find files whether they are stored in the repository root or a subfolder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_first_existing(candidates):
    """Return the first existing file path from a list of relative paths."""
    for relative_path in candidates:
        full_path = os.path.join(BASE_DIR, relative_path)
        if os.path.isfile(full_path):
            return full_path
    return None


MODEL_PATH = find_first_existing([
    "customer_churn_model.joblib",
    "customer_churn_model_v1.joblib",
    os.path.join("saved_models", "customer_churn_model.joblib"),
    os.path.join("saved_models", "customer_churn_model_v1.joblib"),
])

DATA_PATH = find_first_existing([
    "Customer_Cancellations_Training.csv",
    "Customer_Cancellations.csv",
    "customer_cancellations.csv",
    "customer_churn.csv",
    os.path.join("data", "Customer_Cancellations_Training.csv"),
    os.path.join("data", "Customer_Cancellations.csv"),
])

# If the dataset uses a different filename, search the repository for CSV files.
if DATA_PATH is None:
    csv_candidates = []
    for folder in (BASE_DIR, os.path.join(BASE_DIR, "data")):
        if os.path.isdir(folder):
            csv_candidates.extend(
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if name.lower().endswith(".csv")
            )
    for candidate in csv_candidates:
        try:
            columns = set(pd.read_csv(candidate, nrows=0).columns)
            if {"Territory", "DeviceType", "PlanType", "CustomerTenure"}.issubset(columns):
                DATA_PATH = candidate
                break
        except Exception:
            continue

Chat

New Chat
Hi,
👋
How can I help you today?


Explain a complex thing
Explain Artificial Intelligence so that I can explain it to my six-year-old child.

Get suggestions and create new ideas
Please give me the best 10 travel ideas around the world

Translate, summarize, fix grammar and more…
Translate "I love you" French

AITOPIA







Ask me anything...

Group Chat
Web Access

10
Upgrade




