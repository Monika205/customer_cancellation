
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Step 1: Convert the API input into a dictionary
    data = customer.model_dump()

    # Step 2: Create the missing features using notebook formulas
    data["PenaltyLevel"] = (
        "High (4+)" if data["NumberOfPenalties"] > 3
        else "Low (0-3)"
    )

    # FeeBand based on BaseFee
    fee = data["BaseFee"]

    if 0 < fee <= 20:
        data["FeeBand"] = "<=20"
    elif 20 < fee <= 30:
        data["FeeBand"] = "20-30"
    elif 30 < fee <= 38:
        data["FeeBand"] = "30-38"
    elif 38 < fee <= 100:
        data["FeeBand"] = ">38"
    else:
        data["FeeBand"] = None

    # TenureGroup based on CustomerTenure
    tenure = data["CustomerTenure"]

    if tenure <= 60:
        data["TenureGroup"] = "New"
    elif tenure <= 90:
        data["TenureGroup"] = "Mid"
    elif tenure <= 120:
        data["TenureGroup"] = "Established"
    else:
        data["TenureGroup"] = "Long-term"

    # Interaction and ratio features
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

    # Step 3: Create the input DataFrame
    input_data = pd.DataFrame([data])

    # Step 4: Match the feature columns expected by the model
    expected_features = list(model.feature_names_in_)
    input_data = input_data[expected_features]

    # Step 5: Predict churn
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    classes = list(model.classes_)
    churn_index = classes.index(1)
    probability = float(probabilities[churn_index])

    # Step 6: Return the result
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(probability, 4),
        "churn_probability_percent": round(
            probability * 100, 2
        )
    }
