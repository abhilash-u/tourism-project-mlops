
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Visit With Us — Tourism Package Predictor", page_icon="🧳", layout="centered")

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="Abhilashu/tourism-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

st.title("Visit with us Tourism Package Purchase — Prediction")
st.write("Fill the details and click **Predict**. The model estimates the probability that a customer will buy the Tourism Package.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=18, max_value=90, value=35, step=1)
        CityTier = st.number_input("CityTier (1=metro, 2, 3)", min_value=1, max_value=3, value=1, step=1)
        DurationOfPitch = st.number_input("DurationOfPitch (minutes)", min_value=0.0, value=10.0, step=1.0)
        NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1.0, value=3.0, step=1.0)
        NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0.0, value=3.0, step=1.0)
        PreferredPropertyStar = st.number_input("PreferredPropertyStar (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

    with col2:
        NumberOfTrips = st.number_input("NumberOfTrips (per year)", min_value=0.0, value=2.0, step=1.0)
        Passport = st.selectbox("Passport", options=[0,1], index=1)
        PitchSatisfactionScore = st.number_input("PitchSatisfactionScore (1-5)", min_value=1.0, max_value=5.0, value=3.0, step=1.0)
        OwnCar = st.selectbox("OwnCar", options=[0,1], index=0)
        NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting (under 5)", min_value=0, value=0, step=1)
        MonthlyIncome = st.number_input("MonthlyIncome", min_value=0.0, value=25000.0, step=500.0)

    TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Enquiry"])
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Free Lancer"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ProductPitched = st.selectbox("ProductPitched", ["Basic", "Deluxe", "Standard"])
    MaritalStatus = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])

    submitted = st.form_submit_button("Predict")

# Set the classification threshold
classification_threshold = 0.5

if submitted:
    # NOTE: include ALL training features
    row = {
        "Age": float(Age),
        "CityTier": float(CityTier),
        "DurationOfPitch": float(DurationOfPitch),
        "TypeofContact": str(TypeofContact).strip(),
        "Occupation": str(Occupation).strip(),
        "Gender": str(Gender).strip(),
        "NumberOfPersonVisiting": float(NumberOfPersonVisiting),
        "NumberOfFollowups": float(NumberOfFollowups),
        "ProductPitched": str(ProductPitched).strip(),
        "PreferredPropertyStar": float(PreferredPropertyStar),
        "MaritalStatus": str(MaritalStatus).strip(),
        "NumberOfTrips": float(NumberOfTrips),
        "Passport": float(Passport),
        "PitchSatisfactionScore": float(PitchSatisfactionScore),
        "OwnCar": float(OwnCar),
        "NumberOfChildrenVisiting": float(NumberOfChildrenVisiting),
        "Designation": str(Designation).strip(),
        "MonthlyIncome": float(MonthlyIncome),
    }

    X = pd.DataFrame([row])

    proba = model.predict_proba(X)[:, 1][0]
    pred  = int(proba >= classification_threshold)

    st.subheader("Result")
    st.metric("Predicted probability of purchase", f"{proba:.3f}")
    st.write("Prediction:", "**Yes**" if pred==1 else "**No**")
