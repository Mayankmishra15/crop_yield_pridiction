import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and feature columns
model = joblib.load("crop_yield_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.title("🌾 Crop Yield Prediction System")

# ---- USER INPUTS ----
crop_year = st.number_input("Crop Year", min_value=1990, max_value=2035, value=2020)
area = st.number_input("Area (hectares)", min_value=0.0)
rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0)
fertilizer = st.number_input("Fertilizer usage", min_value=0.0)
pesticide = st.number_input("Pesticide usage", min_value=0.0)
yield_lag1 = st.number_input("Previous Year Yield", min_value=0.0)

season = st.selectbox(
    "Season",
    ["Kharif", "Rabi", "Summer", "Whole Year", "Winter"]
)

state = st.selectbox(
    "State",
    [
        "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh",
        "Delhi","Goa","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir",
        "Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra",
        "Manipur","Meghalaya","Mizoram","Nagaland","Odisha","Puducherry",
        "Punjab","Sikkim","Tamil Nadu","Telangana","Tripura",
        "Uttar Pradesh","Uttarakhand","West Bengal"
    ]
)
if st.button("Predict Yield"):
    # Create empty dataframe with all trained features
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill numerical values
    input_df["Crop_Year"] = crop_year
    input_df["Area"] = area
    input_df["Annual_Rainfall"] = rainfall
    input_df["Fertilizer"] = fertilizer
    input_df["Pesticide"] = pesticide
    input_df["Yield_lag1"] = yield_lag1

    # One-hot encode season
    season_col = f"Season_{season} "
    if season_col in input_df.columns:
        input_df[season_col] = 1

    # One-hot encode state
    state_col = f"State_{state}"
    if state_col in input_df.columns:
        input_df[state_col] = 1

    # Predict
    prediction = model.predict(input_df)
    prediction = np.expm1(prediction)

    st.success(f"🌱 Predicted Crop Yield: {prediction[0]:.2f}")
