import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Court to Contract",
    page_icon="🏀",
    layout="centered"
)


# ============================================
# LOAD MODEL
# ============================================

model = joblib.load("nba_salary_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# ============================================
# COURT TO CONTRACT
# ============================================

st.title("🏀 Court to Contract")
st.subheader("NBA Player Salary Prediction Model")

st.write(
    "This application predicts an NBA player's salary cap percentage "
    "using a Random Forest Regression model."
)


# ============================================
# INPUT SECTION
# ============================================

st.header("Enter Player Information")

input_data = {}

for feature in feature_names:

    if feature == "salary_cap_year":
        input_data[feature] = st.number_input(
            "Salary Cap Year",
            min_value=1.0,
            value=100.0
        )

    elif feature in ["age", "years_experience"]:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            min_value=0.0,
            value=25.0
        )

    else:
        input_data[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=0.0
        )


# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the columns are in the same order as during training
input_df = input_df[feature_names]


# ============================================
# PREDICTION
# ============================================

if st.button("Predict Salary"):

    # Predict salary cap percentage
    predicted_salary_cap_pct = model.predict(input_df)[0]

    st.success("Prediction Completed Successfully!")

    st.metric(
        "Predicted Salary Cap Percentage",
        f"{predicted_salary_cap_pct:.2%}"
    )

    # Convert normalized prediction to actual salary
    salary_cap_year = input_data["salary_cap_year"]

    predicted_salary = (
        predicted_salary_cap_pct * salary_cap_year
    )

    st.metric(
        "Estimated NBA Salary",
        f"${predicted_salary:,.2f}"
    )

    st.info(
        "The prediction is generated using the trained Random Forest "
        "Regression model."
    )


# ============================================
# MODEL INFORMATION
# ============================================

st.divider()

st.subheader("About the Model")

st.write(
    "The model was developed using Random Forest Regression. "
    "The normalized target variable, salary_cap_pct, was used to "
    "reduce the effect of salary scale variation and minimize overfitting."
)