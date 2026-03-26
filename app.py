import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

# Page config
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# 🔥 COLORFUL CSS (no layout change)
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

/* Title */
h1 {
    text-align: center;
    color: #00FFD1;
}

/* Input boxes */
.stNumberInput, .stSelectbox {
    background-color: #1f2a40 !important;
    border-radius: 10px;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
}

/* Result box */
.result-box {
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details:")

# SAME INPUTS (no change)
tenure = st.number_input("Tenure", 0, 100, 1)
monthly_charges = st.number_input("Monthly Charges", 0.0, 1000.0, 50.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Encoding
gender = 1 if gender == "Male" else 0
senior = 1 if senior == "Yes" else 0

contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
contract = contract_map[contract]

# Button
if st.button("🚀 Predict"):
    input_data = np.array([[tenure, monthly_charges, total_charges, gender, senior, contract]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.markdown(
            "<div class='result-box' style='background-color:#ff4b5c;'>⚠️ Customer will Churn</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box' style='background-color:#00C897;'>✅ Customer will NOT Churn</div>",
            unsafe_allow_html=True
        )