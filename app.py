import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model/model.pkl")

st.title("Customer Churn Prediction")

st.write("Enter customer details:")

# Example Inputs (basic)
tenure = st.number_input("Tenure", min_value=0, max_value=100, value=1)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total_charges = st.number_input("Total Charges", min_value=0.0, value=100.0)

# Predict button
if st.button("Predict"):
    # Create input array (IMPORTANT: order same as training)
    input_data = np.array([[tenure, monthly_charges, total_charges]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will Churn ")
    else:
        st.success("Customer will NOT Churn ✅")