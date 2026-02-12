import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------
# Load Model and Scaler
# ---------------------------------------------------------
model = pickle.load(open("Random Forest.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Bank Term Deposit Subscription Prediction")
st.write("Predict whether a customer will subscribe to a term deposit.")

# ---------------------------------------------------------
# User Input Form
# ---------------------------------------------------------
st.header("Enter Customer Details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Account Balance", value=1000)
duration = st.number_input("Last Contact Duration (seconds)", value=100)
campaign = st.number_input("Number of Contacts During Campaign", value=1)
pdays = st.number_input("Days Passed After Last Contact (-1 means never)", value=-1)
previous = st.number_input("Number of Previous Contacts", value=0)

job = st.selectbox("Job", [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
])

marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
default = st.selectbox("Credit Default", ["yes", "no"])
housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
contact = st.selectbox("Contact Type", ["cellular", "telephone"])
month = st.selectbox("Last Contact Month", [
    "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
])
poutcome = st.selectbox("Previous Campaign Outcome", ["success", "failure", "other", "unknown"])

# ---------------------------------------------------------
# Convert Inputs to DataFrame
# ---------------------------------------------------------
input_data = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "duration": [duration],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "default": [default],
    "housing": [housing],
    "loan": [loan],
    "contact": [contact],
    "month": [month],
    "poutcome": [poutcome]
})

# ---------------------------------------------------------
# One-Hot Encode Inputs (same as training)
# ---------------------------------------------------------
input_encoded = pd.get_dummies(input_data)

# Align with training columns
training_columns = pickle.load(open("columns.pkl", "rb"))
input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

# ---------------------------------------------------------
# Scale Numerical Features
# ---------------------------------------------------------
scaled_input = scaler.transform(input_encoded)

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"Customer is LIKELY to subscribe. Probability: {probability:.2f}")
    else:
        st.error(f"Customer is NOT likely to subscribe. Probability: {probability:.2f}")