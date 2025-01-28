import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained SVM model, scaler, and label encoder
svm_model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Streamlit App
st.set_page_config(
    page_title="Purchase Prediction App",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Set background color to light
st.markdown(
    """
    <style>
        body {
            background-color: #f7f7f7;
        }
        .reportview-container {
            background-color: #f7f7f7;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Heading for the app
st.title("Purchase Prediction App")
st.write("Predict whether a user will make a purchase based on age, gender, and estimated salary.")

# Input form
st.markdown("---")
st.header("Enter User Details:")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
estimated_salary = st.number_input("Estimated Salary (in $)", min_value=10000, max_value=200000, value=50000, step=5000)

# Prediction button
if st.button("Predict"):
    # Encode gender
    gender_encoded = label_encoder.transform([gender])[0]

    # Scale numerical input features
    numerical_features = np.array([[age, estimated_salary]])
    numerical_features_scaled = scaler.transform(numerical_features)

    # Combine scaled numerical features and encoded gender
    input_features_scaled = np.hstack([numerical_features_scaled, [[gender_encoded]]])

    # Make prediction
    prediction = svm_model.predict(input_features_scaled)[0]

    # Display the result
    st.markdown("---")
    if prediction == 1:
        st.write("The model predicts: **Purchased**")
    else:
        st.write("The model predicts: **Not Purchased**")

# Footer
st.markdown(
    """
    <hr>
    <p style="text-align: center; color: #888;">Developed using Streamlit</p>
    """,
    unsafe_allow_html=True
)
