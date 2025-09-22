import streamlit as st
import pickle
import numpy as np

# Load the trained Linear Regression model
with open('Linear_Regression.pkl', 'rb') as file:
    model = pickle.load(file)

# Set up the Streamlit app
st.set_page_config(page_title="Insurance Charge Predictor", layout="centered")
st.title("💰 Insurance Charges Prediction")
st.markdown("Predict insurance charges based on Age, BMI, and Smoking status.")

# Input fields
age = st.slider("Enter Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("Enter BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
smoker = st.radio("Do you smoke?", ["Yes", "No"])

# Convert smoker to binary
smoker_val = 1 if smoker == "Yes" else 0

# Predict button
if st.button("Predict Charges"):
    # Prepare the input data
    input_data = np.array([[age, bmi, smoker_val]])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(f"💵 Predicted Insurance Charges: ${prediction:,.2f}")
