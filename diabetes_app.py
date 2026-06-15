import streamlit as st
import numpy as np
import pickle

# Load trained model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🩺 Diabetes Prediction System")

st.write("Enter patient details below:")

pregnancies = st.number_input("Pregnancies", min_value=0, value=0)
glucose = st.number_input("Glucose Level", min_value=0, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=0, value=30)

if st.button("Predict"):

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        pedigree,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The person is Diabetic")
    else:
        st.success("✅ The person is Not Diabetic")
        st.sidebar.header("About Project")

st.sidebar.info("""
Machine Learning model trained on the Pima Indians Diabetes Dataset.

Algorithms:
- Data Preprocessing
- StandardScaler
- Support Vector Machine

Developer:
Saurabh Singh
""")