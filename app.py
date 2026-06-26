# app.py

import os
import streamlit as st
import pandas as pd
import joblib


# ======================================================
# PASO 1 - APP CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Hospital Readmission Risk Predictor",
    page_icon="🏥",
    layout="centered"
)


# ======================================================
# PASO 2 - LOAD TRAINED MODEL AND MODEL COLUMNS
# ======================================================
# The model was trained in ml.py and saved inside the models/ folder.
# model.pkl contains the trained XGBoost model.
# model_columns.pkl contains the exact columns used during training.

MODEL_PATH = os.path.join("models", "model.pkl")
COLUMNS_PATH = os.path.join("models", "model_columns.pkl")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)


# ======================================================
# PASO 3 - APP TITLE AND DESCRIPTION
# ======================================================

st.title("🏥 Hospital Readmission Risk Predictor")

st.write(
    "This app predicts the probability of a patient being readmitted "
    "within 30 days after hospital discharge using a machine learning model."
)

st.info(
    "This project is for educational and portfolio purposes. "
    "It is not intended for real clinical decision-making."
)

st.write("---")


# ======================================================
# PASO 4 - PATIENT INPUT FORM
# ======================================================
# These inputs represent the main clinical and hospital variables used by the model.

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 95, 50)
    sex = st.selectbox("Sex", [0, 1])
    bmi = st.slider("BMI", 15.0, 45.0, 27.0)
    ethnicity = st.selectbox(
        "Ethnicity",
        ["African", "Asian", "Caucasian", "Hispanic", "Other"]
    )
    admission_type = st.selectbox("Admission Type", [0, 1, 2])
    length_of_stay = st.slider("Length of Stay (days)", 1.0, 30.0, 5.0)
    icu_admission = st.selectbox("ICU Admission", [0, 1])
    emergency_admission = st.selectbox("Emergency Admission", [0, 1])
    num_procedures = st.slider("Number of Procedures", 0, 10, 2)
    num_medications = st.slider("Number of Medications", 0, 30, 8)

with col2:
    discharge_type = st.selectbox("Discharge Type", [0, 1, 2])
    diabetes = st.selectbox("Diabetes", [0, 1])
    hypertension = st.selectbox("Hypertension", [0, 1])
    copd = st.selectbox("COPD", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
    chronic_kidney_disease = st.selectbox("Chronic Kidney Disease", [0, 1])
    comorbidity_index = st.slider("Comorbidity Index", 0, 10, 2)
    previous_admissions = st.slider("Previous Admissions", 0, 10, 1)
    readmissions_history = st.selectbox("Readmissions History", [0, 1])
    follow_up_scheduled = st.selectbox("Follow-up Scheduled", [0, 1])


st.subheader("Clinical Measurements")

col3, col4 = st.columns(2)

with col3:
    glucose = st.slider("Glucose", 10.0, 300.0, 110.0)
    hemoglobin = st.slider("Hemoglobin", 5.0, 20.0, 13.0)
    creatinine = st.slider("Creatinine", 0.0, 3.0, 1.0)
    white_blood_cells = st.slider("White Blood Cells", 0.0, 20.0, 7.0)

with col4:
    blood_pressure_systolic = st.slider("Systolic Blood Pressure", 70.0, 200.0, 120.0)
    blood_pressure_diastolic = st.slider("Diastolic Blood Pressure", 40.0, 120.0, 80.0)
    oxygen_saturation = st.slider("Oxygen Saturation", 80.0, 100.0, 96.0)
    temperature = st.slider("Temperature", 35.0, 40.0, 36.8)


chronic_disease_score = st.slider("Chronic Disease Score", 0, 10, 2)
medication_complexity = st.slider("Medication Complexity", 0, 10, 3)
discharge_delay = st.slider("Discharge Delay", 0.0, 5.0, 1.0)


# ======================================================
# PASO 5 - BUILD INPUT DATAFRAME
# ======================================================
# The input dictionary must have the same raw columns as the dataset.
# Later, we apply pd.get_dummies() to match the training process.

input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "bmi": bmi,
    "ethnicity": ethnicity,
    "admission_type": admission_type,
    "length_of_stay": length_of_stay,
    "icu_admission": icu_admission,
    "emergency_admission": emergency_admission,
    "num_procedures": num_procedures,
    "num_medications": num_medications,
    "discharge_type": discharge_type,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "copd": copd,
    "heart_disease": heart_disease,
    "chronic_kidney_disease": chronic_kidney_disease,
    "comorbidity_index": comorbidity_index,
    "glucose": glucose,
    "hemoglobin": hemoglobin,
    "creatinine": creatinine,
    "white_blood_cells": white_blood_cells,
    "blood_pressure_systolic": blood_pressure_systolic,
    "blood_pressure_diastolic": blood_pressure_diastolic,
    "oxygen_saturation": oxygen_saturation,
    "temperature": temperature,
    "previous_admissions": previous_admissions,
    "readmissions_history": readmissions_history,
    "chronic_disease_score": chronic_disease_score,
    "medication_complexity": medication_complexity,
    "follow_up_scheduled": follow_up_scheduled,
    "discharge_delay": discharge_delay
}])


# ======================================================
# PASO 6 - PREPROCESS INPUT DATA
# ======================================================
# The model was trained using pd.get_dummies().
# Therefore, the app must apply the same transformation.
# Then we reindex the columns to match the training columns exactly.

input_encoded = pd.get_dummies(input_data)

input_encoded = input_encoded.reindex(
    columns=model_columns,
    fill_value=0
)


# ======================================================
# PASO 7 - MAKE PREDICTION
# ======================================================

st.write("---")

if st.button("Predict Readmission Risk"):

    probability = model.predict_proba(input_encoded)[0][1]
    prediction = model.predict(input_encoded)[0]

    st.subheader("Prediction Result")

    st.metric(
        label="30-Day Readmission Probability",
        value=f"{probability * 100:.1f}%"
    )

    if probability < 0.30:
        st.success("🟢 Low Risk")
        st.write("The patient has a low estimated probability of 30-day readmission.")

    elif probability < 0.70:
        st.warning("🟡 Medium Risk")
        st.write("The patient may benefit from closer follow-up after discharge.")

    else:
        st.error("🔴 High Risk")
        st.write("The patient has a high estimated probability of readmission.")

    if prediction == 1:
        st.write("Model prediction: **Readmission likely**")
    else:
        st.write("Model prediction: **Readmission not likely**")


# ======================================================
# PASO 8 - MODEL INFORMATION
# ======================================================

st.write("---")

st.write("### Model Information")
st.write("- Algorithm: XGBoost Classifier")
st.write("- Target: 30-day hospital readmission")
st.write("- Data type: Synthetic hospital readmission dataset")
st.write("- Use case: Machine learning portfolio project")
st.write("- Output: Estimated probability of readmission")

st.write("---")

st.caption(
    "This tool is a simulation for educational purposes and should not be used "
    "as a medical device or clinical decision system."
)
