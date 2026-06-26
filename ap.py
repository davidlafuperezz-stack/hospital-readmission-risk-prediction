import streamlit as st
import pandas as pd
import joblib

# ======================================================
# 📦 LOAD MODEL
# ======================================================
model = joblib.load("model.pkl")

# ======================================================
# 🏥 APP TITLE
# ======================================================
st.title("🏥 Hospital Readmission Risk Predictor")
st.write("Predict 30-day hospital readmission risk using clinical data and machine learning.")

st.write("---")

# ======================================================
# 🎛 INPUTS
# ======================================================
age = st.slider("Age", 18, 95, 50)
comorbidity_index = st.slider("Comorbidity Index", 0, 10, 2)
length_of_stay = st.slider("Length of Stay (days)", 1, 30, 5)
icu_admission = st.selectbox("ICU Admission", [0, 1])
emergency_admission = st.selectbox("Emergency Admission", [0, 1])
num_medications = st.slider("Number of Medications", 0, 30, 8)
glucose = st.slider("Glucose Level", 50, 300, 110)
hemoglobin = st.slider("Hemoglobin Level", 5, 20, 13)

# ======================================================
# 📊 BUILD INPUT DATA
# ======================================================
input_data = pd.DataFrame([{
    "age": age,
    "comorbidity_index": comorbidity_index,
    "length_of_stay": length_of_stay,
    "icu_admission": icu_admission,
    "emergency_admission": emergency_admission,
    "num_medications": num_medications,
    "glucose": glucose,
    "hemoglobin": hemoglobin
}])

# ======================================================
# 🤖 PREDICTION
# ======================================================
prob = model.predict_proba(input_data)[0][1]

st.subheader("Prediction Result")

st.write(f"Readmission Probability: **{prob:.2f}**")

# ======================================================
# ⚠ RISK CLASSIFICATION
# ======================================================
if prob < 0.3:
    st.success("🟢 Low Risk")
    st.write("Patient has low probability of 30-day readmission.")
elif prob < 0.7:
    st.warning("🟡 Medium Risk")
    st.write("Patient should be monitored and follow-up recommended.")
else:
    st.error("🔴 High Risk")
    st.write("Patient has high risk of readmission. Intervention recommended.")

st.write("---")

# ======================================================
# 🧠 MODEL INFO
# ======================================================
st.write("### Model Information")
st.write("- Algorithm: XGBoost")
st.write("- Trained on synthetic hospital readmission dataset")
st.write("- Purpose: Clinical decision support simulation")

st.write("---")

# ======================================================
# 💡 FOOTER INSIGHT
# ======================================================
st.write("This tool simulates how machine learning can assist healthcare professionals in identifying high-risk patients.")
