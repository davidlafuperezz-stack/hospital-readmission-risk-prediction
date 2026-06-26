import numpy as np
import pandas as pd 

np.random.seed(42)

n = 8000

#DEMOGRAPHICS
age = np.random.randint(18, 95, n)
sex = np.random.binomial(1, 0.52, n)
bmi = np.clip(np.random.normal(27, 6, n), 15, 55)

ethnicity = np.random.choice(["Caucasian", "African", "Asian", "Hispanic", "Other"], n)

# ADMISSION DETAILS
admission_type = np.random.choice([0, 1, 2], n)  # emergency, urgent, elective
length_of_stay = np.clip(np.random.normal(6, 4, n), 1, 30)

icu_admission = np.random.binomial(1, 0.15, n)
emergency_admission = (admission_type == 0).astype(int)

num_procedures = np.random.poisson(2, n)
num_medications = np.random.poisson(8, n)

discharge_type = np.random.choice([0, 1, 2], n)  # home, rehab, transfer

# COMORBIDITIES
diabetes = np.random.binomial(1, 0.25, n)
hypertension = np.random.binomial(1, 0.35, n)
copd = np.random.binomial(1, 0.12, n)
heart_disease = np.random.binomial(1, 0.18, n)
chronic_kidney_disease = np.random.binomial(1, 0.10, n)

comorbidity_index = diabetes + hypertension + copd + heart_disease + chronic_kidney_disease

# LAB RESULTS
glucose = np.random.normal(110, 40, n)
hemoglobin = np.random.normal(13.5, 2.2, n)
creatinine = np.abs(np.random.normal(1.1, 0.5, n))
white_blood_cells = np.random.normal(7.5, 2.5, n)

blood_pressure_systolic = np.random.normal(130, 20, n)
blood_pressure_diastolic = np.random.normal(80, 10, n)

oxygen_saturation = np.clip(np.random.normal(96, 3, n), 70, 100)

temperature = np.random.normal(36.8, 0.8, n)

# HISTORY
previous_admissions = np.random.poisson(1.5, n)
readmissions_history = np.random.binomial(1, 0.3, n)

chronic_disease_score = comorbidity_index + previous_admissions

#HOSPITAL CARE QUALITY
medication_complexity = np.random.poisson(3, n)
follow_up_scheduled = np.random.binomial(1, 0.6, n)

discharge_delay = np.clip(np.random.normal(1.5, 1, n), 0, 10)

# RISK MODEL (REALISTIC LOGIC)
risk = (
    0.03 * age +
    1.2 * comorbidity_index +
    0.8 * previous_admissions +
    0.5 * length_of_stay +
    0.7 * icu_admission +
    0.6 * emergency_admission +
    0.4 * medication_complexity +
    0.3 * discharge_delay +
    0.02 * glucose +
    0.5 * readmissions_history -
    1.0 * follow_up_scheduled
)

risk += np.random.normal(0, 2, n)

prob = 1 / (1 + np.exp(-0.25 * (risk - 8)))

readmitted_30_days = np.random.binomial(1, prob)


# DATAFRAME
df = pd.DataFrame({
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
    "discharge_delay": discharge_delay,

    "readmitted_30_days": readmitted_30_days
})

#SAVE
df.to_csv("hospital_readmission_dataset.csv", index=False)

print("Dataset created:", df.shape)
