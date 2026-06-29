# Hospital Readmission Risk Prediction System

## Overview

This project is an end-to-end machine learning system designed to predict the probability of 30-day hospital readmission using clinical and hospital data.

It includes exploratory data analysis, feature engineering, model training, evaluation, and a deployed interactive web application using Streamlit.

---

## Objective

The main objective is to identify patients at high risk of readmission within 30 days after hospital discharge, enabling better clinical decision support and resource planning.

---

## Dataset

The dataset contains anonymized hospital and clinical information, including:

- Patient demographics
- Clinical measurements
- Comorbidities
- Hospital stay characteristics
- Admission and discharge information

Target variable:

- `readmitted_30_days` (binary classification: 0 = no readmission, 1 = readmission)

---

## Project Workflow

The project follows a complete machine learning pipeline:

1. Exploratory Data Analysis (EDA)
2. Feature engineering and preprocessing
3. Model training and comparison
4. Model evaluation
5. Model selection
6. Deployment with Streamlit

---

## Exploratory Data Analysis

The EDA focuses on understanding patterns and risk factors associated with hospital readmission.

Key analyses include:

- Distribution of 30-day readmissions
- Age distribution analysis
- Length of stay impact
- Comorbidity index analysis
- ICU and emergency admission interaction
- Follow-up appointment impact
- Correlation analysis
- Risk group segmentation
- Kaplan-Meier survival analysis

---

## Machine Learning Models

Three supervised learning models were trained and compared:

### Logistic Regression
Used as a baseline model with standardized features.

### Random Forest
Ensemble model with multiple decision trees and class balancing.

### XGBoost (Final Model)
Selected as the best-performing model based on evaluation metrics.

Hyperparameters:

- n_estimators: 300  
- max_depth: 5  
- learning_rate: 0.05  
- subsample: 0.8  
- colsample_bytree: 0.8  

---

## Model Evaluation

The models were evaluated using the following metrics:

- Accuracy
- ROC-AUC score
- Precision, recall, and F1-score
- Confusion matrix
- ROC curve comparison
- Cross-validation (5 folds)

---

## Key Findings

The analysis revealed the following insights:

- Higher comorbidity index is strongly associated with increased readmission risk.
- Patients admitted through ICU and emergency services have higher risk.
- Scheduled follow-up appointments reduce readmission probability.
- Longer hospital stays correlate with higher readmission rates.
- Survival analysis shows clear separation between high-risk and low-risk patients.

---

## Streamlit Web Application

An interactive web application was built using Streamlit to allow real-time predictions.

### Features

- Patient data input form
- Clinical measurement inputs
- Real-time prediction
- Probability of readmission output
- Risk classification (low, medium, high)

### Output

The application provides:

- Probability of 30-day readmission
- Binary prediction (readmission or not)
- Risk interpretation for decision support

---

## Project Structure

```text
Hospital-Readmission-ML-System/
│
├── data/
│   └── hospital_readmission_dataset.csv
│
├── analysis.py
├── train_model.py
├── app.py
│
├── models/
│   ├── model.pkl
│   └── model_columns.pkl
│
├── requirements.txt
└── README.md
