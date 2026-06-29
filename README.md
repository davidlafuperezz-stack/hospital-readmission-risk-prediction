 hospital-readmission-risk-prediction
Healthcare AI project for predicting hospital readmission risk within 30 days using synthetic patient data. Combines clinical, laboratory, and hospitalization features with machine learning models to support risk stratification and healthcare decision-making.
 Hospital Readmission Risk Prediction System

## Overview

This project uses machine learning to predict whether a patient will be readmitted within 30 days after hospital discharge.

## Objective

To identify high-risk patients using clinical and hospital data.

## Dataset

The dataset includes patient demographics, medical conditions, and hospital stay information.

Target variable: readmitted_30_days

## Workflow

1. Exploratory Data Analysis
2. Feature Engineering
3. Model Training
4. Model Evaluation
5. Deployment with Streamlit

## Models

- Logistic Regression
- Random Forest
- XGBoost (final model)

## How to Run

pip install -r requirements.txt
python train_model.py
streamlit run app.py

## Results

The XGBoost model achieved the best performance and was selected as the final model.

## Disclaimer

This project is for educational purposes only.
