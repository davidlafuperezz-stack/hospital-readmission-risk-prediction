import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)

from xgboost import XGBClassifier

# CONFIG
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# PATHS
DATA_PATH = "data/hospital_readmission_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# LOAD DATA
df = pd.read_csv(DATA_PATH)

# FEATURES / TARGET
target = "readmitted_30_days"

X = df.drop(target, axis=1)
y = df[target]

# ONE-HOT ENCODING
X = pd.get_dummies(X, drop_first=False)

# SAVE TRAINING COLUMNS FOR APP
model_columns = X.columns.tolist()
joblib.dump(model_columns, os.path.join(MODEL_DIR, "model_columns.pkl"))

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# LOGISTIC REGRESSION
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)
log_prob = log_model.predict_proba(X_test_scaled)[:, 1]

print("\n====================")
print("LOGISTIC REGRESSION")
print("====================")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("ROC AUC:", roc_auc_score(y_test, log_prob))
print(classification_report(y_test, log_pred))

# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n====================")
print("RANDOM FOREST")
print("====================")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("ROC AUC:", roc_auc_score(y_test, rf_prob))
print(classification_report(y_test, rf_pred))

# XGBOOST
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\n====================")
print("XGBOOST")
print("====================")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("ROC AUC:", roc_auc_score(y_test, xgb_prob))
print(classification_report(y_test, xgb_pred))

# SAVE BEST MODEL
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "model.pkl"))

print("\nModel saved as models/model.pkl")
print("Model columns saved as models/model_columns.pkl")

# FEATURE IMPORTANCE
feat_df = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values(by="importance", ascending=False)

plt.figure()
sns.barplot(data=feat_df.head(15), x="importance", y="feature")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# ROC CURVE COMPARISON
plt.figure()

models = [
    ("Logistic Regression", log_prob),
    ("Random Forest", rf_prob),
    ("XGBoost", xgb_prob)
]

for name, prob in models:
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.2f}")

plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# CONFUSION MATRIX
plt.figure()
sns.heatmap(confusion_matrix(y_test, xgb_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# CROSS VALIDATION
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="roc_auc")

print("\n====================")
print("CROSS VALIDATION")
print("====================")
print("Mean ROC AUC:", cv_scores.mean())

# FINAL INSIGHTS
print("\nFINAL INSIGHTS:")
print("- XGBoost model trained successfully.")
print("- Model and training columns were saved for use in app.py.")
print("- Clinical and hospital variables are used to predict 30-day readmission risk.")
print("- This model can be connected to a Streamlit app for interactive predictions.")
