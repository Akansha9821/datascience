import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.pipeline import make_pipeline


data = {
    "income": [
        25000, 30000, 35000, 40000, 45000,
        50000, 55000, 60000, 65000, 70000,
        28000, 32000, 42000, 52000, 62000,
        27000, 38000, 48000, 58000, 68000
    ],
    "debt_ratio": [
        0.85, 0.80, 0.75, 0.70, 0.65,
        0.40, 0.35, 0.30, 0.25, 0.20,
        0.90, 0.82, 0.68, 0.45, 0.28,
        0.88, 0.72, 0.50, 0.32, 0.22
    ],
    "default": [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 1, 0, 0,
        1, 1, 0, 0, 0
    ]
}

df = pd.DataFrame(data)
print(df.head())
print(df.head())
print(df.shape)
print(df["default"].value_counts())

X = df[["income", "debt_ratio"]]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred
results["Default_Probability"] = y_prob

print(results)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("ROC-AUC:", roc_auc_score(
    y_test,
    model.predict_proba(X_test)[:, 1]
))

# Predict New Customer
new_customer = pd.DataFrame({
    "income": [45000],
    "debt_ratio": [0.65]
})

prediction = model.predict(new_customer)[0]
probability = model.predict_proba(new_customer)[0, 1]

print("Predicted class:", prediction)
print("Default probability:", round(probability, 4))

lr = model.named_steps["logisticregression"]
scaler = model.named_steps["standardscaler"]

print("Intercept:", lr.intercept_)
print("Coefficients:", lr.coef_)

print("Features:", X.columns.tolist())
