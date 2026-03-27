import streamlit as st
import pandas as pd
import numpy as np

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Title
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("customer_churn_20000.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# -------------------------------
# Create Churn Pattern
# -------------------------------
df["Churn_Flag"] = (
    (df["Support_Interactions"] > 7) &
    (df["NPS_Score"] < 0) &
    (df["Contract_Type"] == "Monthly") &
    (df["Login_Frequency"] < 10)
).astype(int)

# -------------------------------
# Preprocessing
# -------------------------------
df = df.drop_duplicates()
df = df.drop("Customer_ID", axis=1)

X = df.drop("Churn_Flag", axis=1)
y = df["Churn_Flag"]

X = pd.get_dummies(X, drop_first=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Model
# -------------------------------
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------------------
# Metrics
# -------------------------------
st.subheader("📈 Model Performance")

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.write(f"Accuracy: {acc:.2f}")
st.write(f"Precision: {prec:.2f}")
st.write(f"Recall: {rec:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Churn Distribution")
st.bar_chart(df["Churn_Flag"].value_counts())
