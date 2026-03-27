import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Churn Dashboard", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### 🔍 Analyze customer behavior & predict churn")

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("customer_churn_20000.csv")

# -------------------------------
# Create Realistic Churn (with noise)
# -------------------------------
np.random.seed(42)

df["Churn_Flag"] = (
    ((df["Support_Interactions"] > 7) & (df["NPS_Score"] < 0)) |
    ((df["Contract_Type"] == "Monthly") & (df["Login_Frequency"] < 10))
).astype(int)

noise = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
df["Churn_Flag"] = df["Churn_Flag"] ^ noise

# -------------------------------
# SIDEBAR FILTERS 🔥
# -------------------------------
st.sidebar.header("🔎 Filter Data")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract_Type"].unique(),
    default=df["Contract_Type"].unique()
)

df = df[df["Contract_Type"].isin(contract_filter)]

# -------------------------------
# Dataset Preview
# -------------------------------
st.subheader("📂 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# Preprocessing
# -------------------------------
df = df.drop_duplicates()
df = df.drop("Customer_ID", axis=1)

X = df.drop("Churn_Flag", axis=1)
y = df["Churn_Flag"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------------
# Metrics (Top Cards 🔥)
# -------------------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{acc:.2f}")
col2.metric("Precision", f"{prec:.2f}")
col3.metric("Recall", f"{rec:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# -------------------------------
# ROW 1: BAR + PIE CHART 🔥
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Churn Distribution (Bar)")
    fig1, ax1 = plt.subplots()
    df["Churn_Flag"].value_counts().plot(kind='bar', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("🥧 Churn Distribution (Pie)")
    fig2, ax2 = plt.subplots()
    df["Churn_Flag"].value_counts().plot(
        kind='pie', autopct='%1.1f%%', ax=ax2
    )
    ax2.set_ylabel("")
    st.pyplot(fig2)

# -------------------------------
# ROW 2: HEATMAP + CONFUSION MATRIX
# -------------------------------
col3, col4 = st.columns(2)

with col3:
    st.subheader("🔥 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,5))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

with col4:
    st.subheader("📉 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax4)
    ax4.set_xlabel("Predicted")
    ax4.set_ylabel("Actual")
    st.pyplot(fig4)

# -------------------------------
# FEATURE IMPORTANCE
# -------------------------------
st.subheader("⭐ Feature Importance")

importances = model.feature_importances_
features = X.columns

feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

fig5, ax5 = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax5)
st.pyplot(fig5)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("✅ Built with Streamlit | ML Model: Random Forest")
