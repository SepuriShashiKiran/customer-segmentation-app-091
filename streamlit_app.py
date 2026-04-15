import streamlit as st
import numpy as np
import joblib
import pandas as pd
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Retail Customer Segmentation (RFM Model)")

# -----------------------------
# DEBUG: Check files
# -----------------------------
st.subheader("📂 Debug Info")
st.write(os.listdir())

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load("scaler.pkl")
        kmeans = joblib.load("kmeans.pkl")
        return scaler, kmeans
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None


scaler, kmeans = load_models()

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("Enter Customer RFM Values")

recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, value=5)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

# -----------------------------
# SEGMENT LABELS
# -----------------------------
def get_segment(cluster):
    segments = {
        0: "💎 High Value Customers",
        1: "🛍️ Regular Customers",
        2: "⚠️ At Risk Customers",
        3: "🆕 New Customers"
    }
    return segments.get(cluster, "Unknown")

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Segment"):

    if scaler is None or kmeans is None:
        st.error("Models not loaded properly.")
    else:
        try:
            # FIX dtype issue
            input_data = np.array([[recency, frequency, monetary]], dtype=np.float64)

            scaled = scaler.transform(input_data)
            cluster = kmeans.predict(scaled)[0]

            st.success(f"Cluster: {cluster}")
            st.info(get_segment(cluster))

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# OPTIONAL: DATA INSIGHTS
# -----------------------------
if os.path.exists("rfm_data.csv"):
    if st.checkbox("Show Cluster Insights"):
        df = pd.read_csv("rfm_data.csv")
        st.write(df.groupby("Cluster").mean())
