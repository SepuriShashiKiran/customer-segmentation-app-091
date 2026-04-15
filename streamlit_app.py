import streamlit as st
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🚗 Vehicle Recommendation / Customer Segmentation App")

# -------------------------------
# 🔍 Debug: Check available files
# -------------------------------
st.subheader("📂 Debug Info")
files = os.listdir()
st.write("Files in directory:", files)

# -------------------------------
# 📦 Load Models Safely
# -------------------------------
@st.cache_resource
def load_all_models():
    try:
        scaler = joblib.load("scaler.pkl")
    except Exception as e:
        st.error(f"Scaler loading failed: {e}")
        scaler = None

    try:
        kmeans = joblib.load("kmeans.pkl")
    except Exception as e:
        st.error(f"KMeans loading failed: {e}")
        kmeans = None

    try:
        encoder = load_model("encoder.keras")
    except Exception as e:
        st.error(f"Encoder loading failed: {e}")
        encoder = None

    return scaler, kmeans, encoder


scaler, kmeans, encoder = load_all_models()

# -------------------------------
# 🧾 User Inputs
# -------------------------------
st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 70, 25)
income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
spending_score = st.slider("Spending Score", 1, 100, 50)

input_data = np.array([[age, income, spending_score]])

# -------------------------------
# 🚀 Prediction Logic
# -------------------------------
if st.button("Predict Segment"):

    if scaler is None or kmeans is None:
        st.error("Model files not loaded properly. Check logs above.")
    else:
        try:
            # Scale input
            scaled = scaler.transform(input_data)

            # Optional Autoencoder
            if encoder is not None:
                scaled = encoder.predict(scaled)

            # Predict cluster
            cluster = kmeans.predict(scaled)

            st.success(f"🎯 Predicted Cluster: {cluster[0]}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
