import streamlit as st
import numpy as np
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Vehicle Recommendation System", layout="centered")

st.title("🚗 Vehicle Recommendation / Customer Segmentation App")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    try:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)

        return scaler, kmeans, None
    except Exception as e:
        return None, None, str(e)


scaler, kmeans, load_error = load_models()

# -----------------------------
# DEBUG INFO
# -----------------------------
st.subheader("📁 Debug Info")

import os
files = os.listdir()

st.write("Files in directory:")
st.code(files)

if load_error:
    st.error(f"Model loading failed: {load_error}")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.subheader("Enter Customer Details")

age = st.slider("Age", 18, 70, 25)
income = st.number_input("Annual Income", min_value=10000, max_value=200000, value=50000)
spending_score = st.slider("Spending Score", 1, 100, 50)

# -----------------------------
# CLUSTER → MEANING
# -----------------------------
def get_segment_label(cluster):
    labels = {
        0: "💸 High Income - High Spending (Premium Customers)",
        1: "💰 High Income - Low Spending (Careful Buyers)",
        2: "🧑‍💼 Average Customers",
        3: "🎯 Low Income - High Spending (Impulse Buyers)",
        4: "🪙 Low Income - Low Spending (Budget Customers)"
    }
    return labels.get(cluster, "Unknown Segment")


def get_vehicle_recommendation(cluster):
    recommendations = {
        0: ["BMW 3 Series", "Audi A4", "Mercedes C-Class"],
        1: ["Toyota Camry", "Honda City Hybrid", "Skoda Slavia"],
        2: ["Hyundai Creta", "Kia Seltos", "Maruti Grand Vitara"],
        3: ["Hyundai i20", "Tata Altroz", "Maruti Baleno"],
        4: ["Maruti Alto", "Renault Kwid", "Tata Tiago"]
    }
    return recommendations.get(cluster, ["No recommendation available"])


# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Segment"):

    if scaler is None or kmeans is None:
        st.error("Model not loaded properly.")
    else:
        try:
            # 🔥 FIXED dtype issue
            input_data = np.array([[age, income, spending_score]], dtype=np.float64)

            scaled = scaler.transform(input_data)
            cluster = kmeans.predict(scaled)[0]

            # -----------------------------
            # OUTPUT
            # -----------------------------
            st.success(f"Predicted Segment: {cluster}")
            st.info(get_segment_label(cluster))

            st.subheader("🚗 Recommended Vehicles")
            for car in get_vehicle_recommendation(cluster):
                st.write(f"• {car}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
