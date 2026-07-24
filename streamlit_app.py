"""
Farmer Guider AI App — Streamlit Web App (Bonus)
--------------------------------------------------
Loads the K-Means model, scaler, and recommendations trained in
Farmer_Guider_AI_App.ipynb and lets a user enter new soil/weather
readings to get an instant cluster prediction + farming recommendations.

Run with:
    streamlit run streamlit_app.py

Required files in the same folder:
    scaler.pkl, kmeans_model.pkl, recommendations.pkl, cluster_top_crops.pkl
(all produced automatically at the end of the notebook)
"""

import pickle
import pandas as pd
import streamlit as st

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


@st.cache_resource
def load_artifacts():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('recommendations.pkl', 'rb') as f:
        recommendations = pickle.load(f)
    with open('cluster_top_crops.pkl', 'rb') as f:
        top_crops = pickle.load(f)
    return scaler, kmeans, recommendations, top_crops


st.set_page_config(page_title="Farmer Guider AI App", page_icon="🌾", layout="centered")
st.title("🌾 Farmer Guider AI App")
st.write(
    "Enter your soil test and weather readings below. The app groups you with "
    "farmers facing similar conditions and gives tailored recommendations."
)

try:
    scaler, kmeans, recommendations, top_crops = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the notebook `Farmer_Guider_AI_App.ipynb` first — "
        "it saves scaler.pkl, kmeans_model.pkl, recommendations.pkl and cluster_top_crops.pkl "
        "into this same folder."
    )
    st.stop()

st.subheader("Soil Nutrients")
col1, col2, col3 = st.columns(3)
with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=50.0)
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=50.0)
with col3:
    K = st.number_input("Potassium (K)", min_value=0.0, max_value=210.0, value=50.0)

st.subheader("Weather & Soil Conditions")
col4, col5 = st.columns(2)
with col4:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
with col5:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=350.0, value=100.0)

if st.button("Get My Recommendations", type="primary"):
    new_data = pd.DataFrame([{
        'N': N, 'P': P, 'K': K,
        'temperature': temperature, 'humidity': humidity,
        'ph': ph, 'rainfall': rainfall
    }])
    new_scaled = scaler.transform(new_data[FEATURE_COLS])
    cluster_id = int(kmeans.predict(new_scaled)[0])
    rec = recommendations[cluster_id]

    st.success(f"You've been grouped into **Cluster {cluster_id}**")
    st.write(f"Farmers in this group commonly grow: **{', '.join(top_crops[cluster_id])}**")

    st.subheader("📋 Recommendations")
    st.markdown(f"**💧 Irrigation Advice**\n\n{rec['Irrigation Advice']}")
    st.markdown(f"**🌱 Fertilizer Suggestion**\n\n{rec['Fertilizer Suggestion']}")
    st.markdown(f"**🪱 Soil Management Tip**\n\n{rec['Soil Management Tip']}")

st.caption("Built for the Farmer Guider AI App capstone project — Unsupervised Learning (K-Means Clustering).")
