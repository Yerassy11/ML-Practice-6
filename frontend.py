"""Simple Streamlit interface for the Wine prediction API."""

import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Keep the same feature order as sklearn.datasets.load_wine().
FEATURE_DEFAULTS = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280/od315_of_diluted_wines": 3.92,
    "proline": 1065.0,
}

st.set_page_config(page_title="Wine Classifier", layout="centered")
st.title("Wine Classifier")
st.write("Enter the wine measurements and click Predict to get its class.")
st.caption("The fields contain an example. Classes 0, 1, and 2 are wine categories, not quality scores.")

with st.form("wine_features"):
    columns = st.columns(3)
    features = []
    for index, (name, default) in enumerate(FEATURE_DEFAULTS.items()):
        with columns[index % 3]:
            value = st.number_input(
                name.replace("_", " ").capitalize(),
                min_value=0.0,
                value=default,
                format="%.2f",
                key=name,
            )
            features.append(value)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(
                f"{API_URL}/predict", json={"features": features}, timeout=10
            )
            response.raise_for_status()
            prediction = response.json()["prediction"]
        st.success(f"Predicted wine class: {prediction}")
    except requests.RequestException:
        st.error("Prediction request failed. Check that the API is running and try again.")
    except (ValueError, KeyError):
        st.error("The prediction service returned an unexpected response.")
