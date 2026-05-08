import requests
import streamlit as st

import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Housing Price Prediction App")
st.write("Enter housing features and get a predicted MEDV value.")

with st.form("prediction_form"):
    CRIM = st.number_input("CRIM - Crime rate", value=0.00632)
    ZN = st.number_input("ZN - Residential land zoned", value=18.0)
    INDUS = st.number_input("INDUS - Non-retail business acres", value=2.31)
    CHAS = st.selectbox("CHAS - Bounds Charles River?", [0.0, 1.0])
    NOX = st.number_input("NOX - Nitric oxides concentration", value=0.538)
    RM = st.number_input("RM - Average rooms per dwelling", value=6.575)
    AGE = st.number_input("AGE - Older owner-occupied units", value=65.2)
    DIS = st.number_input("DIS - Distance to employment centers", value=4.09)
    RAD = st.number_input("RAD - Highway accessibility index", value=1.0)
    TAX = st.number_input("TAX - Property tax rate", value=296.0)
    PTRATIO = st.number_input("PTRATIO - Pupil-teacher ratio", value=15.3)
    B = st.number_input("B - Population index", value=396.9)
    LSTAT = st.number_input("LSTAT - Lower status population %", value=4.98)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "CRIM": CRIM,
        "ZN": ZN,
        "INDUS": INDUS,
        "CHAS": CHAS,
        "NOX": NOX,
        "RM": RM,
        "AGE": AGE,
        "DIS": DIS,
        "RAD": RAD,
        "TAX": TAX,
        "PTRATIO": PTRATIO,
        "B": B,
        "LSTAT": LSTAT,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload)

        if response.status_code == 200:
            prediction = response.json()["predicted_price"]
            st.success(f"Predicted housing value: ${prediction}K")
        else:
            st.error("Prediction failed. Check the FastAPI server.")

    except requests.exceptions.RequestException:
        st.error("Could not connect to API. Run `make api` first.")
        