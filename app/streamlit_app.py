import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

METRICS_PATH = Path("models/metrics.json")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.json")


st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)


st.title("🏠 Housing Price Prediction App")
st.write(
    "This app uses a trained regression model served by FastAPI. "
    "The project includes Docker, Docker Compose, CI/CD, model comparison, "
    "hyperparameter tuning, and feature importance."
)


def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def load_feature_importance():
    if FEATURE_IMPORTANCE_PATH.exists():
        with open(FEATURE_IMPORTANCE_PATH, "r") as f:
            return json.load(f)
    return []


tab_predict, tab_metrics, tab_importance, tab_about = st.tabs(
    ["Prediction", "Model Metrics", "Feature Importance", "About"]
)


with tab_predict:
    st.subheader("Make a Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        CRIM = st.number_input("CRIM", value=0.00632)
        ZN = st.number_input("ZN", value=18.0)
        INDUS = st.number_input("INDUS", value=2.31)
        CHAS = st.number_input("CHAS", value=0.0)
        NOX = st.number_input("NOX", value=0.538)

    with col2:
        RM = st.number_input("RM", value=6.575)
        AGE = st.number_input("AGE", value=65.2)
        DIS = st.number_input("DIS", value=4.09)
        RAD = st.number_input("RAD", value=1.0)

    with col3:
        TAX = st.number_input("TAX", value=296.0)
        PTRATIO = st.number_input("PTRATIO", value=15.3)
        B = st.number_input("B", value=396.9)
        LSTAT = st.number_input("LSTAT", value=4.98)

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

    if st.button("Predict"):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if response.status_code == 200:
                result = response.json()
                prediction = result["predicted_price"]
                unit = result.get("unit", "thousands_usd")
                st.success(f"Predicted housing value: {prediction:.2f} {unit}")


                
            else:
                st.error(f"Prediction failed: {response.text}")

        except requests.exceptions.RequestException:
            st.error("Could not connect to API. Check that FastAPI is running.")


with tab_metrics:
    st.subheader("Model Comparison and Selected Model")

    metrics = load_metrics()

    if metrics is None:
        st.warning("No metrics file found. Run the training pipeline first.")
    elif isinstance(metrics, list):
        st.error(
            "Old metrics.json format detected. "
            "Delete models/metrics.json and rerun the pipeline."
        )
        st.dataframe(pd.DataFrame(metrics))
        st.stop()
    else:

        st.write(f"Dataset used: `{metrics.get('dataset_used')}`")
        st.write(f"Rows used: `{metrics.get('rows')}`")

        baseline_results = metrics.get("baseline_results", [])
        selected_model = metrics.get("selected_model", {})

        if baseline_results:
            metrics_df = pd.DataFrame(baseline_results)
            st.dataframe(metrics_df, use_container_width=True)

            st.bar_chart(metrics_df.set_index("model")["mae"])

        if selected_model:
            st.subheader("Selected Tuned Model")
            col1, col2, col3 = st.columns(3)

            col1.metric("MAE", f"{selected_model.get('mae', 0):.4f}")
            col2.metric("RMSE", f"{selected_model.get('rmse', 0):.4f}")
            col3.metric("R²", f"{selected_model.get('r2', 0):.4f}")

            st.json(selected_model)


with tab_importance:
    st.subheader("Feature Importance")

    importance = load_feature_importance()

    if not importance:
        st.warning(
            "No feature importance file found. "
            "This may happen if the selected model does not expose feature importance."
        )
    else:
        importance_df = pd.DataFrame(importance)
        st.dataframe(importance_df, use_container_width=True)

        chart_df = importance_df.set_index("feature")
        st.bar_chart(chart_df["importance"])


with tab_about:
    st.subheader("Project Overview")

    st.markdown(
        """
        This project demonstrates an end-to-end ML application.

        It includes:

        - Data validation
        - Model training
        - Model comparison
        - Hyperparameter tuning
        - Feature importance
        - FastAPI model serving
        - Streamlit dashboard
        - Docker and Docker Compose
        - GitHub Actions CI/CD

        The training pipeline uses the full dataset when available.  
        If the full dataset is missing, it falls back to `data/raw/sample.csv`,
        which allows GitHub Actions to run successfully.
        """
    )