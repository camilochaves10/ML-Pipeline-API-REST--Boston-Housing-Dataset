from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/housing_model.joblib")


def load_model():
    return joblib.load(MODEL_PATH)


def predict(input_data: dict):
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return float(prediction)