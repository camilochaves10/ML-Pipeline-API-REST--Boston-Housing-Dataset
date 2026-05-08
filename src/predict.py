from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/housing_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run `make train` before serving predictions."
        )

    return joblib.load(MODEL_PATH)


def predict(input_data: dict) -> float:
    model = load_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    return float(prediction)