from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import joblib


MODEL_PATH = Path("models/housing_model.joblib")
PREDICTIONS_LOG_PATH = Path("logs/predictions.jsonl")
METRICS_PATH = Path("models/metrics.json")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.json")


app = FastAPI(title="Housing Price Prediction API")

model = joblib.load(MODEL_PATH)


class HousingInput(BaseModel):
    CRIM: float | None = None
    ZN: float | None = None
    INDUS: float | None = None
    CHAS: float | None = None
    NOX: float
    RM: float
    AGE: float | None = None
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float | None = None


def log_prediction(input_data: dict, prediction: float) -> None:
    PREDICTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": prediction,
    }

    with open(PREDICTIONS_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/")
def root():
    return {"message": "Housing Price Prediction API"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
def predict(input_data: HousingInput):
    try:
        input_dict = input_data.model_dump()

        df = pd.DataFrame([input_dict])

        prediction = float(model.predict(df)[0])

        log_prediction(input_dict, prediction)

        return {
            "predicted_price": round(prediction, 2),
            "unit": "thousands_usd",
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/model-info")
def model_info():
    metrics = None

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)

    return {
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "metrics_path": str(METRICS_PATH),
        "metrics_exists": METRICS_PATH.exists(),
        "feature_importance_exists": FEATURE_IMPORTANCE_PATH.exists(),
        "metrics": metrics,
    }