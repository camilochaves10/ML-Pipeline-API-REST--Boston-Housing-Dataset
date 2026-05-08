import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from src.monitoring import log_prediction


MODEL_PATH = "models/housing_model.joblib"

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


@app.get("/")
def root():
    return {"message": "Housing ML API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/version")
def model_version():
    return {"model_file": MODEL_PATH}


@app.post("/predict")
def predict(input_data: HousingInput):
    input_dict = input_data.model_dump()

    df = pd.DataFrame([input_dict])
    prediction = float(model.predict(df)[0])

    log_prediction(input_dict, prediction)

    return {
        "predicted_price": round(prediction, 2),
        "unit": "thousands_usd",
    }