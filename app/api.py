import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Housing Price Prediction API")

model = joblib.load("models/housing_model.joblib")


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


@app.post("/predict")
def predict(input_data: HousingInput):
    df = pd.DataFrame([input_data.model_dump()])
    prediction = model.predict(df)[0]

    return {"predicted_price": round(float(prediction), 2)}