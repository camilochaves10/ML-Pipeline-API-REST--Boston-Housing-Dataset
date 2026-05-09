from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.pipeline import FEATURE_COLUMNS, build_pipeline


FULL_DATA_PATH = Path("data/raw/HousingData.csv")
SAMPLE_DATA_PATH = Path("data/raw/sample.csv")

MODEL_PATH = Path("models/housing_model.joblib")
METRICS_PATH = Path("models/metrics.json")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.json")

def save_feature_importance(pipeline) -> None:
    model = pipeline.named_steps["regressor"]

    importances = model.feature_importances_

    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(FEATURE_COLUMNS, importances)
    ]

    rows = sorted(rows, key=lambda row: row["importance"], reverse=True)

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(rows, f, indent=4)

    print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")


def load_training_data() -> pd.DataFrame:
    if FULL_DATA_PATH.exists():
        print(f"Loading full dataset from {FULL_DATA_PATH}")
        return pd.read_csv(FULL_DATA_PATH)

    if SAMPLE_DATA_PATH.exists():
        print(f"Full dataset not found. Loading sample dataset from {SAMPLE_DATA_PATH}")
        return pd.read_csv(SAMPLE_DATA_PATH)

    raise FileNotFoundError(
        f"No dataset found. Expected either {FULL_DATA_PATH} or {SAMPLE_DATA_PATH}."
    )


def validate_training_data(df: pd.DataFrame) -> None:
    required_columns = set(FEATURE_COLUMNS + ["MEDV"])
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df["MEDV"].isnull().any():
        raise ValueError("Target column MEDV contains null values.")


def train() -> None:
    df = load_training_data()
    validate_training_data(df)

    X = df[FEATURE_COLUMNS]
    y = df["MEDV"]

    if len(df) < 10:
        print("Small sample dataset detected. Training and evaluating on the same data.")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    r2 = float(r2_score(y_test, preds))

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    save_feature_importance(pipeline)

    metrics = {
        "dataset_used": str(
            FULL_DATA_PATH if FULL_DATA_PATH.exists() else SAMPLE_DATA_PATH
        ),
        "rows": int(len(df)),
        "selected_model": {
            "model": "random_forest",
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        },
        "baseline_results": [
            {
                "model": "random_forest",
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        ],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")




if __name__ == "__main__":
    train()