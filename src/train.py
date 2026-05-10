from pathlib import Path
import json
import os

import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from src.pipeline import FEATURE_COLUMNS, build_preprocessor


FULL_DATA_PATH = Path("data/raw/HousingData.csv")
SAMPLE_DATA_PATH = Path("data/raw/sample.csv")

MODEL_PATH = Path("models/housing_model.joblib")
METRICS_PATH = Path("models/metrics.json")
FEATURE_IMPORTANCE_PATH = Path("models/feature_importance.json")


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlflow_data")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("housing-price-regression")
    print(f"MLflow tracking URI: {tracking_uri}")

def tune_model(model_name: str, pipeline: Pipeline, X_train, y_train, sample_mode: bool):
    if sample_mode:
        print(f"Sample mode detected. Skipping tuning for {model_name}.")
        return pipeline, {}

    search_spaces = {
        "random_forest": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [None, 5, 10, 20],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        "gradient_boosting": {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1],
            "regressor__max_depth": [2, 3, 5],
        },
    }

    if model_name not in search_spaces:
        print(f"No tuning configured for {model_name}.")
        return pipeline, {}

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_spaces[model_name],
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_


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


def build_model_candidates() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            random_state=42,
        ),
    }


def build_model_pipeline(regressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)

    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
    }


def save_feature_importance(pipeline: Pipeline) -> None:
    regressor = pipeline.named_steps["regressor"]

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = abs(regressor.coef_)
    else:
        print("Model does not expose feature importances or coefficients.")
        return

    rows = [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in zip(FEATURE_COLUMNS, importances)
    ]

    rows = sorted(rows, key=lambda row: row["importance"], reverse=True)

    with open(FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(rows, f, indent=4)

    print(f"Feature importance saved to {FEATURE_IMPORTANCE_PATH}")


def train() -> None:
    setup_mlflow()

    df = load_training_data()
    validate_training_data(df)

    X = df[FEATURE_COLUMNS]
    y = df["MEDV"]

    if len(df) < 10:
        print("Small sample dataset detected. Training and evaluating on same data.")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

    dataset_used = str(FULL_DATA_PATH if FULL_DATA_PATH.exists() else SAMPLE_DATA_PATH)
    candidates = build_model_candidates()

    results = []
    best_pipeline = None
    best_result = None

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for model_name, regressor in candidates.items():
        print(f"Training {model_name}...")

        pipeline = build_model_pipeline(regressor)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        result = {
            "model": model_name,
            **metrics,
        }

        results.append(result)

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset_used", dataset_used)
            mlflow.log_param("rows", int(len(df)))

            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("r2", metrics["r2"])

        if best_result is None or metrics["mae"] < best_result["mae"]:
            best_result = result
            best_pipeline = pipeline

    results = sorted(results, key=lambda row: row["mae"])

    best_model_name = best_result["model"]

    print(f"Tuning best model: {best_model_name}")

    tuned_pipeline, best_params = tune_model(
        best_model_name,
        best_pipeline,
        X_train,
        y_train,
        sample_mode=len(df) < 10,
    )

    tuned_metrics = evaluate_model(tuned_pipeline, X_test, y_test)

    if tuned_metrics["mae"] <= best_result["mae"]:
        print("Tuned model improved or matched baseline. Using tuned model.")
        best_pipeline = tuned_pipeline
        best_result = {
            "model": f"{best_model_name}_tuned",
            **tuned_metrics,
            "best_params": best_params,
        }
    else:
        print("Tuned model did not improve baseline. Keeping baseline model.")

    joblib.dump(best_pipeline, MODEL_PATH)
    save_feature_importance(best_pipeline)

    metrics_output = {
        "dataset_used": dataset_used,
        "rows": int(len(df)),
        "selected_model": best_result,
        "baseline_results": results,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_output, f, indent=4)

    with mlflow.start_run(run_name=f"best_model_{best_result['model']}"):
        mlflow.log_param("selected_model", best_result["model"])
        mlflow.log_param("dataset_used", dataset_used)
        mlflow.log_param("rows", int(len(df)))

        mlflow.log_metric("mae", best_result["mae"])
        mlflow.log_metric("rmse", best_result["rmse"])
        mlflow.log_metric("r2", best_result["r2"])

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

        if FEATURE_IMPORTANCE_PATH.exists():
            mlflow.log_artifact(str(FEATURE_IMPORTANCE_PATH))

    print(f"Best model: {best_result['model']}")
    print(f"MAE:  {best_result['mae']:.4f}")
    print(f"RMSE: {best_result['rmse']:.4f}")
    print(f"R2:   {best_result['r2']:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train()