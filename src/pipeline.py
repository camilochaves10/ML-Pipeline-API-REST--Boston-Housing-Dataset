import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


LOG_TRANSFORM_COLS = ["CRIM", "ZN"]
MEDIAN_COLS = ["INDUS", "AGE", "LSTAT"]
MODE_COLS = ["CHAS"]

OTHER_NUMERIC_COLS = [
    "NOX",
    "RM",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
]

FEATURE_COLUMNS = (
    LOG_TRANSFORM_COLS
    + MEDIAN_COLS
    + MODE_COLS
    + OTHER_NUMERIC_COLS
)


def build_preprocessor() -> ColumnTransformer:
    log_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )

    median_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    mode_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    other_numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("log", log_transformer, LOG_TRANSFORM_COLS),
            ("median", median_transformer, MEDIAN_COLS),
            ("mode", mode_transformer, MODE_COLS),
            ("other_num", other_numeric_transformer, OTHER_NUMERIC_COLS),
        ],
        remainder="drop",
    )


def build_pipeline() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", model),
        ]
    )