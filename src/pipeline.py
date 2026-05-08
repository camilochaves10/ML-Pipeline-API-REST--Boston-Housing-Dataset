import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def build_pipeline() -> Pipeline:
    log_transform_cols = ["CRIM", "ZN"]
    median_cols = ["INDUS", "AGE", "LSTAT"]
    mode_cols = ["CHAS"]

    other_numeric_cols = [
        "NOX",
        "RM",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
    ]

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
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("log", log_transformer, log_transform_cols),
            ("median", median_transformer, median_cols),
            ("mode", mode_transformer, mode_cols),
            ("other_num", other_numeric_transformer, other_numeric_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )