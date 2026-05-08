import pandas as pd

from src.pipeline import build_pipeline


def test_pipeline_trains_with_missing_values():
    X = pd.DataFrame(
        {
            "CRIM": [0.1, None, 0.3],
            "ZN": [18.0, 0.0, None],
            "INDUS": [2.31, None, 7.07],
            "CHAS": [0.0, 1.0, None],
            "NOX": [0.538, 0.469, 0.469],
            "RM": [6.575, 6.421, 7.185],
            "AGE": [65.2, None, 61.1],
            "DIS": [4.09, 4.97, 4.96],
            "RAD": [1, 2, 2],
            "TAX": [296, 242, 242],
            "PTRATIO": [15.3, 17.8, 17.8],
            "B": [396.9, 396.9, 392.83],
            "LSTAT": [4.98, None, 4.03],
        }
    )

    y = [24.0, 21.6, 34.7]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    preds = pipeline.predict(X)

    assert len(preds) == len(y)