from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


LOG_PATH = Path("../logs/predictions.csv")


def log_prediction(input_data: dict, prediction: float) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction,
        **input_data,
    }

    df = pd.DataFrame([row])

    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)