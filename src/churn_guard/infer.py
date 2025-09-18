from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path("models/model.joblib")

class Predictor:
    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.pipe = joblib.load(model_path)

    def predict_one(self, payload: dict) -> dict:
        df = pd.DataFrame([payload])
        prob = float(self.pipe.predict_proba(df)[0, 1])
        return {"churn_probability": prob, "churn_label": int(prob >= 0.5)}
