from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay, classification_report
from sklearn.ensemble import GradientBoostingClassifier

# features
FEATURES_NUMERIC = ["tenure_months", "monthly_charges", "total_charges", "num_support_tickets"]
FEATURES_CATEGORICAL = ["contract_type", "payment_method", "has_addon_streaming"]

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "model.joblib"

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y_train = train["churned"].astype(int)
    X_test = test[FEATURES_NUMERIC + FEATURES_CATEGORICAL]
    y_test = test["churned"].astype(int)
    return X_train, y_train, X_test, y_test

def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAL),
        ]
    )
    clf = GradientBoostingClassifier(random_state=42)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def main(evaluate: bool = True):
    MODEL_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    if evaluate:
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
        with open(REPORTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve — ChurnGuard")
        plt.savefig(REPORTS_DIR / "roc_curve.png", bbox_inches="tight")
        plt.close()
        print(f"Metrics written to {REPORTS_DIR}/metrics.json")

if __name__ == "__main__":
    main(evaluate=True)
