from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "strategy8_web_attack_merged"
RESULTS_DIR = ROOT / "results" / "strategy8b_hulk_benign_thresholds"
VAL_PATH = ROOT / "data" / "processed" / "preprocessed" / "validation.parquet"
PRED_PATH = ROOT / "results" / "strategy8_web_attack_merged" / "validation_predictions.csv"

BENIGN = "BENIGN"
HULK = "DoS Hulk"

# Lower thresholds make it easier to predict DoS Hulk instead of BENIGN.
THRESHOLDS = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]


def merge_labels(labels):
    return labels.replace({
        "Web Attack - Brute Force": "Web Attack",
        "Web Attack - XSS": "Web Attack",
        "Web Attack - Sql Injection": "Web Attack",
    })


def main():
    model_path = MODEL_DIR / "hgb_13class_model.joblib"
    features_path = MODEL_DIR / "feature_columns.joblib"

    for path in [model_path, features_path, VAL_PATH, PRED_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    model = joblib.load(model_path)
    features = joblib.load(features_path)

    df = pd.read_parquet(VAL_PATH)
    X = df[features]
    y_true = merge_labels(df["Label"]).reset_index(drop=True)

    saved = pd.read_csv(PRED_PATH)
    if len(saved) != len(y_true):
        raise ValueError("Saved prediction count does not match validation data")

    # Ensure the loaded model reproduces Strategy 8's original predictions.
    baseline_pred = model.predict(X)
    if not np.array_equal(
        baseline_pred.astype(str),
        saved["predicted_label"].astype(str).to_numpy()
    ):
        raise ValueError(
            "Loaded model predictions differ from saved Strategy 8 predictions"
        )

    classes = list(model.classes_)
    if BENIGN not in classes or HULK not in classes:
        raise ValueError("BENIGN or DoS Hulk is missing from model classes")

    probabilities = model.predict_proba(X)
    p_benign = probabilities[:, classes.index(BENIGN)]
    p_hulk = probabilities[:, classes.index(HULK)]

    # Only rows originally classified as BENIGN or DoS Hulk are eligible
    # for a switch between these two classes. Other predictions are untouched.
    eligible = np.isin(baseline_pred, [BENIGN, HULK])

    rows = []
    predictions_by_threshold = {}

    for threshold in THRESHOLDS:
        pred = baseline_pred.copy()

        choose_hulk = eligible & (p_hulk >= threshold * p_benign)
        choose_benign = eligible & ~choose_hulk

        pred[choose_hulk] = HULK
        pred[choose_benign] = BENIGN

        hulk_true = y_true == HULK
        benign_true = y_true == BENIGN
        hulk_pred = pred == HULK
        benign_pred = pred == BENIGN

        hulk_tp = int((hulk_true & hulk_pred).sum())
        hulk_fn = int((hulk_true & ~hulk_pred).sum())
        hulk_fp = int((~hulk_true & hulk_pred).sum())

        benign_tp = int((benign_true & benign_pred).sum())
        benign_fn = int((benign_true & ~benign_pred).sum())
        benign_fp = int((~benign_true & benign_pred).sum())

        rows.append({
            "threshold": threshold,
            "macro_f1": f1_score(
                y_true, pred, average="macro", zero_division=0
            ),
            "accuracy": accuracy_score(y_true, pred),
            "hulk_f1": f1_score(
                hulk_true, hulk_pred, zero_division=0
            ),
            "benign_f1": f1_score(
                benign_true, benign_pred, zero_division=0
            ),
            "hulk_false_negatives": hulk_fn,
            "hulk_false_positives": hulk_fp,
            "benign_false_negatives": benign_fn,
            "benign_false_positives": benign_fp,
            "hulk_to_benign": int((hulk_true & benign_pred).sum()),
            "benign_to_hulk": int((benign_true & hulk_pred).sum()),
        })

        predictions_by_threshold[threshold] = pred

    results = pd.DataFrame(rows)

    # Save only into a new experiment folder; refuse to overwrite prior output.
    if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
        raise FileExistsError(
            f"{RESULTS_DIR} is not empty; refusing to overwrite results"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "threshold_comparison.csv", index=False)

    print("=" * 78)
    print("STRATEGY 8B: DOS HULK / BENIGN THRESHOLD SWEEP")
    print("=" * 78)
    print("No retraining performed.")
    print("All classes other than BENIGN and DoS Hulk remain unchanged.")
    print("\nResults (percentages):")

    display = results.copy()
    for col in ["macro_f1", "accuracy", "hulk_f1", "benign_f1"]:
        display[col] = display[col].map(lambda x: f"{x:.4%}")

    print(display.to_string(index=False))
    print(f"\nCSV saved to: {(RESULTS_DIR / 'threshold_comparison.csv').resolve()}")
    print("Test data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
