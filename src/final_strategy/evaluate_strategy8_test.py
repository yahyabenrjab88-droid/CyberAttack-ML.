import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data/processed/preprocessed/test.parquet"
MODEL_PATH = ROOT / "models/strategy8_web_attack_merged/hgb_13class_model.joblib"
FEATURES_PATH = ROOT / "models/strategy8_web_attack_merged/feature_columns.joblib"
CLASS_ORDER_PATH = ROOT / "models/strategy8_web_attack_merged/class_order.joblib"

OUTPUT_DIR = ROOT / "results/final_test_evaluation_threshold025"
THRESHOLD = 0.25

# Load the already-trained model and its saved configuration.
model = joblib.load(MODEL_PATH)
features = list(joblib.load(FEATURES_PATH))
class_order = [str(c) for c in joblib.load(CLASS_ORDER_PATH)]
model_classes = [str(c) for c in model.classes_]

if len(features) != 68 or len(set(features)) != 68:
    raise ValueError(f"Expected 68 unique features; got {len(features)}.")

if set(class_order) != set(model_classes):
    raise ValueError("Saved class order does not match the model classes.")

if not TEST_PATH.exists():
    raise FileNotFoundError(f"Test file not found: {TEST_PATH}")

# Read the test split only. Never write to this file.
test = pd.read_parquet(TEST_PATH)

required = set(features) | {"Label"}
missing = sorted(required - set(test.columns))
if missing:
    raise ValueError(f"Test dataset is missing required columns: {missing}")

if test[features].isna().any().any():
    raise ValueError("Test features contain missing values. Stopping safely.")

# Merge Web Attack labels in memory, without modifying the parquet.
merge_map = {
    "Web Attack - Brute Force": "Web Attack",
    "Web Attack - XSS": "Web Attack",
    "Web Attack - Sql Injection": "Web Attack",
}
y_true = test["Label"].replace(merge_map).astype(str).to_numpy()

unknown_labels = sorted(set(y_true) - set(model_classes))
if unknown_labels:
    raise ValueError(f"Test labels not represented by model: {unknown_labels}")

X_test = test[features]

print(f"Test rows: {len(test):,}")
print(f"Features: {len(features)}")
print("Predicting with the saved model; no training or tuning...")

probabilities = model.predict_proba(X_test)
predicted_indices = np.argmax(probabilities, axis=1)
y_pred = np.asarray(model_classes, dtype=object)[predicted_indices]

# Apply the previously selected threshold only to rows whose
# original model prediction is BENIGN or DoS Hulk.
benign_idx = model_classes.index("BENIGN")
hulk_idx = model_classes.index("DoS Hulk")

eligible = np.isin(y_pred, ["BENIGN", "DoS Hulk"])
choose_hulk = (
    probabilities[:, hulk_idx]
    >= THRESHOLD * probabilities[:, benign_idx]
)

y_pred[eligible] = np.where(
    choose_hulk[eligible], "DoS Hulk", "BENIGN"
)

# Compute final test metrics.
macro_f1 = f1_score(
    y_true, y_pred, labels=class_order,
    average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_true, y_pred, labels=class_order,
    average="weighted", zero_division=0
)
accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=class_order, zero_division=0
)

per_class = pd.DataFrame({
    "class": class_order,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "support": support,
})

cm = confusion_matrix(y_true, y_pred, labels=class_order)
confusion_df = pd.DataFrame(
    cm,
    index=[f"actual_{c}" for c in class_order],
    columns=[f"predicted_{c}" for c in class_order],
)

# Store outputs separately; do not overwrite previous experiments.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

per_class.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
confusion_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

report = classification_report(
    y_true, y_pred, labels=class_order,
    zero_division=0, digits=4
)
(OUTPUT_DIR / "classification_report.txt").write_text(
    report, encoding="utf-8"
)

pd.DataFrame({
    "actual": y_true,
    "predicted": y_pred,
}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

summary = {
    "evaluation": "final held-out test evaluation",
    "model": str(MODEL_PATH),
    "test_file": str(TEST_PATH),
    "test_rows": int(len(test)),
    "number_of_classes": len(class_order),
    "number_of_features": len(features),
    "threshold": THRESHOLD,
    "threshold_applies_only_to_original_benign_or_hulk_predictions": True,
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "class_order": class_order,
}

(OUTPUT_DIR / "summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Accuracy:    {accuracy:.4%}")
print(f"Macro-F1:    {macro_f1:.4%}")
print(f"Weighted-F1: {weighted_f1:.4%}")
print(f"\n{report}")
print(f"\nSaved evaluation files to: {OUTPUT_DIR.resolve()}")
print("Original test dataset was not modified.")
