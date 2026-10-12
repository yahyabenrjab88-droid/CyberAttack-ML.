from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "strategy8_web_attack_merged"
MODELS = ROOT / "models" / "strategy8_web_attack_merged"
VAL_PATH = ROOT / "data" / "processed" / "preprocessed" / "validation.parquet"

PRED_PATH = RESULTS / "validation_predictions.csv"
METRICS_PATH = RESULTS / "per_class_metrics.csv"
CM_PATH = RESULTS / "confusion_matrix.csv"
FEATURE_PATH = MODELS / "feature_columns.joblib"

WEB_ATTACK_LABELS = {
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
}

INDICATORS = [
    "Flow Bytes/s_missing",
    "Flow IAT Min_missing",
    "Fwd IAT Min_missing",
]


def merge_labels(labels):
    return labels.replace({
        label: "Web Attack" for label in WEB_ATTACK_LABELS
    })


def main():
    required = [PRED_PATH, METRICS_PATH, CM_PATH, FEATURE_PATH, VAL_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required Strategy 8 files are missing:\n" + "\n".join(missing)
        )

    print("=" * 72)
    print("STRATEGY 8A: VALIDATION ERROR ANALYSIS (READ ONLY)")
    print("=" * 72)

    # Check the saved feature schema.
    features = joblib.load(FEATURE_PATH)
    print(f"\nSaved model feature count: {len(features)}")
    print("Missing-value indicators:")
    for col in INDICATORS:
        print(f"  {col}: {'PRESENT' if col in features else 'MISSING'}")

    if len(features) != 68:
        raise ValueError(f"Expected 68 saved features, found {len(features)}")

    absent = [c for c in INDICATORS if c not in features]
    if absent:
        raise ValueError(f"Missing indicators in saved feature list: {absent}")

    # Read saved validation predictions and the validation targets only.
    pred_df = pd.read_csv(PRED_PATH)
    val_labels = pd.read_parquet(VAL_PATH, columns=["Label"])["Label"]
    y_true = merge_labels(val_labels).reset_index(drop=True)

    if len(pred_df) != len(y_true):
        raise ValueError("Prediction row count does not match validation data")

    saved_true = pred_df["true_label"].astype(str).reset_index(drop=True)
    if not saved_true.equals(y_true.astype(str)):
        raise ValueError(
            "Saved prediction true labels do not match the merged validation labels"
        )

    y_pred = pred_df["predicted_label"].astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    print(f"\nValidation rows checked: {len(pred_df):,}")
    print(f"Classes found: {len(labels)}")

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Recalculated macro-F1: {macro_f1:.4%}")

    # Count every true-versus-predicted pair.
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # The stored confusion matrix should match this independent calculation.
    stored_cm = pd.read_csv(CM_PATH, index_col=0)
    stored_cm.index = [str(x).removeprefix("true_") for x in stored_cm.index]
    stored_cm.columns = [str(x).removeprefix("pred_") for x in stored_cm.columns]
    stored_cm = stored_cm.reindex(index=labels, columns=labels, fill_value=0)

    print(
        "Saved confusion matrix matches predictions:",
        bool(cm_df.equals(stored_cm)),
    )

    # Rank the most common incorrect class predictions.
    errors = []
    for true_label in labels:
        for pred_label in labels:
            count = int(cm_df.loc[true_label, pred_label])
            if true_label != pred_label and count:
                errors.append((true_label, pred_label, count))

    errors.sort(key=lambda x: x[2], reverse=True)

    print("\nTOP 20 CLASS CONFUSIONS:")
    if not errors:
        print("No incorrect predictions found.")
    else:
        print(f"{'ACTUAL LABEL':32} {'PREDICTED LABEL':32} {'COUNT':>8}")
        for actual, predicted, count in errors[:20]:
            print(f"{actual[:32]:32} {predicted[:32]:32} {count:8,}")

    # Per-class false negatives and false positives.
    rows = []
    for label in labels:
        support = int((y_true == label).sum())
        predicted_count = int((y_pred == label).sum())
        correct = int(cm_df.loc[label, label])
        fn = support - correct
        fp = predicted_count - correct
        rows.append({
            "class": label,
            "support": support,
            "correct": correct,
            "false_negatives": fn,
            "false_positives": fp,
            "recall": correct / support if support else 0.0,
            "precision": correct / predicted_count if predicted_count else 0.0,
        })

    class_errors = pd.DataFrame(rows).sort_values(
        ["false_negatives", "false_positives"],
        ascending=False,
    )

    print("\nPER-CLASS ERROR COUNTS:")
    print(class_errors.to_string(
        index=False,
        formatters={
            "recall": "{:.4%}".format,
            "precision": "{:.4%}".format,
        },
    ))

    # Show saved per-class F1 for easy cross-checking.
    metrics = pd.read_csv(METRICS_PATH)
    print("\nSAVED PER-CLASS F1, LOWEST FIRST:")
    print(
        metrics.sort_values("f1")[["class", "support", "precision", "recall", "f1"]]
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE — NO MODEL TRAINING OR FILE MODIFICATION.")
    print("The test dataset was not loaded.")
    print("=" * 72)


if __name__ == "__main__":
    main()
