import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

SEED = 42

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "processed" / "preprocessed" / "train.parquet"
VAL_PATH = ROOT / "data" / "processed" / "preprocessed" / "validation.parquet"

RESULTS_DIR = ROOT / "results" / "strategy8_web_attack_merged"
MODELS_DIR = ROOT / "models" / "strategy8_web_attack_merged"

WEB_ATTACK_LABELS = {
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
}

EXPECTED_INDICATORS = [
    "Flow Bytes/s_missing",
    "Flow IAT Min_missing",
    "Fwd IAT Min_missing",
]


def merge_labels(labels):
    """Merge the three Web Attack subclasses into one target label."""
    return labels.replace({
        label: "Web Attack" for label in WEB_ATTACK_LABELS
    })


def load_dataset(path):
    df = pd.read_parquet(path)

    if "Label" not in df.columns or "source_file" not in df.columns:
        raise ValueError(f"Required Label/source_file column missing: {path}")

    for col in EXPECTED_INDICATORS:
        if col not in df.columns:
            raise ValueError(f"Missing indicator feature {col!r} in {path}")

    feature_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in {"Label", "source_file"}
    ]

    if len(feature_columns) != 68:
        raise ValueError(
            f"Expected 68 numeric features, found {len(feature_columns)}"
        )

    X = df[feature_columns]

    if X.isna().any().any():
        raise ValueError(f"NaN values found in {path}")
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"Non-finite feature values found in {path}")

    y = merge_labels(df["Label"])

    return X, y, feature_columns


def save_distribution(y, path):
    y.value_counts().rename_axis("Label").to_csv(path, header=["count"])


def main():
    for path in (TRAIN_PATH, VAL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

    print("=" * 65)
    print("STRATEGY 8: MERGED WEB ATTACK CLASS")
    print("=" * 65)

    X_train, y_train, feature_columns = load_dataset(TRAIN_PATH)
    X_val, y_val, val_features = load_dataset(VAL_PATH)

    if feature_columns != val_features:
        raise ValueError("Training and validation feature columns differ")

    # Verify the expected original labels before merging.
    original_labels = set(
        pd.read_parquet(TRAIN_PATH, columns=["Label"])["Label"].unique()
    )
    if not WEB_ATTACK_LABELS.issubset(original_labels):
        raise ValueError("One or more expected Web Attack labels are missing")

    expected_train_web = (
        sum(int((pd.read_parquet(
            TRAIN_PATH, columns=["Label"]
        )["Label"] == label).sum()) for label in WEB_ATTACK_LABELS)
    )
    expected_val_web = (
        sum(int((pd.read_parquet(
            VAL_PATH, columns=["Label"]
        )["Label"] == label).sum()) for label in WEB_ATTACK_LABELS)
    )

    if int((y_train == "Web Attack").sum()) != expected_train_web:
        raise ValueError("Training Web Attack merge count does not match")
    if int((y_val == "Web Attack").sum()) != expected_val_web:
        raise ValueError("Validation Web Attack merge count does not match")

    train_counts = y_train.value_counts().sort_index()
    val_counts = y_val.value_counts().sort_index()

    if len(train_counts) != 13 or len(val_counts) != 13:
        raise ValueError("Expected exactly 13 classes after merging")
    if set(y_train.unique()) != set(y_val.unique()):
        raise ValueError("Training and validation class sets differ")

    print(f"\nTraining shape:   {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Features:         {len(feature_columns)}")
    print(f"Training classes: {len(train_counts)}")
    print(f"Validation classes: {len(val_counts)}")

    print("\nMerged training class counts:")
    print(train_counts.to_string())
    print("\nMerged validation class counts:")
    print(val_counts.to_string())

    # Do not overwrite any existing experiment output.
    for directory in (RESULTS_DIR, MODELS_DIR):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"{directory} is not empty. "
                "Stopping to protect existing experiment artifacts."
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    save_distribution(
        y_train, RESULTS_DIR / "training_class_distribution.csv"
    )
    save_distribution(
        y_val, RESULTS_DIR / "validation_class_distribution.csv"
    )

    params = dict(
        learning_rate=0.10,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        random_state=SEED,
        class_weight="balanced",
        verbose=1,
    )

    model = HistGradientBoostingClassifier(**params)

    print("\nTraining the 13-class HGB model...")
    model.fit(X_train, y_train)

    print("\nEvaluating on validation data...")
    predictions = model.predict(X_val)

    labels = list(model.classes_)
    macro_f1 = f1_score(y_val, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_val, predictions, average="weighted", zero_division=0
    )
    accuracy = accuracy_score(y_val, predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, predictions, labels=labels, zero_division=0
    )

    per_class = pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })
    per_class.to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_val, predictions, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"true_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    pd.DataFrame({
        "true_label": y_val.to_numpy(),
        "predicted_label": predictions,
    }).to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)

    summary = {
        "strategy": "strategy8_web_attack_merged",
        "task": "13-class multiclass classification",
        "merged_labels": sorted(WEB_ATTACK_LABELS),
        "merged_label_name": "Web Attack",
        "n_features": len(feature_columns),
        "n_classes": len(labels),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "random_seed": SEED,
        "model_parameters": params,
        "feature_columns": feature_columns,
        "class_order": labels,
    }

    joblib.dump(model, MODELS_DIR / "hgb_13class_model.joblib")
    joblib.dump(feature_columns, MODELS_DIR / "feature_columns.joblib")
    joblib.dump(labels, MODELS_DIR / "class_order.joblib")

    with open(RESULTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("STRATEGY 8 RESULTS")
    print("=" * 65)
    print(f"Classes:     {len(labels)}")
    print(f"Macro-F1:    {macro_f1:.4%}")
    print(f"Weighted-F1: {weighted_f1:.4%}")
    print(f"Accuracy:    {accuracy:.4%}")

    print("\nPer-class metrics:")
    print(per_class.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    print("\nClassification report:")
    print(classification_report(
        y_val, predictions, labels=labels, zero_division=0
    ))

    print(f"\nResults saved to: {RESULTS_DIR.resolve()}")
    print(f"Model saved to:   {MODELS_DIR.resolve()}")
    print("\nTest data was not loaded or evaluated.")


if __name__ == "__main__":
    main()
