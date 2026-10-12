
"""
train_hgb_classweight_balanced.py

Experiment: hgb_classweight_balanced

Hypothesis
----------
The baseline HistGradientBoostingClassifier (macro F1 = 45.57%, accuracy = 93.16%)
underperforms on minority classes primarily because the loss function is dominated
by the large classes.

Adding balanced class weighting should improve macro F1 and minority-class
performance without changing the dataset, features, split, preprocessing,
or model hyperparameters.

ONLY CHANGE VS BASELINE
-----------------------
    class_weight="balanced"

FIXED VS BASELINE
-----------------
    learning_rate=0.10
    max_iter=300
    max_leaf_nodes=31
    min_samples_leaf=20
    l2_regularization=0.0
    early_stopping=True
    validation_fraction=0.10
    n_iter_no_change=20
    random_state=42

    Same locked train/validation data
    Same 68 model features
    Same preprocessing
    Same external validation set

TEST SET
--------
The test set is NOT loaded or evaluated.

Primary metric:
    Macro F1

Secondary metrics:
    Accuracy
    Weighted F1
    Per-class precision / recall / F1
    Confusion matrix

Usage
-----
(.venv) PS C:\\CyberAttack-ML> py src\\train_hgb_classweight_balanced.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================================
# Configuration
# ============================================================================

EXPERIMENT_NAME = "hgb_classweight_balanced"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODELS_DIR = PROJECT_ROOT / "models" / "historical" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results"

LABEL_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"

NON_FEATURE_COLUMNS = {
    LABEL_COLUMN,
    PROVENANCE_COLUMN,
}

EXPECTED_N_FEATURES = 68
RANDOM_STATE = 42


# ============================================================================
# IMPORTANT:
# This must match the baseline exactly except for class_weight.
# ============================================================================

MODEL_PARAMS = {
    "learning_rate": 0.10,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": True,
    "validation_fraction": 0.10,
    "n_iter_no_change": 20,
    "random_state": RANDOM_STATE,

    # THE ONLY EXPERIMENTAL CHANGE
    "class_weight": "balanced",

    "verbose": 1,
}


# ============================================================================
# Helpers
# ============================================================================

def log(message: str) -> None:
    """Print a timestamped message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_parquet(path: Path, name: str) -> pd.DataFrame:
    """Load one locked preprocessed split."""
    if not path.exists():
        raise FileNotFoundError(
            f"{name} parquet not found:\n"
            f"  {path}\n\n"
            f"Expected the locked output from src/preprocess.py."
        )

    log(f"Loading {name}: {path}")

    df = pd.read_parquet(path)

    log(
        f"  {name}: "
        f"{df.shape[0]:,} rows Ã— {df.shape[1]} columns"
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model-input columns, excluding target and provenance."""
    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]


def verify_schema(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[str]:
    """
    Verify that train and validation contain exactly the expected
    model-input schema.
    """

    # Required target
    for name, df in (
        ("train", train_df),
        ("validation", val_df),
    ):
        if LABEL_COLUMN not in df.columns:
            raise ValueError(
                f"{name} set is missing required target column "
                f"'{LABEL_COLUMN}'."
            )

    train_features = get_feature_columns(train_df)
    val_features = get_feature_columns(val_df)

    # Exact feature ordering must match.
    if train_features != val_features:
        only_train = sorted(
            set(train_features) - set(val_features)
        )

        only_validation = sorted(
            set(val_features) - set(train_features)
        )

        raise ValueError(
            "Train/validation feature mismatch.\n"
            f"Only in train: {only_train}\n"
            f"Only in validation: {only_validation}"
        )

    # Exactly 68 model features.
    if len(train_features) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected exactly {EXPECTED_N_FEATURES} model features, "
            f"but found {len(train_features)}."
        )

    # All model features must be numeric.
    non_numeric = [
        column
        for column in train_features
        if not pd.api.types.is_numeric_dtype(
            train_df[column]
        )
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features detected: {non_numeric}"
        )

    # Validation numeric check too.
    validation_non_numeric = [
        column
        for column in val_features
        if not pd.api.types.is_numeric_dtype(
            val_df[column]
        )
    ]

    if validation_non_numeric:
        raise ValueError(
            "Non-numeric validation features detected: "
            f"{validation_non_numeric}"
        )

    log(
        f"Schema check passed: {len(train_features)} model features."
    )

    log(
        "  Train/validation feature order is identical."
    )

    log(
        "  All model features are numeric."
    )

    return train_features


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 72)
    log(f"Experiment: {EXPERIMENT_NAME}")
    log("=" * 72)

    log(
        f"Python: {sys.version.split()[0]}"
    )

    log(
        f"Platform: {platform.platform()}"
    )

    # ----------------------------------------------------------------------
    # Load TRAIN and VALIDATION only.
    #
    # TEST IS NEVER LOADED.
    # ----------------------------------------------------------------------

    train_df = load_parquet(
        TRAIN_PATH,
        "TRAIN",
    )

    val_df = load_parquet(
        VAL_PATH,
        "VALIDATION",
    )

    # ----------------------------------------------------------------------
    # Verify locked schema.
    # ----------------------------------------------------------------------

    feature_cols = verify_schema(
        train_df,
        val_df,
    )

    log(
        f"Model input features: {len(feature_cols)}"
    )

    # ----------------------------------------------------------------------
    # Extract model inputs.
    #
    # We intentionally do NOT perform additional preprocessing here.
    # The preprocessed parquet files are already locked.
    # ----------------------------------------------------------------------

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COLUMN].astype(str)

    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COLUMN].astype(str)

    # Check for unexpected missing/infinite values.
    # preprocess.py already guarantees these should not exist.
    if X_train.isna().any().any():
        raise ValueError(
            "NaN values detected in X_train. "
            "This violates the locked preprocessing contract."
        )

    if X_val.isna().any().any():
        raise ValueError(
            "NaN values detected in X_val. "
            "This violates the locked preprocessing contract."
        )

    if not X_train.map(lambda x: pd.notna(x)).all().all():
        raise ValueError(
            "Invalid values detected in X_train."
        )

    # ----------------------------------------------------------------------
    # Verify classes.
    # ----------------------------------------------------------------------

    train_classes = sorted(
        y_train.unique().tolist()
    )

    val_classes = sorted(
        y_val.unique().tolist()
    )

    if train_classes != val_classes:
        raise ValueError(
            "Train/validation class mismatch.\n"
            f"Train classes: {train_classes}\n"
            f"Validation classes: {val_classes}"
        )

    class_names = train_classes

    log(
        f"Classes ({len(class_names)}): {class_names}"
    )

    log(
        f"TRAIN matrix: "
        f"{X_train.shape[0]:,} Ã— {X_train.shape[1]}"
    )

    log(
        f"VALIDATION matrix: "
        f"{X_val.shape[0]:,} Ã— {X_val.shape[1]}"
    )

    # ----------------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------------

    log("")
    log(
        "Training HistGradientBoostingClassifier..."
    )

    log(
        "Experimental change: class_weight='balanced'"
    )

    log(
        "All other model parameters remain identical to baseline."
    )

    model = HistGradientBoostingClassifier(
        **MODEL_PARAMS
    )

    train_start = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    train_time_sec = (
        time.perf_counter() - train_start
    )

    n_iterations = int(
        getattr(
            model,
            "n_iter_",
            -1,
        )
    )

    log("")
    log(
        f"Training complete: "
        f"{train_time_sec:.2f} seconds"
    )

    log(
        f"Boosting iterations used: "
        f"{n_iterations}"
    )

    # ----------------------------------------------------------------------
    # Validation prediction
    #
    # TEST IS NOT TOUCHED.
    # ----------------------------------------------------------------------

    predict_start = time.perf_counter()

    y_pred = model.predict(
        X_val
    )

    predict_time_sec = (
        time.perf_counter() - predict_start
    )

    log(
        f"Validation prediction complete: "
        f"{predict_time_sec:.2f} seconds"
    )

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    accuracy = float(
        accuracy_score(
            y_val,
            y_pred,
        )
    )

    macro_f1 = float(
        f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_val,
            y_pred,
            labels=class_names,
            zero_division=0,
        )
    )

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    per_class_df = per_class_df.sort_values(
        "support",
        ascending=False,
    )

    # Full sklearn classification report.
    report_dict = classification_report(
        y_val,
        y_pred,
        labels=class_names,
        zero_division=0,
        output_dict=True,
    )

    report_df = (
        pd.DataFrame(report_dict)
        .transpose()
    )

    # Confusion matrix.
    cm = confusion_matrix(
        y_val,
        y_pred,
        labels=class_names,
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    # ----------------------------------------------------------------------
    # Print results
    # ----------------------------------------------------------------------

    log("")
    log("=" * 72)
    log("VALIDATION RESULTS")
    log("=" * 72)

    log(
        f"Accuracy:    {accuracy:.4%}"
    )

    log(
        f"Macro F1:    {macro_f1:.4%}"
    )

    log(
        f"Weighted F1: {weighted_f1:.4%}"
    )

    log("")
    log("Per-class results:")

    print(
        per_class_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ----------------------------------------------------------------------
    # Save model
    # ----------------------------------------------------------------------

    model_path = (
        MODELS_DIR
        / f"{EXPERIMENT_NAME}.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    log(
        f"Saved model -> {model_path}"
    )

    # ----------------------------------------------------------------------
    # Save headline metrics
    # ----------------------------------------------------------------------

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_train_rows": int(len(y_train)),
        "n_validation_rows": int(len(y_val)),
        "n_features": int(len(feature_cols)),
        "n_classes": int(len(class_names)),
    }

    metrics_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(
        f"Saved metrics -> {metrics_path}"
    )

    # ----------------------------------------------------------------------
    # Save classification report
    # ----------------------------------------------------------------------

    report_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_classification_report.csv"
    )

    report_df.to_csv(
        report_path
    )

    log(
        f"Saved classification report -> {report_path}"
    )

    # ----------------------------------------------------------------------
    # Save confusion matrix
    # ----------------------------------------------------------------------

    cm_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_path
    )

    log(
        f"Saved confusion matrix -> {cm_path}"
    )

    # ----------------------------------------------------------------------
    # Save training summary
    # ----------------------------------------------------------------------

    summary = {
        "experiment": EXPERIMENT_NAME,

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "hypothesis": (
            "Balanced class weighting alone improves minority-class "
            "recognition by giving minority classes greater influence "
            "during optimization."
        ),

        "changed_vs_baseline": [
            "class_weight=balanced"
        ],

        "fixed_vs_baseline": {
            "learning_rate": 0.10,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
            "early_stopping": True,
            "validation_fraction": 0.10,
            "n_iter_no_change": 20,
            "random_state": RANDOM_STATE,
            "n_features": len(feature_cols),
        },

        "model_params": {
            key: str(value)
            for key, value in MODEL_PARAMS.items()
        },

        "n_boosting_iterations": n_iterations,

        "data": {
            "train_path": str(TRAIN_PATH),
            "validation_path": str(VAL_PATH),
            "test_path_used": None,
            "n_train_rows": int(len(y_train)),
            "n_validation_rows": int(len(y_val)),
            "n_features": int(len(feature_cols)),
            "n_classes": int(len(class_names)),
            "classes": class_names,
        },

        "timing": {
            "train_time_sec": float(train_time_sec),
            "predict_time_sec": float(predict_time_sec),
        },

        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },

        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },

        "test_set_used": False,
    }

    summary_path = (
        RESULTS_DIR
        / f"{EXPERIMENT_NAME}_training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(
        f"Saved training summary -> {summary_path}"
    )

    # ----------------------------------------------------------------------
    # Final message
    # ----------------------------------------------------------------------

    log("")
    log("=" * 72)
    log("EXPERIMENT COMPLETE")
    log("=" * 72)

    log(
        "The TEST set was NOT loaded, evaluated, or used."
    )

    log(
        f"Macro F1: {macro_f1:.4%}"
    )

    log(
        f"Accuracy: {accuracy:.4%}"
    )

    log(
        f"Weighted F1: {weighted_f1:.4%}"
    )


if __name__ == "__main__":
    main()
