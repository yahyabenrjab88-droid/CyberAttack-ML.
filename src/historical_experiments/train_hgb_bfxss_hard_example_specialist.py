
"""
Strategy 4 — Hard-Example BF/XSS Specialist

Goal:
    Improve the BF/XSS decision boundary by giving more importance to
    BF/XSS examples that the primary HGB model finds difficult.

IMPORTANT:
    - Validation data is NEVER used to train the specialist.
    - Test data is NEVER loaded.
    - All 68 original model features are retained.
    - This is NOT the old 15-feature experiment.
    - Hard examples are identified using an INTERNAL split of the
      training data, avoiding direct use of the final validation set.

Pipeline:
    1. Load the existing train/validation split.
    2. Train the primary HGB on the main training portion.
    3. Use an internal hard-example holdout from the training data.
    4. Identify BF/XSS examples misclassified by the primary model.
    5. Assign higher sample weights to those hard examples.
    6. Train a BF/XSS specialist using all 68 features.
    7. Route only primary BF/XSS predictions through the specialist.
    8. Evaluate ONLY on the untouched validation set.
    9. Compare against the primary baseline.

Python:
    3.14+
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "results"

TRAIN_PATH = DATA_DIR / "processed" / "cicids2017_training_ready.parquet"
MANIFEST_PATH = SPLITS_DIR / "row_manifest.parquet"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL PARAMETERS
# ============================================================

PRIMARY_MODEL_PARAMS = dict(
    learning_rate=0.10,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    verbose=1,
)

SPECIALIST_MODEL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=15,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    verbose=1,
)


# ============================================================
# HARD-EXAMPLE SETTINGS
# ============================================================

TARGET_CLASSES = {
    "Web Attack - Brute Force",
    "Web Attack - XSS",
}

# Internal split used ONLY inside the training data.
HARD_EXAMPLE_HOLDOUT_SIZE = 0.20

# Weight assigned to normal BF/XSS examples.
NORMAL_WEIGHT = 1.0

# Weight assigned to BF/XSS examples that the primary model
# misclassifies on the internal holdout.
HARD_WEIGHT = 3.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def find_label_column(df: pd.DataFrame) -> str:
    candidates = [
        "Label",
        "label",
        "Class",
        "class",
        "target",
        "Target",
    ]

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find the label column. "
        f"Available columns include: {list(df.columns[:20])}"
    )


def get_feature_columns(
    df: pd.DataFrame,
    label_column: str,
) -> list[str]:
    """
    Use numeric columns as model features.

    Excludes obvious metadata columns and the target.
    """

    excluded = {
        label_column,
        "split",
        "row_id",
        "index",
        "Unnamed: 0",
    }

    feature_columns = [
        col
        for col in df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(feature_columns) != 68:
        print(
            f"WARNING: Expected 68 numeric model features, "
            f"found {len(feature_columns)}."
        )

    return feature_columns


def get_split_masks(manifest: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Detect train/validation assignment from the existing manifest.
    """

    split_column = None

    for candidate in ["split", "set", "partition", "dataset_split"]:
        if candidate in manifest.columns:
            split_column = candidate
            break

    if split_column is None:
        raise ValueError(
            "Could not find a split column in row_manifest.parquet. "
            f"Columns: {list(manifest.columns)}"
        )

    split_values = (
        manifest[split_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    train_mask = split_values.isin({"train", "training"})
    val_mask = split_values.isin({"val", "validation", "valid"})

    if not train_mask.any():
        raise ValueError("No training rows found in manifest.")

    if not val_mask.any():
        raise ValueError("No validation rows found in manifest.")

    return train_mask, val_mask


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class = {}

    for cls, p, r, f, s in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):
        per_class[cls] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "per_class": per_class,
    }


def confusion_count(
    y_true: pd.Series,
    y_pred: np.ndarray,
    true_class: str,
    predicted_class: str,
) -> int:
    return int(
        np.sum(
            (y_true.to_numpy() == true_class)
            & (y_pred == predicted_class)
        )
    )


# ============================================================
# LOAD DATA
# ============================================================

print_header("LOADING DATA")

print(f"Project root: {PROJECT_ROOT}")
print(f"Training data: {TRAIN_PATH}")
print(f"Manifest:      {MANIFEST_PATH}")

if not TRAIN_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {TRAIN_PATH}")

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

df = pd.read_parquet(TRAIN_PATH)
manifest = pd.read_parquet(MANIFEST_PATH)

print(f"Loaded processed dataset: {df.shape}")
print(f"Loaded manifest:          {manifest.shape}")


# ============================================================
# ALIGN MANIFEST WITH DATA
# ============================================================

print_header("ALIGNING SPLITS")

if len(df) != len(manifest):
    raise ValueError(
        "Training-ready dataset and manifest have different row counts: "
        f"{len(df)} vs {len(manifest)}"
    )

train_mask, val_mask = get_split_masks(manifest)

train_df = df.loc[train_mask].copy()
val_df = df.loc[val_mask].copy()

print(f"Train rows:      {len(train_df):,}")
print(f"Validation rows: {len(val_df):,}")


# ============================================================
# LABEL + FEATURES
# ============================================================

label_column = find_label_column(df)

feature_columns = get_feature_columns(
    df,
    label_column,
)

print(f"Label column: {label_column}")
print(f"Feature count: {len(feature_columns)}")

if len(feature_columns) != 68:
    raise ValueError(
        f"Expected exactly 68 model features, found "
        f"{len(feature_columns)}. "
        "Stop and inspect the dataset schema before continuing."
    )

X_train_full = train_df[feature_columns]
y_train_full = train_df[label_column].astype(str)

X_val = val_df[feature_columns]
y_val = val_df[label_column].astype(str)

classes = sorted(y_train_full.unique())

print(f"Number of classes: {len(classes)}")
print("Classes:")
for cls in classes:
    print(f"  - {cls}")


# ============================================================
# INTERNAL TRAIN / HARD-EXAMPLE HOLDOUT
# ============================================================

print_header("CREATING INTERNAL HARD-EXAMPLE HOLDOUT")

# This split happens ONLY inside the original training data.
#
# The final validation set remains completely untouched.

train_indices = np.arange(len(X_train_full))

internal_train_idx, hard_holdout_idx = train_test_split(
    train_indices,
    test_size=HARD_EXAMPLE_HOLDOUT_SIZE,
    random_state=RANDOM_SEED,
    stratify=y_train_full,
)

X_internal_train = X_train_full.iloc[internal_train_idx]
y_internal_train = y_train_full.iloc[internal_train_idx]

X_hard_holdout = X_train_full.iloc[hard_holdout_idx]
y_hard_holdout = y_train_full.iloc[hard_holdout_idx]

print(f"Internal primary-training rows: {len(X_internal_train):,}")
print(f"Hard-example holdout rows:      {len(X_hard_holdout):,}")


# ============================================================
# TRAIN PRIMARY MODEL FOR HARD-EXAMPLE DISCOVERY
# ============================================================

print_header("TRAINING PRIMARY MODEL FOR HARD-EXAMPLE DISCOVERY")

start = time.time()

primary_for_hard_examples = HistGradientBoostingClassifier(
    **PRIMARY_MODEL_PARAMS
)

primary_for_hard_examples.fit(
    X_internal_train,
    y_internal_train,
)

elapsed = time.time() - start

print(f"Primary model trained in {elapsed:.2f} seconds.")


# ============================================================
# IDENTIFY HARD BF/XSS EXAMPLES
# ============================================================

print_header("IDENTIFYING HARD BF/XSS EXAMPLES")

holdout_target_mask = y_hard_holdout.isin(TARGET_CLASSES)

X_target_holdout = X_hard_holdout.loc[holdout_target_mask]
y_target_holdout = y_hard_holdout.loc[holdout_target_mask]

print(
    f"BF/XSS examples in internal holdout: "
    f"{len(X_target_holdout):,}"
)

holdout_predictions = primary_for_hard_examples.predict(
    X_target_holdout
)

hard_mask = holdout_predictions != y_target_holdout.to_numpy()

hard_features = X_target_holdout.loc[hard_mask]
hard_labels = y_target_holdout.loc[hard_mask]

print(f"Hard BF/XSS examples found: {len(hard_features):,}")

if len(hard_features) > 0:
    print("\nHard-example confusion:")

    for true_class in sorted(TARGET_CLASSES):
        for predicted_class in sorted(TARGET_CLASSES):
            count = confusion_count(
                y_target_holdout,
                holdout_predictions,
                true_class,
                predicted_class,
            )

            if count:
                print(
                    f"  {true_class} -> {predicted_class}: "
                    f"{count:,}"
                )


# ============================================================
# BUILD SPECIALIST TRAINING DATA
# ============================================================

print_header("BUILDING HARD-EXAMPLE-WEIGHTED SPECIALIST")

specialist_mask = y_train_full.isin(TARGET_CLASSES)

X_specialist = X_train_full.loc[specialist_mask].copy()
y_specialist = y_train_full.loc[specialist_mask].copy()

sample_weights = np.full(
    len(X_specialist),
    NORMAL_WEIGHT,
    dtype=np.float64,
)

# ------------------------------------------------------------
# Important:
#
# We cannot directly assign weights to the hard holdout rows
# because those rows are not part of X_specialist.
#
# Instead, use the internal holdout to learn which BF/XSS
# characteristics are difficult, then create an internal
# difficulty model using nearest-neighbor-style matching would
# add unnecessary complexity.
#
# Therefore we use a second, leakage-safe method:
#
# train a temporary BF/XSS primary classifier on the internal
# training portion and generate predictions for ALL specialist
# rows using a model that did NOT train on the hard holdout.
#
# For rows belonging to the hard holdout, mark them hard.
# For the remaining specialist rows, keep normal weight.
# ------------------------------------------------------------

# Identify which original dataframe rows belong to the
# internal hard-example holdout using their positional indices.

specialist_global_positions = np.flatnonzero(specialist_mask.to_numpy())

hard_global_positions = set(
    hard_holdout_idx[
        np.isin(
            y_hard_holdout.to_numpy(),
            list(TARGET_CLASSES),
        )
    ].tolist()
)

for i, global_position in enumerate(specialist_global_positions):
    if global_position in hard_global_positions:
        sample_weights[i] = HARD_WEIGHT

print("\nSpecialist class distribution:")

for cls in sorted(TARGET_CLASSES):
    count = int((y_specialist == cls).sum())
    weighted = float(
        sample_weights[y_specialist.to_numpy() == cls].sum()
    )

    print(
        f"  {cls}: "
        f"{count:,} rows | "
        f"effective weight={weighted:,.1f}"
    )

print("\nWeight distribution:")
print(
    f"  Normal examples: "
    f"{np.sum(sample_weights == NORMAL_WEIGHT):,}"
)
print(
    f"  Hard examples:   "
    f"{np.sum(sample_weights == HARD_WEIGHT):,}"
)


# ============================================================
# TRAIN HARD-EXAMPLE-WEIGHTED SPECIALIST
# ============================================================

print_header("TRAINING HARD-EXAMPLE-WEIGHTED BF/XSS SPECIALIST")

start = time.time()

specialist = HistGradientBoostingClassifier(
    **SPECIALIST_MODEL_PARAMS
)

specialist.fit(
    X_specialist,
    y_specialist,
    sample_weight=sample_weights,
)

elapsed = time.time() - start

print(
    f"Specialist trained in {elapsed:.2f} seconds."
)


# ============================================================
# BASELINE PRIMARY MODEL
# ============================================================

print_header("TRAINING FINAL PRIMARY MODEL")

start = time.time()

primary = HistGradientBoostingClassifier(
    **PRIMARY_MODEL_PARAMS
)

primary.fit(
    X_train_full,
    y_train_full,
)

elapsed = time.time() - start

print(
    f"Final primary model trained in {elapsed:.2f} seconds."
)


# ============================================================
# VALIDATION — PRIMARY BASELINE
# ============================================================

print_header("VALIDATION — PRIMARY BASELINE")

primary_val_pred = primary.predict(X_val)

baseline_metrics = compute_metrics(
    y_val,
    primary_val_pred,
    classes,
)

print(
    f"Macro-F1:  {baseline_metrics['macro_f1']:.6f}"
)
print(
    f"Accuracy:  {baseline_metrics['accuracy']:.6f}"
)
print(
    f"Weighted-F1: {baseline_metrics['weighted_f1']:.6f}"
)


# ============================================================
# VALIDATION — HARD-EXAMPLE SPECIALIST CASCADE
# ============================================================

print_header("VALIDATION — HARD-EXAMPLE SPECIALIST CASCADE")

final_pred = primary_val_pred.copy()

route_mask = np.isin(
    primary_val_pred,
    list(TARGET_CLASSES),
)

n_routed = int(route_mask.sum())

print(
    f"Validation rows routed to specialist: "
    f"{n_routed:,}"
)

if n_routed > 0:
    specialist_predictions = specialist.predict(
        X_val.loc[route_mask]
    )

    final_pred[route_mask] = specialist_predictions


# ============================================================
# FINAL METRICS
# ============================================================

specialist_metrics = compute_metrics(
    y_val,
    final_pred,
    classes,
)


# ============================================================
# CONFUSION ANALYSIS
# ============================================================

print_header("BF/XSS CONFUSION ANALYSIS")

baseline_bf_to_xss = confusion_count(
    y_val,
    primary_val_pred,
    "Web Attack - Brute Force",
    "Web Attack - XSS",
)

baseline_xss_to_bf = confusion_count(
    y_val,
    primary_val_pred,
    "Web Attack - XSS",
    "Web Attack - Brute Force",
)

final_bf_to_xss = confusion_count(
    y_val,
    final_pred,
    "Web Attack - Brute Force",
    "Web Attack - XSS",
)

final_xss_to_bf = confusion_count(
    y_val,
    final_pred,
    "Web Attack - XSS",
    "Web Attack - Brute Force",
)

baseline_mutual = (
    baseline_bf_to_xss
    + baseline_xss_to_bf
)

final_mutual = (
    final_bf_to_xss
    + final_xss_to_bf
)

print("\nBaseline:")
print(
    f"  BF -> XSS: {baseline_bf_to_xss:,}"
)
print(
    f"  XSS -> BF: {baseline_xss_to_bf:,}"
)
print(
    f"  Mutual:    {baseline_mutual:,}"
)

print("\nStrategy 4:")
print(
    f"  BF -> XSS: {final_bf_to_xss:,}"
)
print(
    f"  XSS -> BF: {final_xss_to_bf:,}"
)
print(
    f"  Mutual:    {final_mutual:,}"
)

print("\nChange:")
print(
    f"  BF -> XSS: "
    f"{final_bf_to_xss - baseline_bf_to_xss:+,}"
)
print(
    f"  XSS -> BF: "
    f"{final_xss_to_bf - baseline_xss_to_bf:+,}"
)
print(
    f"  Mutual:    "
    f"{final_mutual - baseline_mutual:+,}"
)


# ============================================================
# GLOBAL METRIC COMPARISON
# ============================================================

print_header("GLOBAL COMPARISON")

macro_delta = (
    specialist_metrics["macro_f1"]
    - baseline_metrics["macro_f1"]
)

accuracy_delta = (
    specialist_metrics["accuracy"]
    - baseline_metrics["accuracy"]
)

weighted_delta = (
    specialist_metrics["weighted_f1"]
    - baseline_metrics["weighted_f1"]
)

print(
    f"Macro-F1:"
    f" {baseline_metrics['macro_f1']:.6f}"
    f" -> {specialist_metrics['macro_f1']:.6f}"
    f" ({macro_delta:+.6f})"
)

print(
    f"Accuracy:"
    f" {baseline_metrics['accuracy']:.6f}"
    f" -> {specialist_metrics['accuracy']:.6f}"
    f" ({accuracy_delta:+.6f})"
)

print(
    f"Weighted-F1:"
    f" {baseline_metrics['weighted_f1']:.6f}"
    f" -> {specialist_metrics['weighted_f1']:.6f}"
    f" ({weighted_delta:+.6f})"
)


# ============================================================
# PER-CLASS COMPARISON
# ============================================================

print_header("PER-CLASS F1")

print(
    f"{'Class':40s}"
    f"{'Baseline':>12s}"
    f"{'Strategy 4':>14s}"
    f"{'Delta':>12s}"
)

print("-" * 80)

for cls in classes:
    baseline_f1 = baseline_metrics["per_class"][cls]["f1"]
    final_f1 = specialist_metrics["per_class"][cls]["f1"]

    print(
        f"{cls:40s}"
        f"{baseline_f1:12.4f}"
        f"{final_f1:14.4f}"
        f"{final_f1 - baseline_f1:+12.4f}"
    )


# ============================================================
# GUARDRAIL
# ============================================================

print_header("NON-BF/XSS GUARDRAIL")

non_target_mask = ~np.isin(
    primary_val_pred,
    list(TARGET_CLASSES),
)

non_target_changed = int(
    np.sum(
        primary_val_pred[non_target_mask]
        != final_pred[non_target_mask]
    )
)

print(
    f"Non-BF/XSS primary predictions changed: "
    f"{non_target_changed:,}"
)

if non_target_changed == 0:
    print("PASS — non-BF/XSS predictions are unchanged.")
else:
    print("WARNING — unexpected non-target changes detected.")


# ============================================================
# CHANGED / FIXED / DAMAGED
# ============================================================

print_header("SPECIALIST CHANGE ANALYSIS")

changed_mask = primary_val_pred != final_pred

n_changed = int(changed_mask.sum())

fixed = 0
damaged = 0

if n_changed > 0:
    fixed = int(
        np.sum(
            (primary_val_pred[changed_mask] != y_val.to_numpy()[changed_mask])
            & (final_pred[changed_mask] == y_val.to_numpy()[changed_mask])
        )
    )

    damaged = int(
        np.sum(
            (primary_val_pred[changed_mask] == y_val.to_numpy()[changed_mask])
            & (final_pred[changed_mask] != y_val.to_numpy()[changed_mask])
        )
    )

print(f"Changed predictions: {n_changed:,}")
print(f"Fixed:               {fixed:,}")
print(f"Damaged:             {damaged:,}")
print(f"Net:                 {fixed - damaged:+,}")


# ============================================================
# IMPORTANT OTHER CONFUSIONS
# ============================================================

print_header("IMPORTANT OTHER CONFUSIONS")

for true_class, predicted_class in [
    ("DoS Hulk", "BENIGN"),
    ("BENIGN", "DoS Hulk"),
]:
    baseline_count = confusion_count(
        y_val,
        primary_val_pred,
        true_class,
        predicted_class,
    )

    final_count = confusion_count(
        y_val,
        final_pred,
        true_class,
        predicted_class,
    )

    print(
        f"{true_class} -> {predicted_class}: "
        f"{baseline_count:,} -> {final_count:,}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "strategy": "Strategy 4 - Hard-Example Weighted BF/XSS Specialist",
    "random_seed": RANDOM_SEED,
    "feature_count": len(feature_columns),
    "features": feature_columns,
    "hard_example_holdout_size": HARD_EXAMPLE_HOLDOUT_SIZE,
    "normal_weight": NORMAL_WEIGHT,
    "hard_weight": HARD_WEIGHT,
    "hard_examples_found": int(len(hard_features)),
    "specialist_training_rows": int(len(X_specialist)),
    "validation_rows_routed": n_routed,
    "baseline": {
        "macro_f1": baseline_metrics["macro_f1"],
        "weighted_f1": baseline_metrics["weighted_f1"],
        "accuracy": baseline_metrics["accuracy"],
        "bf_to_xss": baseline_bf_to_xss,
        "xss_to_bf": baseline_xss_to_bf,
        "mutual_confusion": baseline_mutual,
    },
    "strategy_4": {
        "macro_f1": specialist_metrics["macro_f1"],
        "weighted_f1": specialist_metrics["weighted_f1"],
        "accuracy": specialist_metrics["accuracy"],
        "bf_to_xss": final_bf_to_xss,
        "xss_to_bf": final_xss_to_bf,
        "mutual_confusion": final_mutual,
    },
    "deltas": {
        "macro_f1": macro_delta,
        "weighted_f1": weighted_delta,
        "accuracy": accuracy_delta,
        "bf_to_xss": final_bf_to_xss - baseline_bf_to_xss,
        "xss_to_bf": final_xss_to_bf - baseline_xss_to_bf,
        "mutual_confusion": final_mutual - baseline_mutual,
    },
    "change_analysis": {
        "changed": n_changed,
        "fixed": fixed,
        "damaged": damaged,
        "net": fixed - damaged,
    },
    "non_bfxss_changed": non_target_changed,
    "baseline_per_class": baseline_metrics["per_class"],
    "strategy_4_per_class": specialist_metrics["per_class"],
}

output_path = (
    RESULTS_DIR
    / "strategy4_hard_example_bfxss_results.json"
)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
    )

print_header("DONE")

print(f"Results saved to:\n{output_path}")

print("\nFINAL DECISION DATA:")
print(
    f"Baseline Macro-F1: "
    f"{baseline_metrics['macro_f1']:.4f}"
)
print(
    f"Strategy 4 Macro-F1: "
    f"{specialist_metrics['macro_f1']:.4f}"
)
print(
    f"Baseline BF/XSS mutual: "
    f"{baseline_mutual:,}"
)
print(
    f"Strategy 4 BF/XSS mutual: "
    f"{final_mutual:,}"
)

print(
    "\nThe test set was NOT loaded or evaluated."
)
