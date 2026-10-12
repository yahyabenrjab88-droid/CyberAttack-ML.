"""
train_hgb_bfxss_specialist_cascade.py

Experiment: hgb_bfxss_specialist_cascade

Current baseline (this script's starting point)
-------------------------------------------------
HistGradientBoostingClassifier, class_weight="balanced", all other
hyperparameters unchanged from the original baseline. Validation Macro F1
= 85.88%. Remaining errors, in priority order:

    1. BF <-> XSS mutual confusion   (53 BF->XSS, 50 XSS->BF)   <- PRIMARY TARGET
    2. DoS Hulk -> BENIGN            (1,386 rows, F1 already 97.09%)  -- do not target
    3. BENIGN -> attack FPs          (1,038 rows, F1 already 99.64%)  -- do not target
    4. Bot                           (~26 errors, F1 already 82.90%)  -- do not target
    5. FTP-Patator precision         (216 BENIGN->FTP-Patator, recall 100%) -- do not target
    6. SQL Injection                 (n=3 in validation -- too small to act on)

Hypothesis
----------
BF and XSS are confused with EACH OTHER specifically, not broadly with the
rest of the dataset. A dedicated binary specialist, trained only on BF/XSS
rows using the FULL 68-feature set (explicitly NOT the previously-rejected
~15-feature restriction, which was tested and made BF/XSS mutual confusion
worse: 99 -> 110), should be able to draw a sharper boundary between just
these two classes than the 15-way primary model, which has to balance that
decision against 13 other classes simultaneously.

What changes vs. the current script
-------------------------------------
A second, small HistGradientBoostingClassifier ("the specialist") is trained
on train-set rows whose Label is "Web Attack - Brute Force" or
"Web Attack - XSS", using the same 68 model features.

At inference time on validation:
    - The primary model predicts as usual for every row.
    - ONLY for rows where the primary model's own prediction is BF or XSS,
      the specialist re-decides between just those two labels.
    - Every row the primary model assigned to any OTHER class is left
      completely untouched -- byte-for-byte identical prediction before
      and after the cascade.

This routing rule is what protects DoS Hulk, BENIGN, Bot, FTP-Patator, and
every other class: the cascade cannot change a prediction that wasn't
already BF or XSS, so it cannot alter those classes' precision or recall
in either direction.

What stays fixed
------------------
    - Train/validation/test split: unchanged, LOCKED.
    - Preprocessing pipeline: unchanged.
    - Primary model hyperparameters: unchanged from the class-weighted
      baseline that produced Macro F1 = 85.88%.
    - Test set: NOT loaded, NOT evaluated, anywhere in this script.

Decision metric
----------------
Primary: change in BF->XSS and XSS->BF confusion counts, and Macro F1.
Guardrail: per-class F1 for every class other than BF/XSS must be
unchanged (by construction of the routing rule -- verified and reported,
not just assumed).

Usage
-----
    (.venv) PS C:\\CyberAttack-ML> py src\\train_hgb_bfxss_specialist_cascade.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

EXPERIMENT_NAME = "hgb_bfxss_specialist_cascade"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODELS_DIR = PROJECT_ROOT / "models" / "historical" / "bfxss_specialist_cascade"
RESULTS_DIR = PROJECT_ROOT / "results"

LABEL_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"
NON_FEATURE_COLUMNS = {LABEL_COLUMN, PROVENANCE_COLUMN}

EXPECTED_N_FEATURES = 68
RANDOM_STATE = 42

LABEL_BF = "Web Attack - Brute Force"
LABEL_XSS = "Web Attack - XSS"
LABEL_BENIGN = "BENIGN"
LABEL_DOS_HULK = "DoS Hulk"

# Primary model: identical to the current class-weighted baseline (85.88% Macro F1).
# NOTHING about the primary model changes in this script.
PRIMARY_MODEL_PARAMS = dict(
    learning_rate=0.10,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    verbose=1,
)

# Specialist model: small, dedicated binary classifier for BF vs. XSS only.
# Full 68 features (not the previously-rejected 15-feature restriction).
# Lighter capacity / tighter early stopping than the primary model because
# the specialist's training set is small (only BF + XSS rows).
SPECIALIST_MODEL_PARAMS = dict(
    learning_rate=0.05,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=5,
    l2_regularization=0.0,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=15,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    verbose=1,
)

# Guardrail: warn if any non-BF/XSS class's F1 moves by more than this amount.
# Under the routing rule this should be exactly 0.0 -- this is a verification
# check, not a tolerance we expect to need.
GUARDRAIL_F1_DELTA_THRESHOLD = 0.0005


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_parquet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} parquet not found at {path}. "
            f"Expected the locked preprocessed output from src/preprocess.py."
        )
    log(f"Loading {name} from {path} ...")
    df = pd.read_parquet(path)
    log(f"  -> {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def verify_schema(train_df: pd.DataFrame, val_df: pd.DataFrame) -> list[str]:
    for name, df in (("train", train_df), ("validation", val_df)):
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"{name} set is missing required column '{LABEL_COLUMN}'")

    train_features = get_feature_columns(train_df)
    val_features = get_feature_columns(val_df)

    if train_features != val_features:
        only_train = set(train_features) - set(val_features)
        only_val = set(val_features) - set(train_features)
        raise ValueError(
            "Feature column mismatch between train and validation.\n"
            f"  Only in train: {sorted(only_train)}\n"
            f"  Only in validation: {sorted(only_val)}"
        )

    if len(train_features) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} model features, found {len(train_features)}: "
            f"{train_features}"
        )

    non_numeric = [
        c for c in train_features
        if not pd.api.types.is_numeric_dtype(train_df[c])
    ]
    if non_numeric:
        raise ValueError(f"Non-numeric model features detected: {non_numeric}")

    log(f"Schema check passed: {len(train_features)} model features, "
        f"train/validation columns identical, all numeric.")
    return train_features


def downcast_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Downcast float64 -> float32 in place to reduce memory footprint."""
    for c in feature_cols:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)
        elif pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, class_names: list[str]):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).set_index("class")

    report_dict = classification_report(
        y_true, y_pred, labels=class_names, zero_division=0, output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_df,
        "report": report_df,
        "confusion_matrix": cm_df,
    }


def confusion_count(cm_df: pd.DataFrame, true_label: str, pred_label: str) -> int:
    if true_label not in cm_df.index or pred_label not in cm_df.columns:
        return 0
    return int(cm_df.loc[true_label, pred_label])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"=== Experiment: {EXPERIMENT_NAME} ===")
    log(f"Python {sys.version.split()[0]} | platform {platform.platform()}")

    # ---- Load (train + validation ONLY; test is never referenced) --------- #
    train_df = load_parquet(TRAIN_PATH, "train")
    val_df = load_parquet(VAL_PATH, "validation")

    feature_cols = verify_schema(train_df, val_df)

    train_df = downcast_features(train_df, feature_cols)
    val_df = downcast_features(val_df, feature_cols)

    if LABEL_BF not in train_df[LABEL_COLUMN].unique() or LABEL_XSS not in train_df[LABEL_COLUMN].unique():
        raise ValueError(
            f"Expected labels '{LABEL_BF}' and '{LABEL_XSS}' in training data. "
            f"Found: {sorted(train_df[LABEL_COLUMN].unique())}"
        )

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COLUMN].astype(str)
    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COLUMN].astype(str)

    class_names = sorted(y_train.unique())
    log(f"Classes ({len(class_names)}): {class_names}")

    del train_df, val_df

    # ------------------------------------------------------------------- #
    # Stage 1: Primary model (identical to current 85.88% Macro F1 baseline)
    # ------------------------------------------------------------------- #
    log("--- Stage 1: training primary model (unchanged from current baseline) ---")
    primary_model = HistGradientBoostingClassifier(**PRIMARY_MODEL_PARAMS)

    primary_train_start = time.perf_counter()
    primary_model.fit(X_train, y_train)
    primary_train_time_sec = time.perf_counter() - primary_train_start
    log(f"Primary model trained in {primary_train_time_sec:.2f}s "
        f"({getattr(primary_model, 'n_iter_', 'unknown')} boosting iterations)")

    predict_start = time.perf_counter()
    y_pred_primary = primary_model.predict(X_val)
    primary_predict_time_sec = time.perf_counter() - predict_start
    log(f"Primary validation prediction complete in {primary_predict_time_sec:.2f}s")

    before = compute_metrics(y_val, y_pred_primary, class_names)
    log(f"[BEFORE cascade] Macro F1 = {before['macro_f1']:.4%}, "
        f"Accuracy = {before['accuracy']:.4%}")

    # ------------------------------------------------------------------- #
    # Stage 2: BF/XSS specialist (trained on train rows only)
    # ------------------------------------------------------------------- #
    log("--- Stage 2: training BF/XSS specialist ---")
    specialist_train_mask = y_train.isin([LABEL_BF, LABEL_XSS])
    X_train_specialist = X_train.loc[specialist_train_mask]
    y_train_specialist = y_train.loc[specialist_train_mask]

    log(f"Specialist training rows: {len(y_train_specialist):,} "
        f"({(y_train_specialist == LABEL_BF).sum():,} BF / "
        f"{(y_train_specialist == LABEL_XSS).sum():,} XSS)")

    specialist_model = HistGradientBoostingClassifier(**SPECIALIST_MODEL_PARAMS)

    specialist_train_start = time.perf_counter()
    specialist_model.fit(X_train_specialist, y_train_specialist)
    specialist_train_time_sec = time.perf_counter() - specialist_train_start
    log(f"Specialist trained in {specialist_train_time_sec:.2f}s "
        f"({getattr(specialist_model, 'n_iter_', 'unknown')} boosting iterations)")

    # ------------------------------------------------------------------- #
    # Stage 3: Cascade routing on validation
    #
    # Only rows where the PRIMARY model's own prediction is BF or XSS are
    # re-decided by the specialist. Every other row is left byte-for-byte
    # identical to the primary model's prediction.
    # ------------------------------------------------------------------- #
    log("--- Stage 3: applying cascade to validation predictions ---")
    route_mask = np.isin(y_pred_primary, [LABEL_BF, LABEL_XSS])
    n_routed = int(route_mask.sum())
    log(f"Rows routed to specialist (primary predicted BF or XSS): {n_routed:,}")

    y_pred_cascade = y_pred_primary.copy()
    if n_routed > 0:
        specialist_predict_start = time.perf_counter()
        specialist_preds = specialist_model.predict(X_val.loc[route_mask])
        specialist_predict_time_sec = time.perf_counter() - specialist_predict_start
        y_pred_cascade[route_mask] = specialist_preds
        n_flipped = int((specialist_preds != y_pred_primary[route_mask]).sum())
        log(f"Specialist prediction complete in {specialist_predict_time_sec:.2f}s; "
            f"changed {n_flipped:,} of {n_routed:,} routed predictions")
    else:
        specialist_predict_time_sec = 0.0

    after = compute_metrics(y_val, y_pred_cascade, class_names)
    log(f"[AFTER cascade]  Macro F1 = {after['macro_f1']:.4%}, "
        f"Accuracy = {after['accuracy']:.4%}")

    # ------------------------------------------------------------------- #
    # Guardrail check: every class other than BF/XSS must be UNCHANGED
    # ------------------------------------------------------------------- #
    log("--- Guardrail check: non-BF/XSS classes must be unaffected ---")
    guardrail_violations = []
    for cls in class_names:
        if cls in (LABEL_BF, LABEL_XSS):
            continue
        f1_before = before["per_class"].loc[cls, "f1"]
        f1_after = after["per_class"].loc[cls, "f1"]
        delta = f1_after - f1_before
        if abs(delta) > GUARDRAIL_F1_DELTA_THRESHOLD:
            guardrail_violations.append((cls, f1_before, f1_after, delta))

    if guardrail_violations:
        log("WARNING: the following classes changed despite not being routed "
            "through the specialist -- investigate before trusting this result:")
        for cls, f1_b, f1_a, delta in guardrail_violations:
            log(f"    {cls}: F1 {f1_b:.4%} -> {f1_a:.4%} (delta {delta:+.4%})")
    else:
        log("PASSED: all non-BF/XSS classes have identical F1 before and after "
            "the cascade (as guaranteed by the routing rule).")

    # ------------------------------------------------------------------- #
    # Before/after comparison report
    # ------------------------------------------------------------------- #
    bf_to_xss_before = confusion_count(before["confusion_matrix"], LABEL_BF, LABEL_XSS)
    xss_to_bf_before = confusion_count(before["confusion_matrix"], LABEL_XSS, LABEL_BF)
    hulk_to_benign_before = confusion_count(before["confusion_matrix"], LABEL_DOS_HULK, LABEL_BENIGN)
    benign_fp_before = int(before["confusion_matrix"].loc[LABEL_BENIGN].drop(LABEL_BENIGN).sum())

    bf_to_xss_after = confusion_count(after["confusion_matrix"], LABEL_BF, LABEL_XSS)
    xss_to_bf_after = confusion_count(after["confusion_matrix"], LABEL_XSS, LABEL_BF)
    hulk_to_benign_after = confusion_count(after["confusion_matrix"], LABEL_DOS_HULK, LABEL_BENIGN)
    benign_fp_after = int(after["confusion_matrix"].loc[LABEL_BENIGN].drop(LABEL_BENIGN).sum())

    comparison = pd.DataFrame(
        {
            "before": [
                before["macro_f1"], before["weighted_f1"], before["accuracy"],
                bf_to_xss_before, xss_to_bf_before, hulk_to_benign_before, benign_fp_before,
            ],
            "after": [
                after["macro_f1"], after["weighted_f1"], after["accuracy"],
                bf_to_xss_after, xss_to_bf_after, hulk_to_benign_after, benign_fp_after,
            ],
        },
        index=[
            "macro_f1", "weighted_f1", "accuracy",
            "BF->XSS count", "XSS->BF count", "DoS Hulk->BENIGN count", "BENIGN false positives",
        ],
    )
    comparison["delta"] = comparison["after"] - comparison["before"]

    per_class_comparison = before["per_class"][["f1", "support"]].rename(
        columns={"f1": "f1_before"}
    ).join(after["per_class"][["f1"]].rename(columns={"f1": "f1_after"}))
    per_class_comparison["delta"] = (
        per_class_comparison["f1_after"] - per_class_comparison["f1_before"]
    )
    per_class_comparison = per_class_comparison.sort_values("support", ascending=False)

    log("=== BEFORE / AFTER summary ===")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))
    log("=== Per-class F1: before / after / delta ===")
    print(per_class_comparison.to_string(float_format=lambda x: f"{x:.4f}"))

    # ------------------------------------------------------------------- #
    # Save artifacts
    # ------------------------------------------------------------------- #
    primary_model_path = MODELS_DIR / f"{EXPERIMENT_NAME}_primary.joblib"
    joblib.dump(primary_model, primary_model_path)
    log(f"Saved primary model -> {primary_model_path}")

    specialist_model_path = MODELS_DIR / f"{EXPERIMENT_NAME}_specialist.joblib"
    joblib.dump(specialist_model, specialist_model_path)
    log(f"Saved specialist model -> {specialist_model_path}")

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "before": {
            "accuracy": before["accuracy"],
            "macro_f1": before["macro_f1"],
            "weighted_f1": before["weighted_f1"],
        },
        "after": {
            "accuracy": after["accuracy"],
            "macro_f1": after["macro_f1"],
            "weighted_f1": after["weighted_f1"],
        },
        "confusion_pairs": {
            "BF_to_XSS": {"before": bf_to_xss_before, "after": bf_to_xss_after},
            "XSS_to_BF": {"before": xss_to_bf_before, "after": xss_to_bf_after},
            "DoSHulk_to_BENIGN": {"before": hulk_to_benign_before, "after": hulk_to_benign_after},
            "BENIGN_false_positives": {"before": benign_fp_before, "after": benign_fp_after},
        },
        "n_routed_to_specialist": n_routed,
        "guardrail_passed": len(guardrail_violations) == 0,
        "n_train_rows": int(len(y_train)),
        "n_validation_rows": int(len(y_val)),
        "n_features": len(feature_cols),
    }
    metrics_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log(f"Saved metrics -> {metrics_path}")

    before["report"].to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_classification_report_before.csv")
    after["report"].to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_classification_report_after.csv")
    before["confusion_matrix"].to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_confusion_matrix_before.csv")
    after["confusion_matrix"].to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_confusion_matrix_after.csv")
    per_class_comparison.to_csv(RESULTS_DIR / f"{EXPERIMENT_NAME}_per_class_comparison.csv")
    log("Saved before/after classification reports, confusion matrices, and per-class comparison")

    summary = {
        "experiment": EXPERIMENT_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": (
            "BF and XSS are confused primarily with each other, not broadly with "
            "other classes. A binary specialist trained only on BF/XSS rows using "
            "the full 68-feature set can separate them more sharply than the "
            "15-way primary model, without affecting any other class because it "
            "is only consulted when the primary model itself predicted BF or XSS."
        ),
        "changed_vs_current_baseline": [
            "Added a second-stage binary specialist for BF vs XSS.",
            "Added cascade routing logic at inference time (validation only; "
            "test set untouched).",
        ],
        "unchanged_vs_current_baseline": [
            "Primary model architecture and all hyperparameters",
            "class_weight='balanced' on the primary model",
            "Train/validation/test split",
            "Preprocessing pipeline",
            "68 model features",
        ],
        "primary_model_params": {k: str(v) for k, v in PRIMARY_MODEL_PARAMS.items()},
        "specialist_model_params": {k: str(v) for k, v in SPECIALIST_MODEL_PARAMS.items()},
        "routing_rule": (
            "A validation row is re-decided by the specialist if and only if the "
            "primary model's own argmax prediction for that row is "
            f"'{LABEL_BF}' or '{LABEL_XSS}'. All other rows are left unchanged."
        ),
        "guardrail": {
            "description": (
                "Per-class F1 for every class other than BF/XSS must be identical "
                "before and after the cascade, since the routing rule cannot touch "
                "predictions outside the two targeted classes."
            ),
            "passed": len(guardrail_violations) == 0,
            "violations": [
                {"class": cls, "f1_before": f1_b, "f1_after": f1_a, "delta": d}
                for cls, f1_b, f1_a, d in guardrail_violations
            ],
        },
        "data": {
            "train_path": str(TRAIN_PATH),
            "validation_path": str(VAL_PATH),
            "test_path_used": None,
            "n_train_rows": int(len(y_train)),
            "n_validation_rows": int(len(y_val)),
            "n_specialist_train_rows": int(len(y_train_specialist)),
            "n_features": len(feature_cols),
            "n_classes": len(class_names),
        },
        "timing": {
            "primary_train_time_sec": primary_train_time_sec,
            "primary_predict_time_sec": primary_predict_time_sec,
            "specialist_train_time_sec": specialist_train_time_sec,
            "specialist_predict_time_sec": specialist_predict_time_sec,
        },
        "metrics_before": {
            "accuracy": before["accuracy"],
            "macro_f1": before["macro_f1"],
            "weighted_f1": before["weighted_f1"],
        },
        "metrics_after": {
            "accuracy": after["accuracy"],
            "macro_f1": after["macro_f1"],
            "weighted_f1": after["weighted_f1"],
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    summary_path = RESULTS_DIR / f"{EXPERIMENT_NAME}_training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(f"Saved training summary -> {summary_path}")

    log("=== Done. Test set was not loaded or evaluated in this script. ===")


if __name__ == "__main__":
    main()
