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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "hgb_bfxss_gated_calibrated_cascade"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODELS_DIR = PROJECT_ROOT / "models" / "historical" / "bfxss_gated_calibrated_cascade"
RESULTS_DIR = PROJECT_ROOT / "results"

LABEL_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"

NON_FEATURE_COLUMNS = {
    LABEL_COLUMN,
    PROVENANCE_COLUMN,
}

EXPECTED_N_FEATURES = 68
RANDOM_STATE = 42

LABEL_BF = "Web Attack - Brute Force"
LABEL_XSS = "Web Attack - XSS"
LABEL_BENIGN = "BENIGN"
LABEL_DOS_HULK = "DoS Hulk"

TAU_GRID = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

CALIBRATION_CV_FOLDS = 5
CALIBRATION_METHOD = "sigmoid"


# ============================================================================
# PRIMARY MODEL
# EXACTLY THE SAME AS CURRENT BASELINE
# ============================================================================

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


# ============================================================================
# SPECIALIST
# SAME BASE ARCHITECTURE AS PREVIOUS EXPERIMENT
# ============================================================================

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
    verbose=0,
)


# ============================================================================
# HELPERS
# ============================================================================

def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def load_parquet(path: Path, name: str) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"{name} parquet not found:\n{path}"
        )

    log(f"Loading {name}: {path}")

    df = pd.read_parquet(path)

    log(
        f"  {name}: "
        f"{df.shape[0]:,} rows x {df.shape[1]} columns"
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:

    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLUMNS
    ]


def verify_schema(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[str]:

    for name, df in (
        ("train", train_df),
        ("validation", val_df),
    ):

        if LABEL_COLUMN not in df.columns:

            raise ValueError(
                f"{name} is missing '{LABEL_COLUMN}'"
            )

    train_features = get_feature_columns(train_df)
    val_features = get_feature_columns(val_df)

    if train_features != val_features:

        only_train = sorted(
            set(train_features) - set(val_features)
        )

        only_val = sorted(
            set(val_features) - set(train_features)
        )

        raise ValueError(
            "Train/validation feature mismatch.\n"
            f"Only train: {only_train}\n"
            f"Only validation: {only_val}"
        )

    if len(train_features) != EXPECTED_N_FEATURES:

        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features, "
            f"found {len(train_features)}"
        )

    non_numeric = [
        c
        for c in train_features
        if not pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if non_numeric:

        raise ValueError(
            f"Non-numeric features detected: {non_numeric}"
        )

    log(
        f"Schema verified: "
        f"{len(train_features)} numeric features"
    )

    return train_features


def downcast_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:

    for c in feature_cols:

        if pd.api.types.is_float_dtype(df[c]):

            df[c] = df[c].astype(np.float32)

        elif pd.api.types.is_integer_dtype(df[c]):

            df[c] = pd.to_numeric(
                df[c],
                downcast="integer",
            )

    return df


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

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

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=class_names,
            zero_division=0,
        )
    )

    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).set_index("class")

    report = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=class_names,
            zero_division=0,
            output_dict=True,
        )
    ).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=class_names,
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "report": report,
        "confusion_matrix": cm_df,
    }


def confusion_count(
    cm_df: pd.DataFrame,
    true_label: str,
    predicted_label: str,
) -> int:

    return int(
        cm_df.loc[
            true_label,
            predicted_label,
        ]
    )


def count_fixed_and_damaged(
    y_true: np.ndarray,
    primary_pred: np.ndarray,
    candidate_pred: np.ndarray,
    changed_mask: np.ndarray,
) -> tuple[int, int, int]:

    fixed = 0
    damaged = 0
    changed = int(changed_mask.sum())

    indices = np.where(changed_mask)[0]

    for i in indices:

        before_correct = (
            primary_pred[i] == y_true[i]
        )

        after_correct = (
            candidate_pred[i] == y_true[i]
        )

        if not before_correct and after_correct:
            fixed += 1

        elif before_correct and not after_correct:
            damaged += 1

    return fixed, damaged, changed


def exact_non_target_guardrail(
    primary_pred: np.ndarray,
    candidate_pred: np.ndarray,
) -> tuple[bool, int]:

    non_target_mask = ~np.isin(
        primary_pred,
        [LABEL_BF, LABEL_XSS],
    )

    differences = (
        candidate_pred[non_target_mask]
        != primary_pred[non_target_mask]
    )

    n_differences = int(differences.sum())

    return (
        n_differences == 0,
        n_differences,
    )


# ============================================================================
# MAIN
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

    log(
        f"=== {EXPERIMENT_NAME} ==="
    )

    log(
        f"Python {sys.version.split()[0]}"
    )

    # ========================================================================
    # LOAD TRAIN + VALIDATION ONLY
    # ========================================================================

    train_df = load_parquet(
        TRAIN_PATH,
        "train",
    )

    val_df = load_parquet(
        VAL_PATH,
        "validation",
    )

    feature_cols = verify_schema(
        train_df,
        val_df,
    )

    train_df = downcast_features(
        train_df,
        feature_cols,
    )

    val_df = downcast_features(
        val_df,
        feature_cols,
    )

    X_train = train_df[feature_cols]
    y_train = train_df[LABEL_COLUMN].astype(str)

    X_val = val_df[feature_cols]
    y_val = val_df[LABEL_COLUMN].astype(str)

    class_names = sorted(
        y_train.unique()
    )

    log(
        f"Classes: {len(class_names)}"
    )

    log(
        f"Features: {len(feature_cols)}"
    )

    # ========================================================================
    # STAGE 1 â€” PRIMARY MODEL
    # ========================================================================

    log(
        "=== Stage 1: Primary HGB ==="
    )

    primary_model = (
        HistGradientBoostingClassifier(
            **PRIMARY_MODEL_PARAMS
        )
    )

    start = time.perf_counter()

    primary_model.fit(
        X_train,
        y_train,
    )

    primary_train_time = (
        time.perf_counter() - start
    )

    log(
        f"Primary training time: "
        f"{primary_train_time:.2f}s"
    )

    y_pred_primary = (
        primary_model.predict(X_val)
    )

    baseline = compute_metrics(
        y_val,
        y_pred_primary,
        class_names,
    )

    bf_xss_before = confusion_count(
        baseline["confusion_matrix"],
        LABEL_BF,
        LABEL_XSS,
    )

    xss_bf_before = confusion_count(
        baseline["confusion_matrix"],
        LABEL_XSS,
        LABEL_BF,
    )

    mutual_before = (
        bf_xss_before +
        xss_bf_before
    )

    log(
        f"Baseline Macro-F1: "
        f"{baseline['macro_f1']:.4%}"
    )

    log(
        f"Baseline Accuracy: "
        f"{baseline['accuracy']:.4%}"
    )

    log(
        f"Baseline BF->XSS: "
        f"{bf_xss_before}"
    )

    log(
        f"Baseline XSS->BF: "
        f"{xss_bf_before}"
    )

    log(
        f"Baseline mutual confusion: "
        f"{mutual_before}"
    )

    # ========================================================================
    # STAGE 2 â€” RAW SPECIALIST
    #
    # This is retained as an explicit reference point.
    # ========================================================================

    log(
        "=== Stage 2: Raw BF/XSS specialist ==="
    )

    specialist_mask = y_train.isin(
        [LABEL_BF, LABEL_XSS]
    )

    X_train_spec = X_train.loc[
        specialist_mask
    ]

    y_train_spec = y_train.loc[
        specialist_mask
    ]

    log(
        f"Specialist training rows: "
        f"{len(y_train_spec):,}"
    )

    log(
        f"  BF: "
        f"{int((y_train_spec == LABEL_BF).sum()):,}"
    )

    log(
        f"  XSS: "
        f"{int((y_train_spec == LABEL_XSS).sum()):,}"
    )

    raw_specialist = (
        HistGradientBoostingClassifier(
            **SPECIALIST_MODEL_PARAMS
        )
    )

    start = time.perf_counter()

    raw_specialist.fit(
        X_train_spec,
        y_train_spec,
    )

    raw_specialist_train_time = (
        time.perf_counter() - start
    )

    # ========================================================================
    # ROUTING MASK
    # ========================================================================

    route_mask = np.isin(
        y_pred_primary,
        [LABEL_BF, LABEL_XSS],
    )

    n_routed = int(
        route_mask.sum()
    )

    log(
        f"Rows routed to BF/XSS specialist: "
        f"{n_routed:,}"
    )

    X_val_routed = X_val.loc[
        route_mask
    ]

    # ========================================================================
    # RAW SPECIALIST PREDICTIONS
    # ========================================================================

    raw_specialist_pred = (
        raw_specialist.predict(
            X_val_routed
        )
    )

    raw_specialist_proba = (
        raw_specialist.predict_proba(
            X_val_routed
        )
    )

    raw_confidence = (
        raw_specialist_proba.max(
            axis=1
        )
    )

    raw_cascade_pred = (
        y_pred_primary.copy()
    )

    routed_indices = np.where(
        route_mask
    )[0]

    raw_cascade_pred[
        routed_indices
    ] = raw_specialist_pred

    raw_changed_mask = (
        raw_cascade_pred !=
        y_pred_primary
    )

    raw_fixed, raw_damaged, raw_changed = (
        count_fixed_and_damaged(
            y_val.to_numpy(),
            y_pred_primary,
            raw_cascade_pred,
            raw_changed_mask,
        )
    )

    raw_metrics = compute_metrics(
        y_val,
        raw_cascade_pred,
        class_names,
    )

    raw_bf_xss = confusion_count(
        raw_metrics["confusion_matrix"],
        LABEL_BF,
        LABEL_XSS,
    )

    raw_xss_bf = confusion_count(
        raw_metrics["confusion_matrix"],
        LABEL_XSS,
        LABEL_BF,
    )

    log(
        "Raw specialist reference:"
    )

    log(
        f"  Macro-F1: "
        f"{raw_metrics['macro_f1']:.4%}"
    )

    log(
        f"  BF->XSS: "
        f"{raw_bf_xss}"
    )

    log(
        f"  XSS->BF: "
        f"{raw_xss_bf}"
    )

    log(
        f"  Mutual: "
        f"{raw_bf_xss + raw_xss_bf}"
    )

    log(
        f"  Changed: "
        f"{raw_changed}"
    )

    log(
        f"  Fixed: "
        f"{raw_fixed}"
    )

    log(
        f"  Damaged: "
        f"{raw_damaged}"
    )

    # ========================================================================
    # STAGE 3 â€” CALIBRATED SPECIALIST
    # ========================================================================

    log(
        "=== Stage 3: Calibrated BF/XSS specialist ==="
    )

    calibrated_specialist = (
        CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(
                **SPECIALIST_MODEL_PARAMS
            ),
            method=CALIBRATION_METHOD,
            cv=CALIBRATION_CV_FOLDS,
        )
    )

    start = time.perf_counter()

    calibrated_specialist.fit(
        X_train_spec,
        y_train_spec,
    )

    calibrated_train_time = (
        time.perf_counter() - start
    )

    log(
        f"Calibrated specialist training: "
        f"{calibrated_train_time:.2f}s"
    )

    calibrated_proba = (
        calibrated_specialist.predict_proba(
            X_val_routed
        )
    )

    calibrated_classes = (
        np.asarray(
            calibrated_specialist.classes_
        )
    )

    calibrated_pred_idx = (
        calibrated_proba.argmax(
            axis=1
        )
    )

    calibrated_specialist_pred = (
        calibrated_classes[
            calibrated_pred_idx
        ]
    )

    calibrated_confidence = (
        calibrated_proba[
            np.arange(
                len(calibrated_proba)
            ),
            calibrated_pred_idx,
        ]
    )

    # ========================================================================
    # THRESHOLD GRID
    # ========================================================================

    log(
        "=== Confidence threshold grid ==="
    )

    grid_rows = []

    predictions_by_tau = {}

    for tau in TAU_GRID:

        candidate = (
            y_pred_primary.copy()
        )

        accept_mask = (
            calibrated_confidence >= tau
        )

        override_indices = (
            routed_indices[
                accept_mask
            ]
        )

        candidate[
            override_indices
        ] = calibrated_specialist_pred[
            accept_mask
        ]

        changed_mask = (
            candidate !=
            y_pred_primary
        )

        fixed, damaged, changed = (
            count_fixed_and_damaged(
                y_val.to_numpy(),
                y_pred_primary,
                candidate,
                changed_mask,
            )
        )

        metrics = compute_metrics(
            y_val,
            candidate,
            class_names,
        )

        bf_to_xss = confusion_count(
            metrics["confusion_matrix"],
            LABEL_BF,
            LABEL_XSS,
        )

        xss_to_bf = confusion_count(
            metrics["confusion_matrix"],
            LABEL_XSS,
            LABEL_BF,
        )

        mutual = (
            bf_to_xss +
            xss_to_bf
        )

        guardrail_ok, guardrail_changes = (
            exact_non_target_guardrail(
                y_pred_primary,
                candidate,
            )
        )

        grid_rows.append(
            {
                "tau": tau,
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "accuracy": metrics["accuracy"],
                "bf_f1": metrics["per_class"].loc[
                    LABEL_BF,
                    "f1",
                ],
                "xss_f1": metrics["per_class"].loc[
                    LABEL_XSS,
                    "f1",
                ],
                "bf_to_xss": bf_to_xss,
                "xss_to_bf": xss_to_bf,
                "total_mutual": mutual,
                "n_overridden": int(
                    accept_mask.sum()
                ),
                "n_changed": changed,
                "n_fixed": fixed,
                "n_damaged": damaged,
                "net_fixed_minus_damaged": (
                    fixed - damaged
                ),
                "guardrail_passed": (
                    guardrail_ok
                ),
                "non_target_changes": (
                    guardrail_changes
                ),
            }
        )

        predictions_by_tau[
            tau
        ] = candidate

    grid_df = pd.DataFrame(
        grid_rows
    )

    print()
    print(
        grid_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================================
    # SELECT BEST THRESHOLD
    #
    # Primary criterion:
    #     lowest total BF/XSS mutual confusion
    #
    # Tie-break:
    #     highest Macro-F1
    #
    # Then:
    #     highest net fixed-damaged
    # ========================================================================

    valid_grid = grid_df[
        grid_df["guardrail_passed"]
    ].copy()

    if valid_grid.empty:

        raise RuntimeError(
            "No threshold passed the exact non-target guardrail."
        )

    best_row = (
        valid_grid.sort_values(
            by=[
                "total_mutual",
                "macro_f1",
                "net_fixed_minus_damaged",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .iloc[0]
    )

    best_tau = float(
        best_row["tau"]
    )

    best_candidate = (
        predictions_by_tau[
            best_tau
        ]
    )

    # ========================================================================
    # IMPORTANT: DO NOT AUTOMATICALLY CALL IT SUCCESSFUL
    #
    # If the best threshold does not improve mutual confusion relative to
    # baseline, retain the baseline as the selected development policy.
    # ========================================================================

    best_mutual = int(
        best_row["total_mutual"]
    )

    if best_mutual < mutual_before:

        selected_policy = (
            f"gated_tau_{best_tau:.2f}"
        )

        selected_prediction = (
            best_candidate
        )

        selected_tau = best_tau

        log(
            f"Selected gated policy: "
            f"tau={best_tau:.2f}"
        )

    else:

        selected_policy = (
            "primary_only"
        )

        selected_prediction = (
            y_pred_primary.copy()
        )

        selected_tau = None

        log(
            "NO gated threshold reduced "
            "total BF/XSS mutual confusion."
        )

        log(
            "Development decision: "
            "retain primary-only policy."
        )

    # ========================================================================
    # FINAL METRICS
    # ========================================================================

    after = compute_metrics(
        y_val,
        selected_prediction,
        class_names,
    )

    bf_xss_after = confusion_count(
        after["confusion_matrix"],
        LABEL_BF,
        LABEL_XSS,
    )

    xss_bf_after = confusion_count(
        after["confusion_matrix"],
        LABEL_XSS,
        LABEL_BF,
    )

    mutual_after = (
        bf_xss_after +
        xss_bf_after
    )

    changed_mask = (
        selected_prediction !=
        y_pred_primary
    )

    fixed, damaged, changed = (
        count_fixed_and_damaged(
            y_val.to_numpy(),
            y_pred_primary,
            selected_prediction,
            changed_mask,
        )
    )

    # ========================================================================
    # EXACT GUARDRAIL
    # ========================================================================

    guardrail_passed, non_target_changes = (
        exact_non_target_guardrail(
            y_pred_primary,
            selected_prediction,
        )
    )

    if guardrail_passed:

        log(
            "GUARDRAIL PASSED: "
            "non-BF/XSS predictions are EXACTLY unchanged."
        )

    else:

        raise RuntimeError(
            "GUARDRAIL FAILED: "
            f"{non_target_changes} non-BF/XSS predictions changed."
        )

    # ========================================================================
    # ADDITIONAL CONFUSION COUNTS
    # ========================================================================

    hulk_benign_before = confusion_count(
        baseline["confusion_matrix"],
        LABEL_DOS_HULK,
        LABEL_BENIGN,
    )

    hulk_benign_after = confusion_count(
        after["confusion_matrix"],
        LABEL_DOS_HULK,
        LABEL_BENIGN,
    )

    benign_fp_before = int(
        baseline["confusion_matrix"]
        .loc[LABEL_BENIGN]
        .drop(LABEL_BENIGN)
        .sum()
    )

    benign_fp_after = int(
        after["confusion_matrix"]
        .loc[LABEL_BENIGN]
        .drop(LABEL_BENIGN)
        .sum()
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("FINAL BEFORE / AFTER")
    print("=" * 80)

    summary_table = pd.DataFrame(
        {
            "Baseline": [
                baseline["macro_f1"],
                baseline["accuracy"],
                baseline["per_class"].loc[
                    LABEL_BF,
                    "f1",
                ],
                baseline["per_class"].loc[
                    LABEL_XSS,
                    "f1",
                ],
                bf_xss_before,
                xss_bf_before,
                mutual_before,
                hulk_benign_before,
                benign_fp_before,
                0,
                0,
                0,
            ],
            "Selected": [
                after["macro_f1"],
                after["accuracy"],
                after["per_class"].loc[
                    LABEL_BF,
                    "f1",
                ],
                after["per_class"].loc[
                    LABEL_XSS,
                    "f1",
                ],
                bf_xss_after,
                xss_bf_after,
                mutual_after,
                hulk_benign_after,
                benign_fp_after,
                changed,
                fixed,
                damaged,
            ],
        },
        index=[
            "Macro-F1",
            "Accuracy",
            "BF F1",
            "XSS F1",
            "BF->XSS",
            "XSS->BF",
            "Total mutual confusion",
            "DoS Hulk->BENIGN",
            "BENIGN false positives",
            "Predictions changed",
            "Errors fixed",
            "Correct predictions damaged",
        ],
    )

    print(
        summary_table.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    print()
    print(
        f"Selected policy: {selected_policy}"
    )

    # ========================================================================
    # PER-CLASS COMPARISON
    # ========================================================================

    per_class = (
        baseline["per_class"][
            ["f1", "support"]
        ]
        .rename(
            columns={
                "f1": "f1_before"
            }
        )
        .join(
            after["per_class"][
                ["f1"]
            ].rename(
                columns={
                    "f1": "f1_after"
                }
            )
        )
    )

    per_class["delta"] = (
        per_class["f1_after"]
        -
        per_class["f1_before"]
    )

    print()
    print("=" * 80)
    print("PER-CLASS F1")
    print("=" * 80)

    print(
        per_class.to_string(
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # ========================================================================
    # SAVE MODELS
    # ========================================================================

    primary_model_path = (
        MODELS_DIR /
        f"{EXPERIMENT_NAME}_primary.joblib"
    )

    calibrated_model_path = (
        MODELS_DIR /
        f"{EXPERIMENT_NAME}_calibrated_specialist.joblib"
    )

    joblib.dump(
        primary_model,
        primary_model_path,
    )

    joblib.dump(
        calibrated_specialist,
        calibrated_model_path,
    )

    # ========================================================================
    # SAVE RESULTS
    # ========================================================================

    grid_path = (
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_threshold_grid.csv"
    )

    grid_df.to_csv(
        grid_path,
        index=False,
    )

    before_report_path = (
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_classification_report_before.csv"
    )

    after_report_path = (
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_classification_report_after.csv"
    )

    baseline["report"].to_csv(
        before_report_path
    )

    after["report"].to_csv(
        after_report_path
    )

    baseline["confusion_matrix"].to_csv(
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_confusion_before.csv"
    )

    after["confusion_matrix"].to_csv(
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_confusion_after.csv"
    )

    per_class.to_csv(
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_per_class.csv"
    )

    # ========================================================================
    # JSON SUMMARY
    # ========================================================================

    results_json = {

        "experiment": EXPERIMENT_NAME,

        "selected_policy": selected_policy,

        "selected_tau": selected_tau,

        "test_set_used": False,

        "n_features": len(feature_cols),

        "n_classes": len(class_names),

        "baseline": {

            "macro_f1": baseline["macro_f1"],

            "accuracy": baseline["accuracy"],

            "bf_f1": baseline[
                "per_class"
            ].loc[
                LABEL_BF,
                "f1",
            ],

            "xss_f1": baseline[
                "per_class"
            ].loc[
                LABEL_XSS,
                "f1",
            ],

            "bf_to_xss": bf_xss_before,

            "xss_to_bf": xss_bf_before,

            "total_mutual_confusion":
                mutual_before,
        },

        "raw_specialist_reference": {

            "macro_f1":
                raw_metrics["macro_f1"],

            "bf_to_xss":
                raw_bf_xss,

            "xss_to_bf":
                raw_xss_bf,

            "total_mutual_confusion":
                raw_bf_xss + raw_xss_bf,

            "changed":
                raw_changed,

            "fixed":
                raw_fixed,

            "damaged":
                raw_damaged,
        },

        "selected": {

            "macro_f1":
                after["macro_f1"],

            "accuracy":
                after["accuracy"],

            "bf_f1":
                after["per_class"].loc[
                    LABEL_BF,
                    "f1",
                ],

            "xss_f1":
                after["per_class"].loc[
                    LABEL_XSS,
                    "f1",
                ],

            "bf_to_xss":
                bf_xss_after,

            "xss_to_bf":
                xss_bf_after,

            "total_mutual_confusion":
                mutual_after,

            "changed":
                changed,

            "fixed":
                fixed,

            "damaged":
                damaged,
        },

        "guardrail": {

            "passed":
                guardrail_passed,

            "non_target_changes":
                non_target_changes,
        },

        "calibration": {

            "method":
                CALIBRATION_METHOD,

            "cv_folds":
                CALIBRATION_CV_FOLDS,
        },

        "threshold_grid":
            TAU_GRID,

        "selection_rule": (
            "Minimize total BF/XSS mutual confusion "
            "on validation. Tie-break by Macro-F1 "
            "and then net fixed-minus-damaged. "
            "If no threshold improves mutual confusion "
            "over the primary baseline, retain "
            "primary-only."
        ),

        "data": {

            "train_path":
                str(TRAIN_PATH),

            "validation_path":
                str(VAL_PATH),

            "test_path_used":
                None,

            "train_rows":
                int(len(y_train)),

            "validation_rows":
                int(len(y_val)),

            "specialist_train_rows":
                int(len(y_train_spec)),
        },

        "environment": {

            "python":
                sys.version.split()[0],

            "platform":
                platform.platform(),
        },

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    metrics_path = (
        RESULTS_DIR /
        f"{EXPERIMENT_NAME}_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            results_json,
            indent=2,
            default=str,
        )
    )

    log(
        f"Results saved to: "
        f"{RESULTS_DIR}"
    )

    log(
        "=== Experiment complete ==="
    )

    log(
        "TEST SET WAS NOT LOADED OR EVALUATED."
    )


if __name__ == "__main__":
    main()
