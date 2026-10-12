"""
train_hgb_bfxss_smote_tomek_specialist.py

Strategy 6: SMOTE-Tomek BF/XSS Specialist Cascade

Hypothesis
----------
The BF/XSS specialist may benefit from a more balanced training boundary.
SMOTE generates synthetic XSS samples to reduce the BF/XSS imbalance, while
Tomek Links remove selected borderline/noisy samples.

Only the BF/XSS specialist training subset is resampled.

What stays fixed
----------------
- Same train/validation parquet files
- Same 68 numeric model features
- Same preprocessing
- Same primary HGB hyperparameters
- Same BF/XSS cascade routing
- Same random seed
- Validation is never resampled
- Test set is never loaded

Strategy 6 does NOT repeat the rejected feature-selection experiment.
All 68 features are retained.

Usage
-----
(.venv) PS C:\\CyberAttack-ML>
python src\\train_hgb_bfxss_smote_tomek_specialist.py
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

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

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

EXPERIMENT_NAME = "strategy6_smote_tomek"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"
TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODELS_DIR = PROJECT_ROOT / "models" / "historical" / EXPERIMENT_NAME
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT_NAME

LABEL_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"
NON_FEATURE_COLUMNS = {LABEL_COLUMN, PROVENANCE_COLUMN}

EXPECTED_N_FEATURES = 68
RANDOM_STATE = 42

LABEL_BF = "Web Attack - Brute Force"
LABEL_XSS = "Web Attack - XSS"
LABEL_BENIGN = "BENIGN"
LABEL_DOS_HULK = "DoS Hulk"

# --------------------------------------------------------------------------- #
# Primary model
#
# EXACTLY the same configuration as the established BF/XSS specialist
# cascade baseline.
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# BF/XSS specialist
#
# Same specialist architecture as the established cascade.
# The ONLY new component is SMOTE-Tomek applied before this model is trained.
# --------------------------------------------------------------------------- #

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

# SMOTE will increase the minority XSS class to match BF.
# Original:
#   BF = 1055
#   XSS = 456
#
# Target after SMOTE, before Tomek cleaning:
#   BF = 1055
#   XSS = 1055
SMOTE_SAMPLING_STRATEGY = {
    LABEL_XSS: 1055,
}

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
            f"{name} parquet not found at {path}."
        )

    log(f"Loading {name} from {path} ...")
    df = pd.read_parquet(path)

    log(f"  -> {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
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
                f"{name} set is missing '{LABEL_COLUMN}'"
            )

    train_features = get_feature_columns(train_df)
    val_features = get_feature_columns(val_df)

    if train_features != val_features:
        raise ValueError(
            "Feature column mismatch between train and validation."
        )

    if len(train_features) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features, "
            f"found {len(train_features)}."
        )

    non_numeric = [
        c
        for c in train_features
        if not pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric model features detected: {non_numeric}"
        )

    log(
        f"Schema check passed: "
        f"{len(train_features)} model features."
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
):

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
        y_true,
        y_pred,
        labels=class_names,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(
        report_dict
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
        "per_class": per_class_df,
        "report": report_df,
        "confusion_matrix": cm_df,
    }


def confusion_count(
    cm_df: pd.DataFrame,
    true_label: str,
    pred_label: str,
) -> int:

    if (
        true_label not in cm_df.index
        or pred_label not in cm_df.columns
    ):
        return 0

    return int(
        cm_df.loc[
            true_label,
            pred_label,
        ]
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

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
        f"=== Experiment: {EXPERIMENT_NAME} ==="
    )

    log(
        f"Python {sys.version.split()[0]} | "
        f"platform {platform.platform()}"
    )

    # ------------------------------------------------------------------ #
    # Load train + validation ONLY
    # ------------------------------------------------------------------ #

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

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        LABEL_COLUMN
    ].astype(str)

    X_val = val_df[
        feature_cols
    ]

    y_val = val_df[
        LABEL_COLUMN
    ].astype(str)

    class_names = sorted(
        y_train.unique()
    )

    log(
        f"Classes ({len(class_names)}): "
        f"{class_names}"
    )

    del train_df
    del val_df

    # ------------------------------------------------------------------ #
    # Stage 1: Primary model
    # ------------------------------------------------------------------ #

    log(
        "--- Stage 1: training primary model ---"
    )

    primary_model = (
        HistGradientBoostingClassifier(
            **PRIMARY_MODEL_PARAMS
        )
    )

    primary_train_start = time.perf_counter()

    primary_model.fit(
        X_train,
        y_train,
    )

    primary_train_time_sec = (
        time.perf_counter()
        - primary_train_start
    )

    log(
        f"Primary model trained in "
        f"{primary_train_time_sec:.2f}s "
        f"({getattr(primary_model, 'n_iter_', 'unknown')} "
        f"boosting iterations)"
    )

    primary_predict_start = (
        time.perf_counter()
    )

    y_pred_primary = (
        primary_model.predict(X_val)
    )

    primary_predict_time_sec = (
        time.perf_counter()
        - primary_predict_start
    )

    log(
        f"Primary validation prediction "
        f"complete in "
        f"{primary_predict_time_sec:.2f}s"
    )

    before = compute_metrics(
        y_val,
        y_pred_primary,
        class_names,
    )

    log(
        f"[PRIMARY] Macro F1 = "
        f"{before['macro_f1']:.4%}, "
        f"Accuracy = "
        f"{before['accuracy']:.4%}"
    )

    # ------------------------------------------------------------------ #
    # Stage 2: Extract BF/XSS training subset
    # ------------------------------------------------------------------ #

    log(
        "--- Stage 2: preparing BF/XSS "
        "specialist training data ---"
    )

    specialist_train_mask = y_train.isin(
        [LABEL_BF, LABEL_XSS]
    )

    X_train_specialist = (
        X_train.loc[
            specialist_train_mask
        ].copy()
    )

    y_train_specialist = (
        y_train.loc[
            specialist_train_mask
        ].copy()
    )

    bf_before = int(
        (y_train_specialist == LABEL_BF).sum()
    )

    xss_before = int(
        (y_train_specialist == LABEL_XSS).sum()
    )

    log(
        f"Original specialist data: "
        f"{len(y_train_specialist):,} rows"
    )

    log(
        f"  BF  = {bf_before:,}"
    )

    log(
        f"  XSS = {xss_before:,}"
    )

    # ------------------------------------------------------------------ #
    # Stage 3: SMOTE-Tomek
    # ------------------------------------------------------------------ #

    log(
        "--- Stage 3: applying SMOTE-Tomek "
        "to BF/XSS training data ONLY ---"
    )

    log(
        f"SMOTE target: XSS -> "
        f"{SMOTE_SAMPLING_STRATEGY[LABEL_XSS]:,}"
    )

    log(
        "Validation data is NOT passed through "
        "SMOTE-Tomek."
    )

    log(
        "Test data is NOT loaded."
    )

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    smote_tomek = SMOTETomek(
        smote=smote,
        random_state=RANDOM_STATE,
    )

    resample_start = time.perf_counter()

    X_specialist_resampled, y_specialist_resampled = (
        smote_tomek.fit_resample(
            X_train_specialist,
            y_train_specialist,
        )
    )

    resample_time_sec = (
        time.perf_counter()
        - resample_start
    )

    bf_after = int(
        (y_specialist_resampled == LABEL_BF).sum()
    )

    xss_after = int(
        (y_specialist_resampled == LABEL_XSS).sum()
    )

    log(
        f"SMOTE-Tomek completed in "
        f"{resample_time_sec:.2f}s"
    )

    log(
        f"After SMOTE-Tomek: "
        f"{len(y_specialist_resampled):,} rows"
    )

    log(
        f"  BF  = {bf_after:,}"
    )

    log(
        f"  XSS = {xss_after:,}"
    )

    log(
        f"  Removed/added net rows: "
        f"{len(y_specialist_resampled) - len(y_train_specialist):+,}"
    )

    # ------------------------------------------------------------------ #
    # Stage 4: Train specialist
    # ------------------------------------------------------------------ #

    log(
        "--- Stage 4: training "
        "SMOTE-Tomek BF/XSS specialist ---"
    )

    specialist_model = (
        HistGradientBoostingClassifier(
            **SPECIALIST_MODEL_PARAMS
        )
    )

    specialist_train_start = (
        time.perf_counter()
    )

    specialist_model.fit(
        X_specialist_resampled,
        y_specialist_resampled,
    )

    specialist_train_time_sec = (
        time.perf_counter()
        - specialist_train_start
    )

    log(
        f"Specialist trained in "
        f"{specialist_train_time_sec:.2f}s "
        f"({getattr(specialist_model, 'n_iter_', 'unknown')} "
        f"boosting iterations)"
    )

    # ------------------------------------------------------------------ #
    # Stage 5: Cascade routing
    # ------------------------------------------------------------------ #

    log(
        "--- Stage 5: applying BF/XSS cascade ---"
    )

    route_mask = np.isin(
        y_pred_primary,
        [LABEL_BF, LABEL_XSS],
    )

    n_routed = int(
        route_mask.sum()
    )

    log(
        f"Rows routed to specialist: "
        f"{n_routed:,}"
    )

    y_pred_cascade = (
        y_pred_primary.copy()
    )

    specialist_predict_start = (
        time.perf_counter()
    )

    if n_routed > 0:

        specialist_preds = (
            specialist_model.predict(
                X_val.loc[route_mask]
            )
        )

        y_pred_cascade[
            route_mask
        ] = specialist_preds

        n_flipped = int(
            (
                specialist_preds
                != y_pred_primary[route_mask]
            ).sum()
        )

    else:

        specialist_preds = np.array(
            [],
            dtype=str,
        )

        n_flipped = 0

    specialist_predict_time_sec = (
        time.perf_counter()
        - specialist_predict_start
    )

    log(
        f"Specialist prediction complete "
        f"in {specialist_predict_time_sec:.2f}s"
    )

    log(
        f"Changed predictions: "
        f"{n_flipped:,}"
    )

    # ------------------------------------------------------------------ #
    # Stage 6: Final metrics
    # ------------------------------------------------------------------ #

    after = compute_metrics(
        y_val,
        y_pred_cascade,
        class_names,
    )

    log(
        f"[FINAL] Macro F1 = "
        f"{after['macro_f1']:.4%}, "
        f"Accuracy = "
        f"{after['accuracy']:.4%}"
    )

    # ------------------------------------------------------------------ #
    # Guardrail
    # ------------------------------------------------------------------ #

    log(
        "--- Guardrail: non-BF/XSS classes ---"
    )

    guardrail_violations = []

    for cls in class_names:

        if cls in (
            LABEL_BF,
            LABEL_XSS,
        ):
            continue

        f1_before = (
            before["per_class"]
            .loc[cls, "f1"]
        )

        f1_after = (
            after["per_class"]
            .loc[cls, "f1"]
        )

        delta = (
            f1_after
            - f1_before
        )

        if abs(delta) > GUARDRAIL_F1_DELTA_THRESHOLD:

            guardrail_violations.append(
                (
                    cls,
                    f1_before,
                    f1_after,
                    delta,
                )
            )

    guardrail_passed = (
        len(guardrail_violations) == 0
    )

    if guardrail_passed:

        log(
            "PASSED: all non-BF/XSS "
            "classes are unchanged."
        )

    else:

        log(
            "WARNING: guardrail violations detected."
        )

        for (
            cls,
            f1_b,
            f1_a,
            delta,
        ) in guardrail_violations:

            log(
                f"  {cls}: "
                f"{f1_b:.4%} -> "
                f"{f1_a:.4%} "
                f"({delta:+.4%})"
            )

    # ------------------------------------------------------------------ #
    # BF/XSS confusion
    # ------------------------------------------------------------------ #

    bf_to_xss_before = confusion_count(
        before["confusion_matrix"],
        LABEL_BF,
        LABEL_XSS,
    )

    xss_to_bf_before = confusion_count(
        before["confusion_matrix"],
        LABEL_XSS,
        LABEL_BF,
    )

    bf_to_xss_after = confusion_count(
        after["confusion_matrix"],
        LABEL_BF,
        LABEL_XSS,
    )

    xss_to_bf_after = confusion_count(
        after["confusion_matrix"],
        LABEL_XSS,
        LABEL_BF,
    )

    mutual_before = (
        bf_to_xss_before
        + xss_to_bf_before
    )

    mutual_after = (
        bf_to_xss_after
        + xss_to_bf_after
    )

    # ------------------------------------------------------------------ #
    # Additional diagnostics
    # ------------------------------------------------------------------ #

    changed_mask = (
        y_pred_primary
        != y_pred_cascade
    )

    n_changed = int(
        changed_mask.sum()
    )

    fixed = int(
        (
            changed_mask
            & (
                y_pred_cascade
                == y_val.to_numpy()
            )
        ).sum()
    )

    damaged = int(
        (
            changed_mask
            & (
                y_pred_primary
                == y_val.to_numpy()
            )
            & (
                y_pred_cascade
                != y_val.to_numpy()
            )
        ).sum()
    )

    # ------------------------------------------------------------------ #
    # Summary table
    # ------------------------------------------------------------------ #

    comparison = pd.DataFrame(
        {
            "before": [
                before["macro_f1"],
                before["weighted_f1"],
                before["accuracy"],
                bf_to_xss_before,
                xss_to_bf_before,
                mutual_before,
            ],
            "after": [
                after["macro_f1"],
                after["weighted_f1"],
                after["accuracy"],
                bf_to_xss_after,
                xss_to_bf_after,
                mutual_after,
            ],
        },
        index=[
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "BF->XSS",
            "XSS->BF",
            "BF/XSS mutual confusion",
        ],
    )

    comparison["delta"] = (
        comparison["after"]
        - comparison["before"]
    )

    per_class_comparison = (
        before["per_class"][
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

    per_class_comparison["delta"] = (
        per_class_comparison["f1_after"]
        - per_class_comparison["f1_before"]
    )

    per_class_comparison = (
        per_class_comparison
        .sort_values(
            "support",
            ascending=False,
        )
    )

    log(
        "=== BEFORE / AFTER ==="
    )

    print(
        comparison.to_string(
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    log(
        f"Changed predictions: {n_changed:,}"
    )

    log(
        f"Fixed: {fixed:,}"
    )

    log(
        f"Damaged: {damaged:,}"
    )

    # ------------------------------------------------------------------ #
    # Save models
    # ------------------------------------------------------------------ #

    primary_model_path = (
        MODELS_DIR
        / "primary_model.joblib"
    )

    specialist_model_path = (
        MODELS_DIR
        / "smote_tomek_specialist.joblib"
    )

    joblib.dump(
        primary_model,
        primary_model_path,
    )

    joblib.dump(
        specialist_model,
        specialist_model_path,
    )

    log(
        f"Saved primary model -> "
        f"{primary_model_path}"
    )

    log(
        f"Saved specialist model -> "
        f"{specialist_model_path}"
    )

    # ------------------------------------------------------------------ #
    # Save reports
    # ------------------------------------------------------------------ #

    before["report"].to_csv(
        RESULTS_DIR
        / "classification_report_before.csv"
    )

    after["report"].to_csv(
        RESULTS_DIR
        / "classification_report_after.csv"
    )

    before["confusion_matrix"].to_csv(
        RESULTS_DIR
        / "confusion_matrix_before.csv"
    )

    after["confusion_matrix"].to_csv(
        RESULTS_DIR
        / "confusion_matrix_after.csv"
    )

    comparison.to_csv(
        RESULTS_DIR
        / "before_after_comparison.csv"
    )

    per_class_comparison.to_csv(
        RESULTS_DIR
        / "per_class_comparison.csv"
    )

    # ------------------------------------------------------------------ #
    # Save metrics JSON
    # ------------------------------------------------------------------ #

    metrics = {

        "experiment": EXPERIMENT_NAME,

        "data": {
            "train_path": str(TRAIN_PATH),
            "validation_path": str(VAL_PATH),
            "test_path_used": None,
            "n_train_rows": int(len(y_train)),
            "n_validation_rows": int(len(y_val)),
            "n_features": len(feature_cols),
            "n_classes": len(class_names),
        },

        "specialist_original_distribution": {
            "BF": bf_before,
            "XSS": xss_before,
            "total": bf_before + xss_before,
        },

        "smote_tomek_distribution": {
            "BF": bf_after,
            "XSS": xss_after,
            "total": int(
                len(y_specialist_resampled)
            ),
        },

        "smote_sampling_strategy": {
            LABEL_XSS: SMOTE_SAMPLING_STRATEGY[
                LABEL_XSS
            ]
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

        "delta": {
            "macro_f1": (
                after["macro_f1"]
                - before["macro_f1"]
            ),
            "weighted_f1": (
                after["weighted_f1"]
                - before["weighted_f1"]
            ),
            "accuracy": (
                after["accuracy"]
                - before["accuracy"]
            ),
        },

        "bf_xss_confusion": {
            "BF_to_XSS": {
                "before": bf_to_xss_before,
                "after": bf_to_xss_after,
                "delta": (
                    bf_to_xss_after
                    - bf_to_xss_before
                ),
            },
            "XSS_to_BF": {
                "before": xss_to_bf_before,
                "after": xss_to_bf_after,
                "delta": (
                    xss_to_bf_after
                    - xss_to_bf_before
                ),
            },
            "mutual": {
                "before": mutual_before,
                "after": mutual_after,
                "delta": (
                    mutual_after
                    - mutual_before
                ),
            },
        },

        "cascade": {
            "n_routed": n_routed,
            "n_changed": n_changed,
            "fixed": fixed,
            "damaged": damaged,
        },

        "guardrail": {
            "passed": guardrail_passed,
            "violations": [
                {
                    "class": cls,
                    "f1_before": f1_b,
                    "f1_after": f1_a,
                    "delta": delta,
                }
                for (
                    cls,
                    f1_b,
                    f1_a,
                    delta,
                ) in guardrail_violations
            ],
        },

        "timing": {
            "primary_train_sec":
                primary_train_time_sec,
            "primary_predict_sec":
                primary_predict_time_sec,
            "smote_tomek_sec":
                resample_time_sec,
            "specialist_train_sec":
                specialist_train_time_sec,
            "specialist_predict_sec":
                specialist_predict_time_sec,
        },

        "environment": {
            "python_version":
                sys.version.split()[0],
            "platform":
                platform.platform(),
        },
    }

    metrics_path = (
        RESULTS_DIR
        / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
            default=str,
        )
    )

    # ------------------------------------------------------------------ #
    # Save full training summary
    # ------------------------------------------------------------------ #

    summary = {

        "experiment":
            EXPERIMENT_NAME,

        "timestamp_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "hypothesis": (
            "Balancing the BF/XSS specialist training "
            "distribution with SMOTE and cleaning borderline "
            "samples with Tomek Links may improve the "
            "BF/XSS decision boundary."
        ),

        "changed": [
            "SMOTE applied to BF/XSS specialist training data.",
            "Tomek Links applied after SMOTE.",
            "XSS oversampled toward the BF training count.",
        ],

        "unchanged": [
            "Primary HGB architecture.",
            "Primary HGB hyperparameters.",
            "BF/XSS specialist HGB architecture.",
            "68 model features.",
            "Train/validation split.",
            "Preprocessing.",
            "Cascade routing rule.",
            "Random seed.",
        ],

        "routing_rule": (
            "Only validation rows whose primary prediction "
            "is BF or XSS are passed to the specialist."
        ),

        "test_set": {
            "loaded": False,
            "evaluated": False,
        },

        "primary_model_params": {
            k: str(v)
            for k, v in PRIMARY_MODEL_PARAMS.items()
        },

        "specialist_model_params": {
            k: str(v)
            for k, v in SPECIALIST_MODEL_PARAMS.items()
        },

        "smote_tomek": {
            "sampling_strategy":
                SMOTE_SAMPLING_STRATEGY,
            "k_neighbors": 5,
            "random_state":
                RANDOM_STATE,
        },

        "guardrail": {
            "passed":
                guardrail_passed,
            "description": (
                "Non-BF/XSS classes must remain unchanged "
                "because the cascade only modifies predictions "
                "already classified by the primary model as "
                "BF or XSS."
            ),
        },

        "metrics_before": {
            "accuracy":
                before["accuracy"],
            "macro_f1":
                before["macro_f1"],
            "weighted_f1":
                before["weighted_f1"],
        },

        "metrics_after": {
            "accuracy":
                after["accuracy"],
            "macro_f1":
                after["macro_f1"],
            "weighted_f1":
                after["weighted_f1"],
        },
    }

    summary_path = (
        RESULTS_DIR
        / "training_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            default=str,
        )
    )

    log(
        f"Saved results -> "
        f"{RESULTS_DIR}"
    )

    log(
        "=== Strategy 6 complete. "
        "Test set was NOT loaded. ==="
    )


if __name__ == "__main__":
    main()