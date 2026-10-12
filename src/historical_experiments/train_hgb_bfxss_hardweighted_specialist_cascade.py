"""
train_hgb_bfxss_hardweighted_specialist_cascade.py

Strategy 4 -- Hard-example-weighted BF/XSS specialist cascade.

Purpose
-------
Test whether explicitly up-weighting BF/XSS examples that a leak-free
out-of-fold specialist finds difficult can improve BF/XSS separation.

Previous experiments
--------------------
Strategy 1:
    Primary HGB baseline
    Macro-F1 = 85.8601%
    Accuracy = 99.4470%
    BF F1 = 0.7383
    XSS F1 = 0.4464
    BF->XSS = 60
    XSS->BF = 48
    Mutual confusion = 108

Strategy 2:
    Unconditional BF/XSS specialist
    Macro-F1 = 86.3277%
    Mutual confusion = 106
    Rejected because the targeted improvement was negligible and the
    specialist damaged correct BF predictions.

Strategy 3:
    Calibrated confidence-gated specialist
    Best mutual-confusion configuration:
        tau = 0.55
        Macro-F1 = 84.77%
        BF->XSS = 3
        XSS->BF = 85
        Mutual confusion = 88
    Rejected because the apparent reduction in mutual confusion came
    from strongly shifting errors toward XSS->BF and XSS F1 collapsed.

Strategy 4 hypothesis
---------------------
The specialist may benefit from explicitly emphasizing BF/XSS examples
that are difficult for a specialist itself.

Procedure
---------
1. Load the LOCKED train/validation datasets.
2. Train the primary model using the exact Strategy 1 parameters.
3. On BF/XSS training rows only, perform 5-fold OOF specialist predictions.
4. Mark rows as hard if:
       - the OOF specialist misclassified them, OR
       - OOF confidence margin is below HARD_MARGIN_THRESHOLD.
5. Build balanced class weights.
6. Multiply hard-example weights by HARD_WEIGHT_MULTIPLIER.
7. Train the specialist on ALL BF/XSS training rows using those weights.
8. Calibrate the specialist using train-only cross-validation.
9. Grid-search hard-example multiplier x confidence threshold on validation.
10. Selection rule:
       minimize BF/XSS mutual confusion first,
       then maximize Macro-F1.
11. Only validation rows whose PRIMARY prediction is BF/XSS can be changed.
12. Test data is NEVER loaded.

IMPORTANT
---------
This script uses:
    data/processed/preprocessed/train.parquet
    data/processed/preprocessed/validation.parquet

It does NOT use:
    data/processed/cicids2017_training_ready.parquet
    data/splits/row_manifest.parquet

Expected model features: 68
Target column: Label

Actual BF/XSS labels:
    Web Attack - Brute Force
    Web Attack - XSS
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


# =============================================================================
# PROJECT PATHS -- LOCKED PIPELINE
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"

TRAIN_PATH = DATA_DIR / "train.parquet"
VAL_PATH = DATA_DIR / "validation.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS_DIR / "strategy4_hardweighted_specialist_cascade"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# LOCKED SCHEMA
# =============================================================================

LABEL_COL = "Label"
PROVENANCE_COL = "source_file"

EXPECTED_N_FEATURES = 68

BF_CLASS = "Web Attack - Brute Force"
XSS_CLASS = "Web Attack - XSS"

RANDOM_STATE = 42


# =============================================================================
# EXACT PRIMARY MODEL PARAMETERS FROM STRATEGY 1
# =============================================================================

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


# =============================================================================
# EXACT BF/XSS SPECIALIST PARAMETERS FROM STRATEGY 3
#
# These parameters are deliberately preserved so that hard-weight multiplier
# 1.0 represents the same specialist learning setup used previously.
#
# Balanced class weighting is represented explicitly through sample_weight
# because we need to multiply it by the hard-example multiplier.
# =============================================================================

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
    verbose=1,
)


# =============================================================================
# STRATEGY 4 HYPERPARAMETERS
# =============================================================================

N_OOF_FOLDS = 5

# OOF confidence margin:
#   0.00 = completely uncertain
#   1.00 = completely confident
#
# A row below this threshold is considered difficult.
HARD_MARGIN_THRESHOLD = 0.20

# Multiplier applied ONLY to hard examples.
#
# 1.0 = no additional hard-example weighting.
# 2.0 = hard examples receive 2x their normal balanced class weight.
# etc.
HARD_WEIGHT_GRID = [1.0, 2.0, 3.0, 5.0, 8.0]

# Same confidence grid used in the previous gated experiment.
TAU_GRID = np.round(np.arange(0.50, 0.96, 0.05), 2)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_verify(path):
    print(f"\nLoading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{path}\n\n"
            f"Expected current project structure:\n"
            f"{PROJECT_ROOT / 'data' / 'processed' / 'preprocessed' / 'train.parquet'}\n"
            f"{PROJECT_ROOT / 'data' / 'processed' / 'preprocessed' / 'validation.parquet'}"
        )

    df = pd.read_parquet(path)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"{path} does not contain required label column '{LABEL_COL}'. "
            f"Columns found: {list(df.columns)}"
        )

    excluded = {LABEL_COL, PROVENANCE_COL}

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    if len(feature_cols) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"{path}: expected {EXPECTED_N_FEATURES} model features, "
            f"found {len(feature_cols)}."
        )

    return df, feature_cols


def verify_matching_schema(train_df, val_df, feature_cols):
    train_features = [
        c for c in train_df.columns
        if c not in {LABEL_COL, PROVENANCE_COL}
    ]

    val_features = [
        c for c in val_df.columns
        if c not in {LABEL_COL, PROVENANCE_COL}
    ]

    if train_features != val_features:
        raise ValueError(
            "Train and validation feature schemas do not match."
        )

    if train_features != feature_cols:
        raise ValueError(
            "Feature schema verification failed."
        )


# =============================================================================
# PRIMARY MODEL
# =============================================================================

def train_primary(X_train, y_train):
    print("\nTraining primary HGB...")

    model = HistGradientBoostingClassifier(
        **PRIMARY_MODEL_PARAMS
    )

    model.fit(X_train, y_train)

    return model


# =============================================================================
# OOF HARD-EXAMPLE IDENTIFICATION
# =============================================================================

def identify_hard_examples_oof(
    X_bx,
    y_bx_bin,
    n_folds,
    margin_threshold,
    seed,
):
    """
    Identify difficult BF/XSS training examples without leakage.

    Each row is predicted by an OOF specialist that did NOT train on that row.

    A row is hard if:
        1. It is misclassified, OR
        2. Its confidence margin is below margin_threshold.

    y_bx_bin:
        0 = Web Attack - Brute Force
        1 = Web Attack - XSS
    """

    n = len(y_bx_bin)

    oof_pred = np.empty(n, dtype=np.int8)
    oof_margin = np.empty(n, dtype=np.float64)

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    print(
        f"\nRunning {n_folds}-fold OOF hard-example detection..."
    )

    for fold_idx, (fit_idx, hold_idx) in enumerate(
        skf.split(X_bx, y_bx_bin),
        start=1,
    ):
        print(
            f"  OOF fold {fold_idx}/{n_folds}: "
            f"train={len(fit_idx)}, holdout={len(hold_idx)}"
        )

        fold_params = SPECIALIST_MODEL_PARAMS.copy()
        fold_params["random_state"] = seed + fold_idx

        # For OOF detection we use ordinary balanced class weighting.
        fold_params["class_weight"] = "balanced"

        fold_model = HistGradientBoostingClassifier(
            **fold_params
        )

        fold_model.fit(
            X_bx.iloc[fit_idx],
            y_bx_bin[fit_idx],
        )

        proba = fold_model.predict_proba(
            X_bx.iloc[hold_idx]
        )[:, 1]

        oof_pred[hold_idx] = (
            proba >= 0.5
        ).astype(np.int8)

        oof_margin[hold_idx] = (
            np.abs(proba - 0.5) * 2.0
        )

    misclassified = oof_pred != y_bx_bin

    low_margin = (
        oof_margin < margin_threshold
    )

    hard_mask = (
        misclassified |
        low_margin
    )

    additional_low_margin = (
        low_margin &
        ~misclassified
    )

    print(
        "\nOOF hard-example results:"
    )

    print(
        f"  Total BF/XSS rows: {n}"
    )

    print(
        f"  Misclassified OOF rows: {int(misclassified.sum())}"
    )

    print(
        f"  Additional low-margin rows: "
        f"{int(additional_low_margin.sum())}"
    )

    print(
        f"  Total hard rows: {int(hard_mask.sum())}"
    )

    print(
        f"  Hard percentage: "
        f"{100.0 * hard_mask.mean():.2f}%"
    )

    return hard_mask


# =============================================================================
# SAMPLE WEIGHTS
# =============================================================================

def build_sample_weights(
    y_bx_bin,
    hard_mask,
    hard_multiplier,
):
    """
    Build:

        final_weight =
            balanced_class_weight
            *
            hard_example_multiplier

    for hard examples.

    Easy examples retain their normal balanced class weight.
    """

    base_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_bx_bin,
    )

    final_weight = base_weight.copy()

    final_weight[hard_mask] *= hard_multiplier

    return final_weight


# =============================================================================
# WEIGHT-CHECKED HGB
# =============================================================================

class WeightCheckedHGB(
    HistGradientBoostingClassifier
):
    """
    Thin wrapper used only to verify that the exact sample_weight vector
    constructed by Strategy 4 reaches HistGradientBoostingClassifier.fit().
    """

    def fit(
        self,
        X,
        y,
        sample_weight=None,
    ):
        if sample_weight is None:
            raise AssertionError(
                "Strategy 4 requires sample_weight."
            )

        self.last_fit_sample_weight_ = np.asarray(
            sample_weight
        ).copy()

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
        )


# =============================================================================
# TRAIN HARD-WEIGHTED SPECIALIST
# =============================================================================

def train_hard_weighted_specialist(
    X_bx,
    y_bx_bin,
    sample_weight,
    seed,
):
    params = SPECIALIST_MODEL_PARAMS.copy()

    # We explicitly provide balanced class weighting ourselves through
    # sample_weight so the hard multiplier can be applied multiplicatively.
    #
    # Using class_weight="balanced" simultaneously would double-apply
    # the class balancing.
    params.pop("class_weight", None)

    params["random_state"] = seed

    model = WeightCheckedHGB(
        **params
    )

    model.fit(
        X_bx,
        y_bx_bin,
        sample_weight=sample_weight,
    )

    # Critical plumbing assertion.
    if not np.allclose(
        model.last_fit_sample_weight_,
        sample_weight,
    ):
        raise AssertionError(
            "The sample weights reaching HGB.fit() do not match "
            "the computed hard-example weights."
        )

    return model


# =============================================================================
# CALIBRATED SPECIALIST
# =============================================================================

def calibrate_specialist(
    X_bx,
    y_bx_bin,
    sample_weight,
    seed,
):
    """
    Train the exact Strategy-3-style specialist architecture,
    but with the Strategy-4 sample weights.

    Calibration uses 5-fold cross-validation over TRAIN ONLY.
    Validation/test are never used during fitting.
    """

    params = SPECIALIST_MODEL_PARAMS.copy()

    # Again, balanced class weighting is already encoded in sample_weight.
    params.pop("class_weight", None)

    params["random_state"] = seed

    base = HistGradientBoostingClassifier(
        **params
    )

    calibrated = CalibratedClassifierCV(
        base,
        method="sigmoid",
        cv=5,
    )

    calibrated.fit(
        X_bx,
        y_bx_bin,
        sample_weight=sample_weight,
    )

    return calibrated


# =============================================================================
# CASCADE APPLICATION
# =============================================================================

def apply_cascade(
    primary_pred,
    X_val,
    eligible_mask,
    calibrated_specialist,
    tau,
):
    """
    Only primary BF/XSS predictions are eligible.

    Every other primary prediction remains untouched.
    """

    final_pred = primary_pred.copy()

    if not eligible_mask.any():
        return final_pred

    eligible_X = X_val.loc[
        eligible_mask
    ]

    proba = calibrated_specialist.predict_proba(
        eligible_X
    )

    confidence = proba.max(axis=1)

    specialist_pred_bin = (
        proba.argmax(axis=1)
    )

    specialist_pred_label = np.where(
        specialist_pred_bin == 1,
        XSS_CLASS,
        BF_CLASS,
    )

    override = (
        confidence >= tau
    )

    eligible_positions = np.flatnonzero(
        eligible_mask.to_numpy()
    )

    final_pred[
        eligible_positions[override]
    ] = specialist_pred_label[override]

    return final_pred


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(
    y_true,
    y_pred,
    primary_pred,
):
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    labels = sorted(
        pd.unique(y_true)
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    per_class = {}

    for (
        label,
        p,
        r,
        f,
        s,
    ) in zip(
        labels,
        precision,
        recall,
        f1,
        support,
    ):
        per_class[label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }

    bf_to_xss = int(
        np.sum(
            (y_true == BF_CLASS) &
            (y_pred == XSS_CLASS)
        )
    )

    xss_to_bf = int(
        np.sum(
            (y_true == XSS_CLASS) &
            (y_pred == BF_CLASS)
        )
    )

    mutual_confusion = (
        bf_to_xss +
        xss_to_bf
    )

    # Construction-level guardrail:
    # anything that the primary did NOT call BF/XSS must remain identical.
    untouched_mask = ~np.isin(
        primary_pred,
        [BF_CLASS, XSS_CLASS],
    )

    guardrail_ok = bool(
        np.array_equal(
            y_pred[untouched_mask],
            primary_pred[untouched_mask],
        )
    )

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "bf_f1": per_class.get(
            BF_CLASS,
            {},
        ).get("f1"),
        "xss_f1": per_class.get(
            XSS_CLASS,
            {},
        ).get("f1"),
        "bf_to_xss": bf_to_xss,
        "xss_to_bf": xss_to_bf,
        "mutual_confusion": mutual_confusion,
        "guardrail_ok": guardrail_ok,
        "per_class": per_class,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STRATEGY 4 -- HARD-EXAMPLE-WEIGHTED BF/XSS SPECIALIST")
    print("=" * 80)

    print("\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print("\nTraining data:")
    print(f"  {TRAIN_PATH}")

    print("\nValidation data:")
    print(f"  {VAL_PATH}")

    print("\nIMPORTANT: test data is NOT loaded by this script.")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    train_df, feature_cols = load_and_verify(
        TRAIN_PATH
    )

    val_df, val_feature_cols = load_and_verify(
        VAL_PATH
    )

    verify_matching_schema(
        train_df,
        val_df,
        feature_cols,
    )

    if feature_cols != val_feature_cols:
        raise ValueError(
            "Train/validation feature ordering differs."
        )

    print(
        f"\nTrain shape: {train_df.shape}"
    )

    print(
        f"Validation shape: {val_df.shape}"
    )

    print(
        f"Number of model features: {len(feature_cols)}"
    )

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        LABEL_COL
    ].values

    X_val = val_df[
        feature_cols
    ]

    y_val = val_df[
        LABEL_COL
    ].values

    # -------------------------------------------------------------------------
    # VERIFY ACTUAL BF/XSS LABELS EXIST
    # -------------------------------------------------------------------------

    train_labels = set(
        train_df[LABEL_COL].unique()
    )

    if BF_CLASS not in train_labels:
        raise ValueError(
            f"BF class '{BF_CLASS}' was not found in training labels."
        )

    if XSS_CLASS not in train_labels:
        raise ValueError(
            f"XSS class '{XSS_CLASS}' was not found in training labels."
        )

    # -------------------------------------------------------------------------
    # PRIMARY BASELINE
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "TRAINING PRIMARY MODEL -- STRATEGY 1 BASELINE"
    )

    print(
        "=" * 80
    )

    primary = train_primary(
        X_train,
        y_train,
    )

    primary_val_pred = primary.predict(
        X_val
    )

    baseline_metrics = evaluate(
        y_val,
        primary_val_pred,
        primary_val_pred,
    )

    print(
        "\nBASELINE RESULTS"
    )

    print(
        f"  Macro F1: "
        f"{baseline_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"  Accuracy: "
        f"{baseline_metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"  BF F1: "
        f"{baseline_metrics['bf_f1']:.4f}"
    )

    print(
        f"  XSS F1: "
        f"{baseline_metrics['xss_f1']:.4f}"
    )

    print(
        f"  BF -> XSS: "
        f"{baseline_metrics['bf_to_xss']}"
    )

    print(
        f"  XSS -> BF: "
        f"{baseline_metrics['xss_to_bf']}"
    )

    print(
        f"  Mutual confusion: "
        f"{baseline_metrics['mutual_confusion']}"
    )

    # -------------------------------------------------------------------------
    # BF/XSS TRAINING SUBSET
    # -------------------------------------------------------------------------

    bx_train_mask = train_df[
        LABEL_COL
    ].isin(
        [BF_CLASS, XSS_CLASS]
    )

    X_bx = X_train.loc[
        bx_train_mask
    ].reset_index(
        drop=True
    )

    y_bx_labels = train_df.loc[
        bx_train_mask,
        LABEL_COL,
    ].values

    # 0 = BF
    # 1 = XSS
    y_bx_bin = (
        y_bx_labels == XSS_CLASS
    ).astype(
        np.int8
    )

    print(
        "\nBF/XSS specialist training subset:"
    )

    print(
        f"  Total: {len(y_bx_bin)}"
    )

    print(
        f"  {BF_CLASS}: "
        f"{int((y_bx_bin == 0).sum())}"
    )

    print(
        f"  {XSS_CLASS}: "
        f"{int((y_bx_bin == 1).sum())}"
    )

    # -------------------------------------------------------------------------
    # OOF HARD EXAMPLES
    # -------------------------------------------------------------------------

    hard_mask = identify_hard_examples_oof(
        X_bx=X_bx,
        y_bx_bin=y_bx_bin,
        n_folds=N_OOF_FOLDS,
        margin_threshold=HARD_MARGIN_THRESHOLD,
        seed=RANDOM_STATE,
    )

    # -------------------------------------------------------------------------
    # CASCADE ELIGIBILITY
    # -------------------------------------------------------------------------

    eligible_mask = pd.Series(
        np.isin(
            primary_val_pred,
            [BF_CLASS, XSS_CLASS],
        ),
        index=X_val.index,
    )

    print(
        "\nValidation cascade eligibility:"
    )

    print(
        f"  Rows eligible: "
        f"{int(eligible_mask.sum())}"
    )

    print(
        f"  Total validation rows: "
        f"{len(X_val)}"
    )

    # -------------------------------------------------------------------------
    # GRID SEARCH
    # -------------------------------------------------------------------------

    grid_results = []

    best = None

    for hard_multiplier in HARD_WEIGHT_GRID:

        print(
            "\n" + "=" * 80
        )

        print(
            f"HARD WEIGHT MULTIPLIER = "
            f"{hard_multiplier}"
        )

        print(
            "=" * 80
        )

        sample_weight = build_sample_weights(
            y_bx_bin=y_bx_bin,
            hard_mask=hard_mask,
            hard_multiplier=hard_multiplier,
        )

        # ---------------------------------------------------------------------
        # multiplier=1.0 sanity check
        # ---------------------------------------------------------------------

        if hard_multiplier == 1.0:

            plain_balanced_weight = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_bx_bin,
                )
            )

            if not np.allclose(
                sample_weight,
                plain_balanced_weight,
            ):
                raise AssertionError(
                    "Multiplier=1.0 is not equivalent to normal balanced "
                    "sample weights."
                )

            print(
                "  Multiplier 1.0 confirmed: "
                "no hard-example boost."
            )

        # ---------------------------------------------------------------------
        # TRAIN UNCALIBRATED SPECIALIST
        #
        # This is also the explicit plumbing verification.
        # ---------------------------------------------------------------------

        print(
            "\nTraining hard-weighted specialist..."
        )

        _ = train_hard_weighted_specialist(
            X_bx=X_bx,
            y_bx_bin=y_bx_bin,
            sample_weight=sample_weight,
            seed=RANDOM_STATE,
        )

        # ---------------------------------------------------------------------
        # CALIBRATE
        # ---------------------------------------------------------------------

        print(
            "\nCalibrating specialist..."
        )

        calibrated_specialist = calibrate_specialist(
            X_bx=X_bx,
            y_bx_bin=y_bx_bin,
            sample_weight=sample_weight,
            seed=RANDOM_STATE,
        )

        # ---------------------------------------------------------------------
        # TAU GRID
        # ---------------------------------------------------------------------

        for tau in TAU_GRID:

            final_pred = apply_cascade(
                primary_pred=primary_val_pred,
                X_val=X_val,
                eligible_mask=eligible_mask,
                calibrated_specialist=calibrated_specialist,
                tau=tau,
            )

            metrics = evaluate(
                y_true=y_val,
                y_pred=final_pred,
                primary_pred=primary_val_pred,
            )

            if not metrics["guardrail_ok"]:
                raise AssertionError(
                    f"GUARDRAIL FAILED at "
                    f"multiplier={hard_multiplier}, "
                    f"tau={tau}"
                )

            row = {
                "hard_weight_multiplier": float(
                    hard_multiplier
                ),
                "tau": float(tau),
                "macro_f1": metrics[
                    "macro_f1"
                ],
                "weighted_f1": metrics[
                    "weighted_f1"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "bf_f1": metrics[
                    "bf_f1"
                ],
                "xss_f1": metrics[
                    "xss_f1"
                ],
                "bf_to_xss": metrics[
                    "bf_to_xss"
                ],
                "xss_to_bf": metrics[
                    "xss_to_bf"
                ],
                "mutual_confusion": metrics[
                    "mutual_confusion"
                ],
                "guardrail_ok": metrics[
                    "guardrail_ok"
                ],
            }

            grid_results.append(
                row
            )

            # -------------------------------------------------------------
            # Selection rule:
            #
            # 1. Minimize BF/XSS mutual confusion.
            # 2. If tied, maximize Macro-F1.
            # -------------------------------------------------------------

            key = (
                metrics["mutual_confusion"],
                -metrics["macro_f1"],
            )

            if (
                best is None
                or key < best["key"]
            ):
                best = {
                    "key": key,
                    "row": row,
                    "metrics": metrics,
                    "pred": final_pred.copy(),
                    "multiplier": hard_multiplier,
                    "tau": float(tau),
                    "specialist": calibrated_specialist,
                }

    # -------------------------------------------------------------------------
    # GRID RESULTS
    # -------------------------------------------------------------------------

    grid_df = pd.DataFrame(
        grid_results
    ).sort_values(
        ["mutual_confusion", "macro_f1"],
        ascending=[
            True,
            False,
        ],
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FULL STRATEGY 4 GRID"
    )

    print(
        "=" * 80
    )

    print(
        grid_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # SELECTED CONFIGURATION
    # -------------------------------------------------------------------------

    best_metrics = best[
        "metrics"
    ]

    best_pred = best[
        "pred"
    ]

    best_multiplier = best[
        "multiplier"
    ]

    best_tau = best[
        "tau"
    ]

    best_specialist = best[
        "specialist"
    ]

    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTED STRATEGY 4 CONFIGURATION"
    )

    print(
        "=" * 80
    )

    print(
        f"  Hard weight multiplier: "
        f"{best_multiplier}"
    )

    print(
        f"  Tau: "
        f"{best_tau}"
    )

    print(
        "\n  Macro F1:"
    )

    print(
        f"    {baseline_metrics['macro_f1'] * 100:.4f}%"
        f" -> "
        f"{best_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        "\n  Accuracy:"
    )

    print(
        f"    {baseline_metrics['accuracy'] * 100:.4f}%"
        f" -> "
        f"{best_metrics['accuracy'] * 100:.4f}%"
    )

    print(
        "\n  BF F1:"
    )

    print(
        f"    {baseline_metrics['bf_f1']:.4f}"
        f" -> "
        f"{best_metrics['bf_f1']:.4f}"
    )

    print(
        "\n  XSS F1:"
    )

    print(
        f"    {baseline_metrics['xss_f1']:.4f}"
        f" -> "
        f"{best_metrics['xss_f1']:.4f}"
    )

    print(
        "\n  BF -> XSS:"
    )

    print(
        f"    {baseline_metrics['bf_to_xss']}"
        f" -> "
        f"{best_metrics['bf_to_xss']}"
    )

    print(
        "\n  XSS -> BF:"
    )

    print(
        f"    {baseline_metrics['xss_to_bf']}"
        f" -> "
        f"{best_metrics['xss_to_bf']}"
    )

    print(
        "\n  Mutual confusion:"
    )

    print(
        f"    {baseline_metrics['mutual_confusion']}"
        f" -> "
        f"{best_metrics['mutual_confusion']}"
    )

    print(
        "\n  Guardrail:"
    )

    print(
        f"    {best_metrics['guardrail_ok']}"
    )

    # -------------------------------------------------------------------------
    # DECISION
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STRATEGY 4 DECISION"
    )

    print(
        "=" * 80
    )

    if (
        best_metrics["mutual_confusion"]
        >= baseline_metrics["mutual_confusion"]
    ):

        print(
            "\nStrategy 4 DID NOT beat the true baseline "
            "on BF/XSS mutual confusion."
        )

        print(
            "Hard-example weighting therefore does not provide "
            "evidence of improvement under the predefined selection rule."
        )

    elif best_multiplier == 1.0:

        print(
            "\nStrategy 4 DID NOT benefit from additional "
            "hard-example weighting."
        )

        print(
            "The selected multiplier reverted to 1.0, meaning the "
            "hard-example weighting itself provided no benefit."
        )

    else:

        print(
            "\nStrategy 4 IMPROVED the targeted metric "
            "relative to the true baseline."
        )

        print(
            "The selected configuration used an actual hard-example "
            "weight multiplier greater than 1.0."
        )

    # -------------------------------------------------------------------------
    # PER-CLASS COMPARISON
    # -------------------------------------------------------------------------

    all_classes = sorted(
        set(
            baseline_metrics[
                "per_class"
            ]
        )
        |
        set(
            best_metrics[
                "per_class"
            ]
        )
    )

    comparison_rows = []

    for cls in all_classes:

        baseline_class = (
            baseline_metrics[
                "per_class"
            ].get(
                cls,
                {},
            )
        )

        final_class = (
            best_metrics[
                "per_class"
            ].get(
                cls,
                {},
            )
        )

        baseline_f1 = (
            baseline_class.get(
                "f1"
            )
        )

        final_f1 = (
            final_class.get(
                "f1"
            )
        )

        delta = None

        if (
            baseline_f1 is not None
            and final_f1 is not None
        ):
            delta = (
                final_f1 -
                baseline_f1
            )

        comparison_rows.append(
            {
                "class": cls,
                "baseline_f1": baseline_f1,
                "final_f1": final_f1,
                "delta_f1": delta,
                "support": baseline_class.get(
                    "support"
                ),
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "PER-CLASS F1: BASELINE VS STRATEGY 4"
    )

    print(
        "=" * 80
    )

    print(
        comparison_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # SAVE ARTIFACTS
    # -------------------------------------------------------------------------

    print(
        "\nSaving artifacts..."
    )

    MODEL_OUTPUT_DIR = (
        PROJECT_ROOT
        / "models"
        / "historical"
        / "strategy4_hardweighted_specialist_cascade"
    )
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        primary,
        MODEL_OUTPUT_DIR / "primary_model.joblib",
    )

    joblib.dump(
        best_specialist,
        MODEL_OUTPUT_DIR / "specialist_calibrated_hardweighted.joblib",
    )

    grid_df.to_csv(
        OUTPUT_DIR /
        "grid_search_results.csv",
        index=False,
    )

    comparison_df.to_csv(
        OUTPUT_DIR /
        "per_class_comparison.csv",
        index=False,
    )

    summary = {
        "experiment": (
            "hgb_bfxss_hardweighted_specialist_cascade"
        ),
        "baseline": {
            key: value
            for key, value
            in baseline_metrics.items()
            if key != "per_class"
        },
        "selected_hard_weight_multiplier": float(
            best_multiplier
        ),
        "selected_tau": float(
            best_tau
        ),
        "final": {
            key: value
            for key, value
            in best_metrics.items()
            if key != "per_class"
        },
        "n_train_rows": int(
            len(train_df)
        ),
        "n_validation_rows": int(
            len(val_df)
        ),
        "n_features": int(
            len(feature_cols)
        ),
        "n_bfxss_train_rows": int(
            len(y_bx_bin)
        ),
        "n_hard_rows_flagged": int(
            hard_mask.sum()
        ),
        "hard_percentage": float(
            hard_mask.mean()
        ),
        "hard_margin_threshold": float(
            HARD_MARGIN_THRESHOLD
        ),
        "n_oof_folds": int(
            N_OOF_FOLDS
        ),
        "test_loaded": False,
        "guardrail_ok": bool(
            best_metrics["guardrail_ok"]
        ),
    }

    with open(
        OUTPUT_DIR /
        "training_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            default=float,
        )

    # -------------------------------------------------------------------------
    # CONFUSION MATRIX
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        y_val,
        best_pred,
        labels=all_classes,
    )

    pd.DataFrame(
        cm,
        index=all_classes,
        columns=all_classes,
    ).to_csv(
        OUTPUT_DIR /
        "confusion_matrix_final.csv"
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "DONE"
    )

    print(
        "=" * 80
    )

    print(
        f"\nArtifacts written to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nTest set loaded: FALSE"
    )


if __name__ == "__main__":
    main()
