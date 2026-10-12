import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
    / "train.parquet"
)

VAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessed"
    / "validation.parquet"
)

# IMPORTANT:
# Strategy 7 has completely separate output directories.
RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "strategy7_oof_smote_tomek"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "historical"
    / "strategy7_oof_smote_tomek"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


LABEL_COL = "Label"

BF_CLASS = "Web Attack - Brute Force"
XSS_CLASS = "Web Attack - XSS"

RANDOM_STATE = 42
N_OOF_FOLDS = 5

EXPECTED_N_FEATURES = 68


# =============================================================================
# SMOTE-TOMEK CONFIGURATION
# =============================================================================

# Original BF/XSS training distribution:
#
# BF  = 1055
# XSS = 456
#
# We synthesize XSS up to the BF count before Tomek cleaning.
#
# Tomek cleaning can subsequently remove samples from either class.
SMOTE_TARGET_XSS = 1055

SMOTE_K_NEIGHBORS = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_verify(
    path: Path,
    expected_n_features: int,
    label_col: str,
):
    print(f"Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_parquet(path)

    if label_col not in df.columns:
        raise ValueError(
            f"{path}: label column '{label_col}' was not found."
        )

    numeric_feature_cols = [
        c
        for c in df.columns
        if c != label_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    non_numeric_cols = [
        c
        for c in df.columns
        if c != label_col
        and not pd.api.types.is_numeric_dtype(df[c])
    ]

    print(f"  Shape: {df.shape}")

    if non_numeric_cols:
        print(
            f"  Ignoring non-numeric metadata columns: "
            f"{non_numeric_cols}"
        )

    print(
        f"  Numeric model features: "
        f"{len(numeric_feature_cols)}"
    )

    if len(numeric_feature_cols) != expected_n_features:
        raise AssertionError(
            f"{path}: expected "
            f"{expected_n_features} numeric features, "
            f"found {len(numeric_feature_cols)}"
        )

    if df[label_col].isna().any():
        raise ValueError(
            f"{path}: Label contains NaN values."
        )

    return df, numeric_feature_cols


# =============================================================================
# SCHEMA VERIFICATION
# =============================================================================

def verify_matching_schema(
    train_df,
    val_df,
    feature_cols,
    label_col,
):
    train_feature_cols = [
        c
        for c in train_df.columns
        if c != label_col
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    val_feature_cols = [
        c
        for c in val_df.columns
        if c != label_col
        and pd.api.types.is_numeric_dtype(val_df[c])
    ]

    if train_feature_cols != feature_cols:
        raise AssertionError(
            "Training numeric feature-column order "
            "does not match feature_cols."
        )

    if val_feature_cols != feature_cols:
        raise AssertionError(
            "Validation numeric feature-column order "
            "does not match training feature-column order."
        )

    if train_feature_cols != val_feature_cols:
        raise AssertionError(
            "Training and validation numeric feature-column "
            "orders differ."
        )

    if train_df[label_col].isna().any():
        raise ValueError(
            "Training labels contain NaN values."
        )

    if val_df[label_col].isna().any():
        raise ValueError(
            "Validation labels contain NaN values."
        )

    print("Schema verification: PASSED")


# =============================================================================
# METRICS
# =============================================================================

def evaluate(
    y_true,
    y_pred,
    primary_pred,
    all_classes,
):
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_classes,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=all_classes,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=all_classes,
            zero_division=0,
        )
    )

    per_class = {}

    for label, p, r, f, s in zip(
        all_classes,
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
            (y_true == BF_CLASS)
            & (y_pred == XSS_CLASS)
        )
    )

    xss_to_bf = int(
        np.sum(
            (y_true == XSS_CLASS)
            & (y_pred == BF_CLASS)
        )
    )

    mutual_confusion = (
        bf_to_xss + xss_to_bf
    )

    eligible_mask = np.isin(
        primary_pred,
        [BF_CLASS, XSS_CLASS],
    )

    changed_mask = (
        y_pred != primary_pred
    )

    changed_outside_eligible = bool(
        np.any(
            changed_mask
            & ~eligible_mask
        )
    )

    guardrail_ok = (
        not changed_outside_eligible
    )

    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "bf_f1": float(
            per_class[BF_CLASS]["f1"]
        ),
        "xss_f1": float(
            per_class[XSS_CLASS]["f1"]
        ),
        "bf_to_xss": bf_to_xss,
        "xss_to_bf": xss_to_bf,
        "mutual_confusion": mutual_confusion,
        "per_class": per_class,
        "changed_predictions": int(
            np.sum(changed_mask)
        ),
        "changed_outside_eligible": (
            changed_outside_eligible
        ),
        "guardrail_ok": guardrail_ok,
    }


# =============================================================================
# OOF PROBABILITIES
# =============================================================================

def generate_oof_probabilities(
    X_train,
    y_train,
    n_folds,
    seed,
):
    """
    Generate leak-free out-of-fold probabilities.

    Each training row receives probabilities from a model that did not
    train on that row.
    """

    class_order = np.sort(
        np.unique(y_train)
    )

    n_classes = len(
        class_order
    )

    oof_proba = np.zeros(
        (
            len(y_train),
            n_classes,
        ),
        dtype=np.float64,
    )

    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    print()
    print("=" * 80)
    print("GENERATING LEAK-FREE OOF PROBABILITIES")
    print("=" * 80)

    for fold_idx, (
        fit_idx,
        hold_idx,
    ) in enumerate(
        skf.split(
            X_train,
            y_train,
        ),
        start=1,
    ):
        print()
        print(
            f"OOF fold {fold_idx}/{n_folds}"
        )

        fold_model = (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=seed + fold_idx,
            )
        )

        fold_model.fit(
            X_train.iloc[fit_idx],
            y_train[fit_idx],
        )

        proba = (
            fold_model.predict_proba(
                X_train.iloc[hold_idx]
            )
        )

        fold_classes = list(
            fold_model.classes_
        )

        col_map = [
            fold_classes.index(c)
            for c in class_order
        ]

        oof_proba[hold_idx] = (
            proba[:, col_map]
        )

        print(
            f"  Fit rows: "
            f"{len(fit_idx):,}"
        )

        print(
            f"  OOF rows: "
            f"{len(hold_idx):,}"
        )

    if not np.isfinite(
        oof_proba
    ).all():
        raise ValueError(
            "OOF probabilities contain NaN "
            "or infinite values."
        )

    print()
    print("OOF generation complete.")

    return (
        oof_proba,
        class_order,
    )


# =============================================================================
# PRIMARY PRODUCTION MODEL
# =============================================================================

def train_primary(
    X_train,
    y_train,
    seed,
):
    print()
    print("=" * 80)
    print("TRAINING PRODUCTION PRIMARY HGB")
    print("=" * 80)

    model = (
        HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=seed,
        )
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =============================================================================
# BUILD BF/XSS META DATA
# =============================================================================

def build_meta_training_data(
    X_train,
    y_train,
    oof_proba,
):
    """
    Keep only true BF/XSS training rows.

    Meta features:
        68 raw numeric features
        +
        15 OOF class probabilities
        =
        83 meta features
    """

    bx_mask = np.isin(
        y_train,
        [BF_CLASS, XSS_CLASS],
    )

    X_bx_raw = (
        X_train.loc[bx_mask]
        .reset_index(drop=True)
    )

    oof_bx = (
        oof_proba[bx_mask]
    )

    y_bx = (
        y_train[bx_mask]
    )

    y_bx_bin = (
        y_bx == XSS_CLASS
    ).astype(np.int8)

    print()
    print("=" * 80)
    print("BF/XSS META TRAINING DATA")
    print("=" * 80)

    print(
        f"BF/XSS rows: "
        f"{len(X_bx_raw):,}"
    )

    print(
        f"BF rows: "
        f"{np.sum(y_bx == BF_CLASS):,}"
    )

    print(
        f"XSS rows: "
        f"{np.sum(y_bx == XSS_CLASS):,}"
    )

    print(
        f"Raw numeric features: "
        f"{X_bx_raw.shape[1]}"
    )

    print(
        f"OOF probability features: "
        f"{oof_bx.shape[1]}"
    )

    print(
        f"Total meta features: "
        f"{X_bx_raw.shape[1] + oof_bx.shape[1]}"
    )

    return (
        X_bx_raw,
        oof_bx,
        y_bx_bin,
    )


# =============================================================================
# SMOTE-TOMEK META TRAINING DATA
# =============================================================================

def apply_smote_tomek_to_meta_data(
    X_bx_raw,
    oof_bx,
    y_bx_bin,
    seed,
):
    """
    Apply SMOTE-Tomek ONLY to BF/XSS meta-training data.

    The validation set is never passed through SMOTE-Tomek.

    Raw features are standardized before the distance-based
    SMOTE operation. OOF probabilities are already on a bounded
    probability scale and are appended afterward.
    """

    print()
    print("=" * 80)
    print("APPLYING SMOTE-TOMEK TO BF/XSS META DATA")
    print("=" * 80)

    print(
        "Validation data: NOT passed through SMOTE-Tomek"
    )

    print(
        "Test data: NOT loaded"
    )

    scaler = StandardScaler()

    X_raw_scaled = (
        scaler.fit_transform(
            X_bx_raw
        )
    )

    X_meta_original = np.hstack(
        [
            X_raw_scaled,
            oof_bx,
        ]
    )

    if not np.isfinite(
        X_meta_original
    ).all():
        raise ValueError(
            "Meta-training features contain "
            "NaN or infinite values before SMOTE-Tomek."
        )

    print(
        f"Original meta shape: "
        f"{X_meta_original.shape}"
    )

    print(
        f"Original BF: "
        f"{np.sum(y_bx_bin == 0):,}"
    )

    print(
        f"Original XSS: "
        f"{np.sum(y_bx_bin == 1):,}"
    )

    smote = SMOTE(
        sampling_strategy={
            1: SMOTE_TARGET_XSS
        },
        random_state=seed,
        k_neighbors=SMOTE_K_NEIGHBORS,
    )

    smote_tomek = SMOTETomek(
        smote=smote,
        random_state=seed,
    )

    X_resampled, y_resampled = (
        smote_tomek.fit_resample(
            X_meta_original,
            y_bx_bin,
        )
    )

    if not np.isfinite(
        X_resampled
    ).all():
        raise ValueError(
            "SMOTE-Tomek produced NaN "
            "or infinite values."
        )

    print()
    print(
        f"After SMOTE-Tomek: "
        f"{X_resampled.shape}"
    )

    print(
        f"Resampled BF: "
        f"{np.sum(y_resampled == 0):,}"
    )

    print(
        f"Resampled XSS: "
        f"{np.sum(y_resampled == 1):,}"
    )

    print(
        f"Net row change: "
        f"{len(X_resampled) - len(X_meta_original):+,}"
    )

    return (
        X_resampled,
        y_resampled,
        scaler,
    )


# =============================================================================
# TRAIN SMOTE-TOMEK META-LEARNER
# =============================================================================

def train_meta_model(
    X_resampled,
    y_resampled,
    seed,
):
    """
    Train the binary logistic regression meta-learner
    using the SMOTE-Tomek-resampled meta dataset.
    """

    print()
    print("=" * 80)
    print("TRAINING SMOTE-TOMEK BF/XSS META-LEARNER")
    print("=" * 80)

    meta_model = (
        LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    )

    meta_model.fit(
        X_resampled,
        y_resampled,
    )

    print(
        f"Training shape: "
        f"{X_resampled.shape}"
    )

    print(
        "Meta-learner training complete."
    )

    return meta_model


# =============================================================================
# APPLY META OVERRIDE
# =============================================================================

def apply_meta_override(
    primary_pred,
    X_val,
    primary_val_proba,
    scaler,
    meta_model,
):
    """
    Only modify rows where the primary model predicts BF or XSS.

    All other primary predictions remain untouched.
    """

    eligible_mask = np.isin(
        primary_pred,
        [BF_CLASS, XSS_CLASS],
    )

    final_pred = (
        primary_pred.copy()
    )

    n_eligible = int(
        eligible_mask.sum()
    )

    print()
    print("=" * 80)
    print("APPLYING BF/XSS META OVERRIDE")
    print("=" * 80)

    print(
        f"Eligible validation rows: "
        f"{n_eligible:,}"
    )

    if n_eligible == 0:
        return (
            final_pred,
            eligible_mask,
        )

    X_elig_raw = (
        X_val.loc[
            eligible_mask
        ]
    )

    X_elig_scaled = (
        scaler.transform(
            X_elig_raw
        )
    )

    primary_proba_eligible = (
        primary_val_proba[
            eligible_mask
        ]
    )

    X_meta_eligible = np.hstack(
        [
            X_elig_scaled,
            primary_proba_eligible,
        ]
    )

    if not np.isfinite(
        X_meta_eligible
    ).all():
        raise ValueError(
            "Validation meta features contain "
            "NaN or infinite values."
        )

    meta_pred_bin = (
        meta_model.predict(
            X_meta_eligible
        )
    )

    meta_pred_label = np.where(
        meta_pred_bin == 1,
        XSS_CLASS,
        BF_CLASS,
    )

    eligible_indices = (
        np.flatnonzero(
            eligible_mask
        )
    )

    final_pred[
        eligible_indices
    ] = meta_pred_label

    return (
        final_pred,
        eligible_mask,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STRATEGY 7 -- OOF STACKING + SMOTE-TOMEK")
    print("=" * 80)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        "Test set: NOT LOADED"
    )

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================

    train_df, feature_cols = (
        load_and_verify(
            TRAIN_PATH,
            EXPECTED_N_FEATURES,
            LABEL_COL,
        )
    )

    val_df, val_feature_cols = (
        load_and_verify(
            VAL_PATH,
            EXPECTED_N_FEATURES,
            LABEL_COL,
        )
    )

    verify_matching_schema(
        train_df,
        val_df,
        feature_cols,
        LABEL_COL,
    )

    # =========================================================================
    # 2. BUILD MATRICES
    # =========================================================================

    X_train = train_df[
        feature_cols
    ]

    y_train = (
        train_df[
            LABEL_COL
        ].to_numpy()
    )

    X_val = val_df[
        feature_cols
    ]

    y_val = (
        val_df[
            LABEL_COL
        ].to_numpy()
    )

    print()
    print(
        f"Train shape: {train_df.shape}"
    )

    print(
        f"Validation shape: {val_df.shape}"
    )

    print(
        f"Model features: {len(feature_cols)}"
    )

    # =========================================================================
    # 3. CLASS SET
    # =========================================================================

    all_classes = np.sort(
        np.unique(y_train)
    )

    if len(all_classes) != 15:
        raise AssertionError(
            f"Expected 15 classes, "
            f"found {len(all_classes)}"
        )

    val_classes = np.sort(
        np.unique(y_val)
    )

    if not np.array_equal(
        all_classes,
        val_classes,
    ):
        raise AssertionError(
            "Training and validation class sets differ."
        )

    print(
        f"Classes: {len(all_classes)}"
    )

    # =========================================================================
    # 4. GENERATE OOF PROBABILITIES
    # =========================================================================

    (
        oof_proba,
        class_order,
    ) = generate_oof_probabilities(
        X_train,
        y_train,
        N_OOF_FOLDS,
        RANDOM_STATE,
    )

    if not np.array_equal(
        class_order,
        all_classes,
    ):
        raise AssertionError(
            "OOF class order does not match "
            "global class order."
        )

    # =========================================================================
    # 5. TRAIN PRODUCTION PRIMARY MODEL
    # =========================================================================

    primary = train_primary(
        X_train,
        y_train,
        RANDOM_STATE,
    )

    if not np.array_equal(
        primary.classes_,
        class_order,
    ):
        raise AssertionError(
            "Production primary model class order "
            "does not match OOF class order."
        )

    # =========================================================================
    # 6. PRIMARY VALIDATION PREDICTIONS
    # =========================================================================

    primary_val_pred = (
        primary.predict(
            X_val
        )
    )

    primary_val_proba = (
        primary.predict_proba(
            X_val
        )
    )

    if not np.isfinite(
        primary_val_proba
    ).all():
        raise ValueError(
            "Primary validation probabilities contain "
            "NaN or infinite values."
        )

    baseline_metrics = evaluate(
        y_val,
        primary_val_pred,
        primary_val_pred,
        all_classes,
    )

    print()
    print("=" * 80)
    print("PRIMARY BASELINE")
    print("=" * 80)

    print(
        f"Macro-F1: "
        f"{baseline_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Accuracy: "
        f"{baseline_metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"BF F1: "
        f"{baseline_metrics['bf_f1']:.4f}"
    )

    print(
        f"XSS F1: "
        f"{baseline_metrics['xss_f1']:.4f}"
    )

    print(
        f"BF -> XSS: "
        f"{baseline_metrics['bf_to_xss']}"
    )

    print(
        f"XSS -> BF: "
        f"{baseline_metrics['xss_to_bf']}"
    )

    print(
        f"Mutual confusion: "
        f"{baseline_metrics['mutual_confusion']}"
    )

    # =========================================================================
    # 7. BUILD BF/XSS META DATA
    # =========================================================================

    (
        X_bx_raw,
        oof_bx,
        y_bx_bin,
    ) = build_meta_training_data(
        X_train,
        y_train,
        oof_proba,
    )

    # =========================================================================
    # 8. SMOTE-TOMEK
    # =========================================================================

    (
        X_meta_resampled,
        y_meta_resampled,
        meta_scaler,
    ) = apply_smote_tomek_to_meta_data(
        X_bx_raw,
        oof_bx,
        y_bx_bin,
        RANDOM_STATE,
    )

    # =========================================================================
    # 9. TRAIN META-LEARNER
    # =========================================================================

    meta_model = train_meta_model(
        X_meta_resampled,
        y_meta_resampled,
        RANDOM_STATE,
    )

    # =========================================================================
    # 10. APPLY META OVERRIDE
    # =========================================================================

    (
        final_pred,
        eligible_mask,
    ) = apply_meta_override(
        primary_val_pred,
        X_val,
        primary_val_proba,
        meta_scaler,
        meta_model,
    )

    # =========================================================================
    # 11. GUARDRAIL
    # =========================================================================

    changed_mask = (
        final_pred
        != primary_val_pred
    )

    changed_outside_eligible = (
        changed_mask
        & ~eligible_mask
    )

    if np.any(
        changed_outside_eligible
    ):
        raise AssertionError(
            "GUARDRAIL FAILED: predictions outside "
            "primary BF/XSS predictions were changed."
        )

    guardrail_ok = True

    print()
    print(
        "Guardrail: PASSED"
    )

    # =========================================================================
    # 12. FINAL EVALUATION
    # =========================================================================

    final_metrics = evaluate(
        y_val,
        final_pred,
        primary_val_pred,
        all_classes,
    )

    if not final_metrics[
        "guardrail_ok"
    ]:
        raise AssertionError(
            "Evaluation guardrail failed."
        )

    # =========================================================================
    # 13. RESULTS
    # =========================================================================

    print()
    print("=" * 80)
    print("STRATEGY 7 RESULTS")
    print("=" * 80)

    print()
    print(
        f"Macro-F1:"
        f" {baseline_metrics['macro_f1'] * 100:.4f}%"
        f" -> "
        f"{final_metrics['macro_f1'] * 100:.4f}%"
        f"  "
        f"({(
            final_metrics['macro_f1']
            - baseline_metrics['macro_f1']
        ) * 100:+.4f} pp)"
    )

    print(
        f"Accuracy:"
        f" {baseline_metrics['accuracy'] * 100:.4f}%"
        f" -> "
        f"{final_metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"Weighted-F1:"
        f" {baseline_metrics['weighted_f1'] * 100:.4f}%"
        f" -> "
        f"{final_metrics['weighted_f1'] * 100:.4f}%"
    )

    print()
    print(
        f"BF F1:"
        f" {baseline_metrics['bf_f1']:.4f}"
        f" -> "
        f"{final_metrics['bf_f1']:.4f}"
    )

    print(
        f"XSS F1:"
        f" {baseline_metrics['xss_f1']:.4f}"
        f" -> "
        f"{final_metrics['xss_f1']:.4f}"
    )

    print()
    print(
        f"BF -> XSS:"
        f" {baseline_metrics['bf_to_xss']}"
        f" -> "
        f"{final_metrics['bf_to_xss']}"
    )

    print(
        f"XSS -> BF:"
        f" {baseline_metrics['xss_to_bf']}"
        f" -> "
        f"{final_metrics['xss_to_bf']}"
    )

    print(
        f"Mutual confusion:"
        f" {baseline_metrics['mutual_confusion']}"
        f" -> "
        f"{final_metrics['mutual_confusion']}"
    )

    print()
    print(
        f"Changed predictions: "
        f"{final_metrics['changed_predictions']}"
    )

    # =========================================================================
    # 14. PER-CLASS COMPARISON
    # =========================================================================

    per_class_rows = []

    for label in all_classes:

        baseline_f1 = (
            baseline_metrics[
                "per_class"
            ][label]["f1"]
        )

        final_f1 = (
            final_metrics[
                "per_class"
            ][label]["f1"]
        )

        support = (
            final_metrics[
                "per_class"
            ][label]["support"]
        )

        per_class_rows.append(
            {
                "class": label,
                "f1_before": baseline_f1,
                "f1_after": final_f1,
                "delta": (
                    final_f1
                    - baseline_f1
                ),
                "support": support,
            }
        )

    per_class_df = (
        pd.DataFrame(
            per_class_rows
        )
    )

    per_class_df.to_csv(
        RESULTS_DIR
        / "per_class_comparison.csv",
        index=False,
    )

    # =========================================================================
    # 15. CONFUSION MATRICES
    # =========================================================================

    cm_before = confusion_matrix(
        y_val,
        primary_val_pred,
        labels=all_classes,
    )

    cm_after = confusion_matrix(
        y_val,
        final_pred,
        labels=all_classes,
    )

    cm_before_df = pd.DataFrame(
        cm_before,
        index=all_classes,
        columns=all_classes,
    )

    cm_after_df = pd.DataFrame(
        cm_after,
        index=all_classes,
        columns=all_classes,
    )

    cm_before_df.to_csv(
        RESULTS_DIR
        / "confusion_matrix_before.csv"
    )

    cm_after_df.to_csv(
        RESULTS_DIR
        / "confusion_matrix_after.csv"
    )

    # =========================================================================
    # 16. BEFORE/AFTER SUMMARY
    # =========================================================================

    comparison_df = pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "before": baseline_metrics["macro_f1"],
                "after": final_metrics["macro_f1"],
                "delta": (
                    final_metrics["macro_f1"]
                    - baseline_metrics["macro_f1"]
                ),
            },
            {
                "metric": "weighted_f1",
                "before": baseline_metrics["weighted_f1"],
                "after": final_metrics["weighted_f1"],
                "delta": (
                    final_metrics["weighted_f1"]
                    - baseline_metrics["weighted_f1"]
                ),
            },
            {
                "metric": "accuracy",
                "before": baseline_metrics["accuracy"],
                "after": final_metrics["accuracy"],
                "delta": (
                    final_metrics["accuracy"]
                    - baseline_metrics["accuracy"]
                ),
            },
            {
                "metric": "bf_f1",
                "before": baseline_metrics["bf_f1"],
                "after": final_metrics["bf_f1"],
                "delta": (
                    final_metrics["bf_f1"]
                    - baseline_metrics["bf_f1"]
                ),
            },
            {
                "metric": "xss_f1",
                "before": baseline_metrics["xss_f1"],
                "after": final_metrics["xss_f1"],
                "delta": (
                    final_metrics["xss_f1"]
                    - baseline_metrics["xss_f1"]
                ),
            },
            {
                "metric": "bf_to_xss",
                "before": baseline_metrics["bf_to_xss"],
                "after": final_metrics["bf_to_xss"],
                "delta": (
                    final_metrics["bf_to_xss"]
                    - baseline_metrics["bf_to_xss"]
                ),
            },
            {
                "metric": "xss_to_bf",
                "before": baseline_metrics["xss_to_bf"],
                "after": final_metrics["xss_to_bf"],
                "delta": (
                    final_metrics["xss_to_bf"]
                    - baseline_metrics["xss_to_bf"]
                ),
            },
            {
                "metric": "mutual_confusion",
                "before": baseline_metrics["mutual_confusion"],
                "after": final_metrics["mutual_confusion"],
                "delta": (
                    final_metrics["mutual_confusion"]
                    - baseline_metrics["mutual_confusion"]
                ),
            },
            {
                "metric": "changed_predictions",
                "before": 0,
                "after": final_metrics["changed_predictions"],
                "delta": final_metrics["changed_predictions"],
            },
        ]
    )

    comparison_df.to_csv(
        RESULTS_DIR
        / "before_after_comparison.csv",
        index=False,
    )

    # =========================================================================
    # 17. SAVE MODELS
    # =========================================================================

    joblib.dump(
        primary,
        MODELS_DIR
        / "primary_model.joblib",
    )

    joblib.dump(
        meta_model,
        MODELS_DIR
        / "meta_model_logreg_smote_tomek.joblib",
    )

    joblib.dump(
        meta_scaler,
        MODELS_DIR
        / "meta_feature_scaler.joblib",
    )

    joblib.dump(
        class_order,
        MODELS_DIR
        / "class_order.joblib",
    )

    # =========================================================================
    # 18. SAVE SUMMARY
    # =========================================================================

    summary = {
        "experiment": (
            "hgb_bfxss_oof_smote_tomek"
        ),

        "configuration": {
            "random_state": RANDOM_STATE,
            "n_oof_folds": N_OOF_FOLDS,
            "expected_numeric_features": (
                EXPECTED_N_FEATURES
            ),
            "actual_numeric_features": (
                len(feature_cols)
            ),
            "oof_probability_feature_count": (
                len(class_order)
            ),
            "total_meta_features": (
                len(feature_cols)
                + len(class_order)
            ),
            "smote_target_xss": SMOTE_TARGET_XSS,
            "smote_k_neighbors": SMOTE_K_NEIGHBORS,
            "test_loaded": False,
        },

        "meta_training_before_resampling": {
            "rows": int(len(X_bx_raw)),
            "bf_rows": int(
                np.sum(y_bx_bin == 0)
            ),
            "xss_rows": int(
                np.sum(y_bx_bin == 1)
            ),
        },

        "meta_training_after_smote_tomek": {
            "rows": int(
                len(X_meta_resampled)
            ),
            "bf_rows": int(
                np.sum(y_meta_resampled == 0)
            ),
            "xss_rows": int(
                np.sum(y_meta_resampled == 1)
            ),
        },

        "meta_model": {
            "model": "LogisticRegression",
            "type": "binary",
            "target_classes": [
                BF_CLASS,
                XSS_CLASS,
            ],
            "penalty": "l2",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },

        "baseline": {
            "macro_f1": baseline_metrics[
                "macro_f1"
            ],
            "weighted_f1": baseline_metrics[
                "weighted_f1"
            ],
            "accuracy": baseline_metrics[
                "accuracy"
            ],
            "bf_f1": baseline_metrics[
                "bf_f1"
            ],
            "xss_f1": baseline_metrics[
                "xss_f1"
            ],
            "bf_to_xss": baseline_metrics[
                "bf_to_xss"
            ],
            "xss_to_bf": baseline_metrics[
                "xss_to_bf"
            ],
            "mutual_confusion": baseline_metrics[
                "mutual_confusion"
            ],
        },

        "final": {
            "macro_f1": final_metrics[
                "macro_f1"
            ],
            "weighted_f1": final_metrics[
                "weighted_f1"
            ],
            "accuracy": final_metrics[
                "accuracy"
            ],
            "bf_f1": final_metrics[
                "bf_f1"
            ],
            "xss_f1": final_metrics[
                "xss_f1"
            ],
            "bf_to_xss": final_metrics[
                "bf_to_xss"
            ],
            "xss_to_bf": final_metrics[
                "xss_to_bf"
            ],
            "mutual_confusion": final_metrics[
                "mutual_confusion"
            ],
            "changed_predictions": final_metrics[
                "changed_predictions"
            ],
        },

        "deltas": {
            "macro_f1": (
                final_metrics["macro_f1"]
                - baseline_metrics["macro_f1"]
            ),
            "weighted_f1": (
                final_metrics["weighted_f1"]
                - baseline_metrics["weighted_f1"]
            ),
            "accuracy": (
                final_metrics["accuracy"]
                - baseline_metrics["accuracy"]
            ),
            "bf_f1": (
                final_metrics["bf_f1"]
                - baseline_metrics["bf_f1"]
            ),
            "xss_f1": (
                final_metrics["xss_f1"]
                - baseline_metrics["xss_f1"]
            ),
            "bf_to_xss": (
                final_metrics["bf_to_xss"]
                - baseline_metrics["bf_to_xss"]
            ),
            "xss_to_bf": (
                final_metrics["xss_to_bf"]
                - baseline_metrics["xss_to_bf"]
            ),
            "mutual_confusion": (
                final_metrics["mutual_confusion"]
                - baseline_metrics["mutual_confusion"]
            ),
        },

        "guardrail_ok": guardrail_ok,

        "n_train_rows": int(
            len(X_train)
        ),

        "n_validation_rows": int(
            len(X_val)
        ),

        "n_numeric_features": int(
            len(feature_cols)
        ),

        "n_classes": int(
            len(class_order)
        ),

        "n_bfxss_train_rows": int(
            len(X_bx_raw)
        ),

        "test_loaded": False,
    }

    summary_path = (
        RESULTS_DIR
        / "training_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # =========================================================================
    # 19. FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("STRATEGY 7 COMPLETE")
    print("=" * 80)

    print()
    print(
        f"Baseline Macro-F1: "
        f"{baseline_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Final Macro-F1: "
        f"{final_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Macro-F1 delta: "
        f"{(
            final_metrics['macro_f1']
            - baseline_metrics['macro_f1']
        ) * 100:+.4f} percentage points"
    )

    print()
    print(
        f"Baseline mutual confusion: "
        f"{baseline_metrics['mutual_confusion']}"
    )

    print(
        f"Final mutual confusion: "
        f"{final_metrics['mutual_confusion']}"
    )

    print(
        f"Mutual confusion delta: "
        f"{(
            final_metrics['mutual_confusion']
            - baseline_metrics['mutual_confusion']
        ): +d}"
    )

    print()
    print(
        f"BF -> XSS: "
        f"{baseline_metrics['bf_to_xss']}"
        f" -> "
        f"{final_metrics['bf_to_xss']}"
    )

    print(
        f"XSS -> BF: "
        f"{baseline_metrics['xss_to_bf']}"
        f" -> "
        f"{final_metrics['xss_to_bf']}"
    )

    print()
    print(
        f"Guardrail: {guardrail_ok}"
    )

    print()
    print(
        "Test set loaded: FALSE"
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_DIR
    )

    print()
    print(
        "Models saved to:"
    )

    print(
        MODELS_DIR
    )


if __name__ == "__main__":
    main()