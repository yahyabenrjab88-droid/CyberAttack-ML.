import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.stats import binomtest

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

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "strategy5_oof_stacked_metalearner"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
    / "historical"
    / "strategy5_oof_stacked_metalearner"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


LABEL_COL = "Label"

BF_CLASS = "Web Attack - Brute Force"
XSS_CLASS = "Web Attack - XSS"

RANDOM_STATE = 42
N_OOF_FOLDS = 5
N_BOOTSTRAP = 2000

# IMPORTANT:
# The parquet files contain:
#   68 numeric ML features
#   1 non-numeric source/filename column
#   1 Label column
#
# Only the 68 numeric features are used by the ML models.
EXPECTED_N_FEATURES = 68


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

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Only numeric columns are model features.
    #
    # CICIDS2017 parquet contains a source/filename metadata column containing
    # values such as:
    #
    # Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    #
    # That column is NOT used as a model feature.
    # -------------------------------------------------------------------------

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

    print(
        f"  Label column: {label_col}"
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
        and pd.api.types.is_numeric_dtype(
            train_df[c]
        )
    ]

    val_feature_cols = [
        c
        for c in val_df.columns
        if c != label_col
        and pd.api.types.is_numeric_dtype(
            val_df[c]
        )
    ]

    # Exact order matters.
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

    print(
        "Schema verification: PASSED"
    )


# =============================================================================
# METRICS
# =============================================================================

def evaluate(
    y_true,
    y_pred,
    primary_pred,
    all_classes,
):
    """
    Evaluate predictions.

    all_classes is fixed so that Macro-F1 always uses the same
    15-class denominator, including during bootstrap.
    """

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
    print(
        "OOF generation complete."
    )

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
# BINARY LOGISTIC META-LEARNER
# =============================================================================

def train_meta_model(
    X_bx_raw,
    oof_bx,
    y_bx_bin,
    seed,
):
    """
    Train binary L2-regularized logistic regression.

    Target:
        0 = BF
        1 = XSS
    """

    print()
    print("=" * 80)
    print("TRAINING BF/XSS BINARY LOGISTIC META-LEARNER")
    print("=" * 80)

    scaler = StandardScaler()

    X_bx_scaled = (
        scaler.fit_transform(
            X_bx_raw
        )
    )

    X_meta = np.hstack(
        [
            X_bx_scaled,
            oof_bx,
        ]
    )

    if not np.isfinite(
        X_meta
    ).all():
        raise ValueError(
            "Meta-training features contain "
            "NaN or infinite values."
        )

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
        X_meta,
        y_bx_bin,
    )

    print(
        f"Meta-training shape: "
        f"{X_meta.shape}"
    )

    print(
        "Meta-learner training complete."
    )

    return (
        meta_model,
        scaler,
    )


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
# BOOTSTRAP
# =============================================================================

def bootstrap_deltas(
    y_val,
    primary_pred,
    final_pred,
    all_classes,
    n_boot,
    seed,
):
    """
    Bootstrap paired validation predictions.

    Macro-F1 uses the complete fixed class list.
    """

    print()
    print("=" * 80)
    print("BOOTSTRAPPING DELTAS")
    print("=" * 80)

    rng = np.random.default_rng(
        seed
    )

    n = len(y_val)

    macro_f1_deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    mutual_conf_deltas = np.empty(
        n_boot,
        dtype=np.float64,
    )

    for b in range(n_boot):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        y_sample = (
            y_val[idx]
        )

        primary_sample = (
            primary_pred[idx]
        )

        final_sample = (
            final_pred[idx]
        )

        base_metrics = evaluate(
            y_sample,
            primary_sample,
            primary_sample,
            all_classes,
        )

        final_metrics = evaluate(
            y_sample,
            final_sample,
            primary_sample,
            all_classes,
        )

        macro_f1_deltas[b] = (
            final_metrics["macro_f1"]
            - base_metrics["macro_f1"]
        )

        mutual_conf_deltas[b] = (
            final_metrics[
                "mutual_confusion"
            ]
            - base_metrics[
                "mutual_confusion"
            ]
        )

    def percentile_ci(values):
        return (
            float(
                np.percentile(
                    values,
                    2.5,
                )
            ),
            float(
                np.percentile(
                    values,
                    97.5,
                )
            ),
        )

    macro_ci = percentile_ci(
        macro_f1_deltas
    )

    mutual_ci = percentile_ci(
        mutual_conf_deltas
    )

    print(
        f"Macro-F1 delta 95% CI: "
        f"[{macro_ci[0]:.6f}, "
        f"{macro_ci[1]:.6f}]"
    )

    print(
        f"Mutual confusion delta 95% CI: "
        f"[{mutual_ci[0]:.2f}, "
        f"{mutual_ci[1]:.2f}]"
    )

    return {
        "n_bootstrap": int(
            n_boot
        ),
        "macro_f1_delta_ci_95": {
            "lower": macro_ci[0],
            "upper": macro_ci[1],
        },
        "mutual_confusion_delta_ci_95": {
            "lower": mutual_ci[0],
            "upper": mutual_ci[1],
        },
    }


# =============================================================================
# MCNEMAR
# =============================================================================

def mcnemar_on_bfxss_rows(
    y_val,
    primary_pred,
    final_pred,
):
    """
    Exact McNemar test restricted to true BF/XSS validation rows.

    Tests whether the two models have different paired correctness
    outcomes on the targeted BF/XSS population.
    """

    bx_mask = np.isin(
        y_val,
        [BF_CLASS, XSS_CLASS],
    )

    y_bx = (
        y_val[bx_mask]
    )

    primary_bx = (
        primary_pred[bx_mask]
    )

    final_bx = (
        final_pred[bx_mask]
    )

    primary_correct = (
        primary_bx == y_bx
    )

    final_correct = (
        final_bx == y_bx
    )

    damaged = int(
        np.sum(
            primary_correct
            & ~final_correct
        )
    )

    fixed = int(
        np.sum(
            ~primary_correct
            & final_correct
        )
    )

    discordant = (
        fixed + damaged
    )

    if discordant == 0:
        p_value = 1.0

    else:
        test = binomtest(
            fixed,
            discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = float(
            test.pvalue
        )

    print()
    print("=" * 80)
    print("EXACT MCNEMAR TEST -- TRUE BF/XSS ROWS")
    print("=" * 80)

    print(
        f"Target rows: "
        f"{len(y_bx):,}"
    )

    print(
        f"Fixed: {fixed}"
    )

    print(
        f"Damaged: {damaged}"
    )

    print(
        f"Discordant: {discordant}"
    )

    print(
        f"Exact McNemar p-value: "
        f"{p_value:.6f}"
    )

    return {
        "target_rows": int(
            len(y_bx)
        ),
        "fixed": fixed,
        "damaged": damaged,
        "discordant": discordant,
        "p_value": p_value,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("STRATEGY 5 -- OOF STACKED BF/XSS META-LEARNER")
    print("=" * 80)

    print()
    print(
        "Project root:"
    )
    print(
        PROJECT_ROOT
    )

    print()
    print(
        "Test set: NOT LOADED"
    )

    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================

    print()
    print(
        "Loading data..."
    )

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
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Train shape: "
        f"{train_df.shape}"
    )

    print(
        f"Validation shape: "
        f"{val_df.shape}"
    )

    print(
        f"Number of model features: "
        f"{len(feature_cols)}"
    )

    print(
        "Test set NOT loaded."
    )

    # =========================================================================
    # 3. CLASS SET
    # =========================================================================

    all_classes = np.sort(
        np.unique(y_train)
    )

    print(
        f"Number of classes: "
        f"{len(all_classes)}"
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

    # =========================================================================
    # 4. OOF PROBABILITIES
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
    # 5. PRODUCTION PRIMARY MODEL
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

    print()
    print("=" * 80)
    print("PRIMARY VALIDATION PREDICTIONS")
    print("=" * 80)

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
    print(
        f"Baseline Macro-F1: "
        f"{baseline_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"Baseline BF F1: "
        f"{baseline_metrics['bf_f1']:.4f}"
    )

    print(
        f"Baseline XSS F1: "
        f"{baseline_metrics['xss_f1']:.4f}"
    )

    print(
        f"Baseline BF -> XSS: "
        f"{baseline_metrics['bf_to_xss']}"
    )

    print(
        f"Baseline XSS -> BF: "
        f"{baseline_metrics['xss_to_bf']}"
    )

    print(
        f"Baseline mutual confusion: "
        f"{baseline_metrics['mutual_confusion']}"
    )

    # =========================================================================
    # 7. BF/XSS META DATA
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
    # 8. TRAIN META MODEL
    # =========================================================================

    (
        meta_model,
        meta_scaler,
    ) = train_meta_model(
        X_bx_raw,
        oof_bx,
        y_bx_bin,
        RANDOM_STATE,
    )

    # =========================================================================
    # 9. APPLY META OVERRIDE
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
    # 10. GUARDRAIL
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
    # 11. FINAL EVALUATION
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

    print()
    print("=" * 80)
    print("STRATEGY 5 RESULTS")
    print("=" * 80)

    print()
    print(
        f"Macro-F1:"
        f" {baseline_metrics['macro_f1'] * 100:.4f}%"
        f" -> "
        f"{final_metrics['macro_f1'] * 100:.4f}%"
    )

    print(
        f"Accuracy:"
        f" {baseline_metrics['accuracy'] * 100:.4f}%"
        f" -> "
        f"{final_metrics['accuracy'] * 100:.4f}%"
    )

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

    print(
        f"Changed predictions:"
        f" {final_metrics['changed_predictions']}"
    )

    # =========================================================================
    # 12. MCNEMAR
    # =========================================================================

    mcnemar_results = (
        mcnemar_on_bfxss_rows(
            y_val,
            primary_val_pred,
            final_pred,
        )
    )

    # =========================================================================
    # 13. BOOTSTRAP
    # =========================================================================

    bootstrap_results = (
        bootstrap_deltas(
            y_val,
            primary_val_pred,
            final_pred,
            all_classes,
            N_BOOTSTRAP,
            RANDOM_STATE,
        )
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
                "baseline_f1": baseline_f1,
                "final_f1": final_f1,
                "delta_f1": (
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

    # =========================================================================
    # 15. CONFUSION MATRIX
    # =========================================================================

    cm = confusion_matrix(
        y_val,
        final_pred,
        labels=all_classes,
    )

    cm_df = pd.DataFrame(
        cm,
        index=all_classes,
        columns=all_classes,
    )

    cm_df.index.name = ""

    cm_df.to_csv(
        RESULTS_DIR
        / "confusion_matrix_final.csv"
    )

    per_class_df.to_csv(
        RESULTS_DIR
        / "per_class_comparison.csv",
        index=False,
    )

    # =========================================================================
    # 16. SAVE MODELS
    # =========================================================================

    joblib.dump(
        primary,
        MODELS_DIR
        / "primary_model.joblib",
    )

    joblib.dump(
        meta_model,
        MODELS_DIR
        / "meta_model_logreg.joblib",
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
    # 17. SAVE SUMMARY
    # =========================================================================

    summary = {
        "experiment": (
            "hgb_bfxss_oof_stacked_metalearner"
        ),

        "configuration": {
            "random_state": RANDOM_STATE,
            "n_oof_folds": N_OOF_FOLDS,
            "n_bootstrap": N_BOOTSTRAP,
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
            "test_loaded": False,
        },

        "ignored_non_numeric_columns": [
            c
            for c in train_df.columns
            if c != LABEL_COL
            and not pd.api.types.is_numeric_dtype(
                train_df[c]
            )
        ],

        "meta_model": {
            "model": (
                "LogisticRegression"
            ),
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
                final_metrics[
                    "mutual_confusion"
                ]
                - baseline_metrics[
                    "mutual_confusion"
                ]
            ),
        },

        "mcnemar": mcnemar_results,

        "bootstrap": bootstrap_results,

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
    # 18. FINAL SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("STRATEGY 5 COMPLETE")
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
        ):+d}"
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

    print()
    print(
        "Strategy 5 experiment finished."
    )


if __name__ == "__main__":
    main()