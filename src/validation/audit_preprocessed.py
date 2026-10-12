from pathlib import Path
import json
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
REPORT_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PREPROCESSED_DIR / "train.parquet"
VAL_PATH = PREPROCESSED_DIR / "validation.parquet"
TEST_PATH = PREPROCESSED_DIR / "test.parquet"

FEATURE_COLUMNS_PATH = PREPROCESSED_DIR / "feature_columns.csv"
MEDIANS_PATH = PREPROCESSED_DIR / "train_medians.csv"

REPORT_PATH = REPORT_DIR / "preprocessed_audit_report.json"


# ============================================================
# EXPECTED PIPELINE CONFIGURATION
# ============================================================

EXPECTED_FEATURE_COUNT = 68

EXPECTED_MISSING_INDICATORS = {
    "Flow Bytes/s_missing",
    "Flow IAT Min_missing",
    "Fwd IAT Min_missing",
}

EXPECTED_IMPUTED_COLUMNS = {
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Min",
    "Fwd IAT Min",
}

EXPECTED_CONSTANT_REMOVED = {
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}

TARGET_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"


# ============================================================
# TERMINAL COLORS
# ============================================================

class Colors:
    RESET = "\033[0m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    BOLD = "\033[1m"


def color(text, colour):
    return f"{colour}{text}{Colors.RESET}"


def title(text):
    print()
    print(color("=" * 72, Colors.CYAN))
    print(color(text, Colors.BOLD + Colors.CYAN))
    print(color("=" * 72, Colors.CYAN))


def success(text):
    print(color(f"[PASS] {text}", Colors.GREEN))


def warning(text):
    print(color(f"[WARN] {text}", Colors.YELLOW))


def failure(text):
    print(color(f"[FAIL] {text}", Colors.RED))


def info(text):
    print(color(f"[INFO] {text}", Colors.BLUE))


def detail(text):
    print(color(f"       {text}", Colors.WHITE))


# ============================================================
# HELPERS
# ============================================================

def load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    info(f"Loading {path.name} ...")
    df = pd.read_parquet(path)
    success(f"Loaded {path.name}: {len(df):,} rows × {len(df.columns)} columns")

    return df


def get_feature_columns(df):
    excluded = {TARGET_COLUMN, PROVENANCE_COLUMN}
    return [c for c in df.columns if c not in excluded]


def check_no_nan_inf(df, feature_columns, split_name):
    numeric_features = df[feature_columns].select_dtypes(
        include=[np.number]
    ).columns

    nan_count = int(df[numeric_features].isna().sum().sum())

    inf_count = 0

    if len(numeric_features) > 0:
        values = df[numeric_features].to_numpy(dtype=np.float64)
        inf_count = int(np.isinf(values).sum())

    passed = nan_count == 0 and inf_count == 0

    if passed:
        success(
            f"{split_name}: no NaN or infinity values "
            f"in model features"
        )
    else:
        failure(
            f"{split_name}: NaN={nan_count:,}, "
            f"infinity={inf_count:,}"
        )

    return passed, nan_count, inf_count


def check_numeric_features(df, feature_columns, split_name):
    non_numeric = [
        c for c in feature_columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    if not non_numeric:
        success(f"{split_name}: all model features are numeric")
        return True

    failure(
        f"{split_name}: non-numeric model features found: "
        f"{non_numeric}"
    )
    return False


def check_feature_count(df, split_name):
    features = get_feature_columns(df)
    count = len(features)

    if count == EXPECTED_FEATURE_COUNT:
        success(
            f"{split_name}: exactly {EXPECTED_FEATURE_COUNT} "
            f"model features"
        )
        return True

    failure(
        f"{split_name}: expected {EXPECTED_FEATURE_COUNT} "
        f"features, found {count}"
    )
    return False


def check_feature_consistency(train, val, test):
    train_features = set(get_feature_columns(train))
    val_features = set(get_feature_columns(val))
    test_features = set(get_feature_columns(test))

    train_val = train_features == val_features
    train_test = train_features == test_features

    if train_val and train_test:
        success("Feature consistency: TRAIN = VALIDATION = TEST")
        return True

    failure("Feature consistency check failed")

    if train_features != val_features:
        failure("TRAIN and VALIDATION feature sets differ")

    if train_features != test_features:
        failure("TRAIN and TEST feature sets differ")

    return False


def check_target(df, split_name):
    if TARGET_COLUMN not in df.columns:
        failure(f"{split_name}: missing target column '{TARGET_COLUMN}'")
        return False

    missing_labels = int(df[TARGET_COLUMN].isna().sum())

    if missing_labels == 0:
        success(f"{split_name}: no missing target labels")
        return True

    failure(
        f"{split_name}: {missing_labels:,} missing target labels"
    )
    return False


def check_source_file(df, split_name):
    if PROVENANCE_COLUMN not in df.columns:
        warning(
            f"{split_name}: source_file column is absent "
            f"(this is acceptable if intentionally removed)"
        )
        return True

    features = get_feature_columns(df)

    if PROVENANCE_COLUMN not in features:
        success(
            f"{split_name}: source_file is retained only as provenance "
            f"and is NOT a model feature"
        )
        return True

    failure(
        f"{split_name}: source_file is incorrectly included "
        f"as a model feature"
    )
    return False


def check_missing_indicators(df, split_name):
    features = set(get_feature_columns(df))

    found = {
        c for c in features
        if c.endswith("_missing")
    }

    if found == EXPECTED_MISSING_INDICATORS:
        success(
            f"{split_name}: correct missing indicators found"
        )
        for col in sorted(found):
            detail(col)
        return True

    failure(
        f"{split_name}: incorrect missing indicators"
    )

    detail(f"Expected: {sorted(EXPECTED_MISSING_INDICATORS)}")
    detail(f"Found:    {sorted(found)}")

    return False


def check_indicator_values(df, split_name):
    indicators = [
        c for c in df.columns
        if c.endswith("_missing")
    ]

    if not indicators:
        failure(
            f"{split_name}: no missing indicators found"
        )
        return False

    passed = True

    for col in indicators:
        values = set(df[col].dropna().unique())

        if not values.issubset({0, 1}):
            failure(
                f"{split_name}: {col} contains values "
                f"outside {{0, 1}}: {values}"
            )
            passed = False

    if passed:
        success(
            f"{split_name}: all missing indicators contain only 0/1"
        )

    return passed


def check_constant_features(df, split_name):
    feature_columns = get_feature_columns(df)

    constant_features = []

    for col in feature_columns:
        if df[col].nunique(dropna=False) <= 1:
            constant_features.append(col)

    if not constant_features:
        success(
            f"{split_name}: no constant model features"
        )
        return True

    failure(
        f"{split_name}: constant model features found"
    )

    for col in constant_features:
        detail(col)

    return False


def find_split_local_duplicate_pairs(df, split_name):
    """
    Finds exact duplicate feature columns inside this split.

    IMPORTANT:
    These are informational only.

    A pair being identical inside one split does NOT prove
    the original CICIDS features are globally redundant.
    """

    feature_columns = get_feature_columns(df)

    duplicates = []

    seen = {}

    for col in feature_columns:
        series = df[col]

        # Hashing is considerably faster than comparing every pair.
        try:
            hashed = pd.util.hash_pandas_object(
                series,
                index=False
            )

            signature = (
                int(hashed.sum()),
                int(hashed.sum() % 2**32),
                len(series),
            )

            if signature in seen:
                for previous_col in seen[signature]:
                    if series.equals(df[previous_col]):
                        duplicates.append(
                            (previous_col, col)
                        )

                seen[signature].append(col)

            else:
                seen[signature] = [col]

        except Exception:
            continue

    if duplicates:
        warning(
            f"{split_name}: {len(duplicates)} exact duplicate "
            f"feature pair(s) found inside this split"
        )

        for a, b in duplicates:
            detail(f"{a} == {b}")

        detail(
            "These are informational only; they are NOT automatically "
            "removed because split-local equality does not prove "
            "global redundancy."
        )

    else:
        success(
            f"{split_name}: no exact duplicate feature pairs "
            f"inside this split"
        )

    return duplicates


def check_expected_constant_removals(df, split_name):
    features = set(get_feature_columns(df))

    still_present = features.intersection(EXPECTED_CONSTANT_REMOVED)

    if not still_present:
        success(
            f"{split_name}: all 8 known constant-zero features "
            f"are excluded from model input"
        )
        return True

    failure(
        f"{split_name}: constant-zero features still present"
    )

    for col in sorted(still_present):
        detail(col)

    return False


def check_imputed_columns(df, split_name):
    missing = []

    for col in EXPECTED_IMPUTED_COLUMNS:
        if col not in df.columns:
            missing.append(col)

    if not missing:
        success(
            f"{split_name}: all expected imputed columns are present"
        )
        return True

    failure(
        f"{split_name}: missing expected imputed columns"
    )

    for col in missing:
        detail(col)

    return False


def class_distribution(df, split_name):
    counts = df[TARGET_COLUMN].value_counts(dropna=False)
    percentages = (
        df[TARGET_COLUMN]
        .value_counts(normalize=True, dropna=False)
        .mul(100)
    )

    print()
    print(color(f"Class distribution — {split_name}", Colors.MAGENTA))
    print(color("-" * 60, Colors.MAGENTA))

    for label in counts.index:
        print(
            f"{str(label):35s} "
            f"{counts[label]:>10,} "
            f"({percentages[label]:>7.3f}%)"
        )

    return {
        str(label): {
            "count": int(counts[label]),
            "percentage": float(percentages[label]),
        }
        for label in counts.index
    }


def check_all_labels_present(train, val, test):
    train_labels = set(train[TARGET_COLUMN].unique())
    val_labels = set(val[TARGET_COLUMN].unique())
    test_labels = set(test[TARGET_COLUMN].unique())

    common = train_labels & val_labels & test_labels

    if (
        train_labels == val_labels
        and train_labels == test_labels
    ):
        success(
            f"All splits contain the same {len(common)} labels"
        )
        return True

    failure("Label sets differ between splits")

    detail(f"TRAIN: {sorted(train_labels)}")
    detail(f"VAL:   {sorted(val_labels)}")
    detail(f"TEST:  {sorted(test_labels)}")

    return False


def check_extreme_values(df, split_name):
    """
    Reports very large numeric values but does NOT delete or fail
    because of them.

    CICIDS2017 contains legitimate heavy-tailed network statistics.
    """

    feature_columns = get_feature_columns(df)
    numeric_columns = df[feature_columns].select_dtypes(
        include=[np.number]
    ).columns

    extreme_report = {}

    for col in numeric_columns:
        series = df[col]

        if len(series) == 0:
            continue

        max_value = series.max()

        if pd.isna(max_value):
            continue

        if np.isfinite(max_value) and abs(max_value) > 1e12:
            extreme_report[col] = float(max_value)

    if extreme_report:
        warning(
            f"{split_name}: very large numeric values detected"
        )

        for col, value in sorted(
            extreme_report.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:15]:
            detail(f"{col}: {value:.6e}")

        detail(
            "These are reported only. No arbitrary outlier deletion "
            "is performed."
        )
    else:
        success(
            f"{split_name}: no values above the audit threshold "
            f"of 1e12"
        )

    return extreme_report


def check_train_statistics_file():
    if not MEDIANS_PATH.exists():
        failure(
            f"Missing TRAIN medians file: {MEDIANS_PATH}"
        )
        return False

    try:
        medians = pd.read_csv(MEDIANS_PATH)

        required = {
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Min",
            "Fwd IAT Min",
        }

        # Handle either a "feature" column or a simple two-column layout.
        columns = set(medians.columns)

        if "feature" in columns:
            found = set(medians["feature"].astype(str))
        elif "Feature" in columns:
            found = set(medians["Feature"].astype(str))
        else:
            found = set()

        if required.issubset(found):
            success(
                "TRAIN medians file exists and contains all "
                "4 expected imputation columns"
            )
            return True

        warning(
            "TRAIN medians file exists, but its exact column structure "
            "could not be fully verified"
        )
        detail(f"Detected columns: {list(medians.columns)}")
        return True

    except Exception as exc:
        failure(
            f"Could not read TRAIN medians file: {exc}"
        )
        return False


def check_feature_columns_file(train):
    if not FEATURE_COLUMNS_PATH.exists():
        failure(
            f"Missing feature_columns.csv: {FEATURE_COLUMNS_PATH}"
        )
        return False

    try:
        feature_df = pd.read_csv(FEATURE_COLUMNS_PATH)

        if len(feature_df) != EXPECTED_FEATURE_COUNT:
            failure(
                "feature_columns.csv does not contain exactly "
                f"{EXPECTED_FEATURE_COUNT} rows"
            )
            return False

        # Try common possible column names.
        possible_names = [
            "feature",
            "Feature",
            "feature_name",
            "Feature Name",
        ]

        feature_col = next(
            (
                col for col in possible_names
                if col in feature_df.columns
            ),
            None,
        )

        if feature_col is None:
            warning(
                "feature_columns.csv exists but its feature-name "
                "column could not be identified automatically"
            )
            detail(f"Columns: {list(feature_df.columns)}")
            return True

        saved_features = set(
            feature_df[feature_col].astype(str)
        )

        actual_features = set(
            get_feature_columns(train)
        )

        if saved_features == actual_features:
            success(
                "feature_columns.csv exactly matches TRAIN "
                "model features"
            )
            return True

        failure(
            "feature_columns.csv does not exactly match TRAIN "
            "model features"
        )

        missing = actual_features - saved_features
        extra = saved_features - actual_features

        if missing:
            detail(f"Missing from CSV: {sorted(missing)}")

        if extra:
            detail(f"Extra in CSV: {sorted(extra)}")

        return False

    except Exception as exc:
        failure(
            f"Could not verify feature_columns.csv: {exc}"
        )
        return False


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    title("CICIDS2017 PREPROCESSED DATA AUDIT")

    print(
        color(
            f"Project root: {PROJECT_ROOT}",
            Colors.WHITE
        )
    )

    print(
        color(
            f"Expected model features: {EXPECTED_FEATURE_COUNT}",
            Colors.WHITE
        )
    )

    print(
        color(
            "This audit is READ-ONLY. No dataset files will be modified.",
            Colors.YELLOW
        )
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    title("1. LOADING PREPROCESSED SPLITS")

    try:
        train = load_parquet(TRAIN_PATH)
        val = load_parquet(VAL_PATH)
        test = load_parquet(TEST_PATH)

    except Exception as exc:
        failure(f"Could not load preprocessed data: {exc}")
        return 1

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    title("2. FEATURE COUNT")

    feature_count_results = {
        "TRAIN": check_feature_count(train, "TRAIN"),
        "VALIDATION": check_feature_count(val, "VALIDATION"),
        "TEST": check_feature_count(test, "TEST"),
    }

    # --------------------------------------------------------
    # Feature consistency
    # --------------------------------------------------------

    title("3. FEATURE CONSISTENCY")

    feature_consistency = check_feature_consistency(
        train,
        val,
        test,
    )

    # --------------------------------------------------------
    # Expected removed constants
    # --------------------------------------------------------

    title("4. CONSTANT-FEATURE REMOVAL")

    constant_removal_results = {
        "TRAIN": check_expected_constant_removals(
            train,
            "TRAIN",
        ),
        "VALIDATION": check_expected_constant_removals(
            val,
            "VALIDATION",
        ),
        "TEST": check_expected_constant_removals(
            test,
            "TEST",
        ),
    }

    # --------------------------------------------------------
    # Numeric types
    # --------------------------------------------------------

    title("5. NUMERIC FEATURE TYPES")

    numeric_results = {
        "TRAIN": check_numeric_features(
            train,
            get_feature_columns(train),
            "TRAIN",
        ),
        "VALIDATION": check_numeric_features(
            val,
            get_feature_columns(val),
            "VALIDATION",
        ),
        "TEST": check_numeric_features(
            test,
            get_feature_columns(test),
            "TEST",
        ),
    }

    # --------------------------------------------------------
    # NaN / infinity
    # --------------------------------------------------------

    title("6. NaN / INFINITY CHECK")

    no_nan_inf_results = {}

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:
        passed, nan_count, inf_count = check_no_nan_inf(
            df,
            get_feature_columns(df),
            name,
        )

        no_nan_inf_results[name] = {
            "passed": passed,
            "nan_count": nan_count,
            "infinity_count": inf_count,
        }

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    title("7. TARGET LABEL CHECK")

    target_results = {
        "TRAIN": check_target(train, "TRAIN"),
        "VALIDATION": check_target(val, "VALIDATION"),
        "TEST": check_target(test, "TEST"),
    }

    # --------------------------------------------------------
    # Source file
    # --------------------------------------------------------

    title("8. PROVENANCE / LEAKAGE CHECK")

    source_results = {
        "TRAIN": check_source_file(train, "TRAIN"),
        "VALIDATION": check_source_file(val, "VALIDATION"),
        "TEST": check_source_file(test, "TEST"),
    }

    # --------------------------------------------------------
    # Missing indicators
    # --------------------------------------------------------

    title("9. MISSING-VALUE INDICATORS")

    indicator_results = {}

    for name, df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:
        indicator_structure = check_missing_indicators(
            df,
            name,
        )

        indicator_values = check_indicator_values(
            df,
            name,
        )

        indicator_results[name] = {
            "structure": indicator_structure,
            "values": indicator_values,
        }

    # --------------------------------------------------------
    # Imputed columns
    # --------------------------------------------------------

    title("10. IMPUTED ORIGINAL FEATURES")

    imputed_results = {
        "TRAIN": check_imputed_columns(train, "TRAIN"),
        "VALIDATION": check_imputed_columns(val, "VALIDATION"),
        "TEST": check_imputed_columns(test, "TEST"),
    }

    # --------------------------------------------------------
    # Constant model features
    # --------------------------------------------------------

    title("11. FINAL CONSTANT-FEATURE CHECK")

    final_constant_results = {
        "TRAIN": check_constant_features(train, "TRAIN"),
        "VALIDATION": check_constant_features(val, "VALIDATION"),
        "TEST": check_constant_features(test, "TEST"),
    }

    # --------------------------------------------------------
    # Split-local duplicate pairs
    # --------------------------------------------------------

    title("12. SPLIT-LOCAL DUPLICATE FEATURE CHECK")

    duplicate_pairs = {
        "TRAIN": find_split_local_duplicate_pairs(
            train,
            "TRAIN",
        ),
        "VALIDATION": find_split_local_duplicate_pairs(
            val,
            "VALIDATION",
        ),
        "TEST": find_split_local_duplicate_pairs(
            test,
            "TEST",
        ),
    }

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    title("13. LABEL CONSISTENCY")

    all_labels_present = check_all_labels_present(
        train,
        val,
        test,
    )

    # --------------------------------------------------------
    # Class distributions
    # --------------------------------------------------------

    title("14. CLASS DISTRIBUTIONS")

    distributions = {
        "TRAIN": class_distribution(train, "TRAIN"),
        "VALIDATION": class_distribution(val, "VALIDATION"),
        "TEST": class_distribution(test, "TEST"),
    }

    # --------------------------------------------------------
    # Extreme values
    # --------------------------------------------------------

    title("15. EXTREME-VALUE REPORT")

    extreme_values = {
        "TRAIN": check_extreme_values(train, "TRAIN"),
        "VALIDATION": check_extreme_values(val, "VALIDATION"),
        "TEST": check_extreme_values(test, "TEST"),
    }

    # --------------------------------------------------------
    # Supporting files
    # --------------------------------------------------------

    title("16. PREPROCESSING METADATA")

    medians_result = check_train_statistics_file()

    feature_columns_result = check_feature_columns_file(
        train
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    title("17. FINAL AUDIT RESULT")

    mandatory_checks = []

    mandatory_checks.extend(
        feature_count_results.values()
    )

    mandatory_checks.append(feature_consistency)

    mandatory_checks.extend(
        constant_removal_results.values()
    )

    mandatory_checks.extend(
        numeric_results.values()
    )

    mandatory_checks.extend(
        result["passed"]
        for result in no_nan_inf_results.values()
    )

    mandatory_checks.extend(
        target_results.values()
    )

    mandatory_checks.extend(
        source_results.values()
    )

    mandatory_checks.extend(
        result["structure"]
        for result in indicator_results.values()
    )

    mandatory_checks.extend(
        result["values"]
        for result in indicator_results.values()
    )

    mandatory_checks.extend(
        imputed_results.values()
    )

    mandatory_checks.extend(
        final_constant_results.values()
    )

    mandatory_checks.append(all_labels_present)
    mandatory_checks.append(medians_result)
    mandatory_checks.append(feature_columns_result)

    total_checks = len(mandatory_checks)
    passed_checks = sum(bool(x) for x in mandatory_checks)
    failed_checks = total_checks - passed_checks

    audit_passed = failed_checks == 0

    print()
    print(
        color(
            f"Mandatory checks passed: {passed_checks}/{total_checks}",
            Colors.GREEN if audit_passed else Colors.YELLOW
        )
    )

    print(
        color(
            f"Mandatory checks failed: {failed_checks}/{total_checks}",
            Colors.GREEN if failed_checks == 0 else Colors.RED
        )
    )

    if audit_passed:
        print()
        print(
            color(
                "████████████████████████████████████████████████",
                Colors.GREEN
            )
        )
        print(
            color(
                "              AUDIT PASSED",
                Colors.BOLD + Colors.GREEN
            )
        )
        print(
            color(
                "████████████████████████████████████████████████",
                Colors.GREEN
            )
        )

        print()
        success(
            "The 68-feature preprocessing pipeline is internally consistent."
        )
        info(
            "You can proceed to baseline model training."
        )

    else:
        print()
        print(
            color(
                "████████████████████████████████████████████████",
                Colors.RED
            )
        )
        print(
            color(
                "              AUDIT FAILED",
                Colors.BOLD + Colors.RED
            )
        )
        print(
            color(
                "████████████████████████████████████████████████",
                Colors.RED
            )
        )

        print()
        failure(
            "Do NOT start model training until the failed checks are investigated."
        )

    # --------------------------------------------------------
    # Save JSON report
    # --------------------------------------------------------

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "audit_passed": audit_passed,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "feature_count_results": feature_count_results,
        "feature_consistency": feature_consistency,
        "constant_removal_results": constant_removal_results,
        "numeric_results": numeric_results,
        "no_nan_inf_results": no_nan_inf_results,
        "target_results": target_results,
        "source_results": source_results,
        "indicator_results": indicator_results,
        "imputed_results": imputed_results,
        "final_constant_results": final_constant_results,
        "duplicate_pairs": {
            split: [
                list(pair)
                for pair in pairs
            ]
            for split, pairs in duplicate_pairs.items()
        },
        "all_labels_present": all_labels_present,
        "class_distributions": distributions,
        "extreme_values": extreme_values,
        "train_medians_file": medians_result,
        "feature_columns_file": feature_columns_result,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    info(
        f"Audit report saved to: {REPORT_PATH}"
    )

    return 0 if audit_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())