
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "preprocessed"

TRAIN_INPUT = DATA_DIR / "train.parquet"
VAL_INPUT = DATA_DIR / "validation.parquet"
TEST_INPUT = DATA_DIR / "test.parquet"

TRAIN_OUTPUT = OUTPUT_DIR / "train.parquet"
VAL_OUTPUT = OUTPUT_DIR / "validation.parquet"
TEST_OUTPUT = OUTPUT_DIR / "test.parquet"

MEDIANS_OUTPUT = OUTPUT_DIR / "train_medians.csv"
FEATURES_OUTPUT = OUTPUT_DIR / "feature_columns.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "Label"
PROVENANCE_COLUMN = "source_file"

# These were verified to be constant zero in the cleaned CICIDS2017
# dataset and are excluded from MODEL INPUT.
CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# These original columns contain missing/invalid values that need
# TRAIN-only median imputation.
IMPUTE_COLUMNS = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Min",
    "Fwd IAT Min",
]

# Missingness indicators are created ONLY for these three.
#
# Flow Packets/s_missing is intentionally NOT created because its
# missingness pattern is exactly identical to Flow Bytes/s_missing.
MISSING_INDICATOR_COLUMNS = [
    "Flow Bytes/s",
    "Flow IAT Min",
    "Fwd IAT Min",
]

EXPECTED_MODEL_FEATURE_COUNT = 68


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


def ctext(text, colour):
    return f"{colour}{text}{Colors.RESET}"


def header(text):
    print()
    print(ctext("=" * 72, Colors.CYAN))
    print(ctext(text, Colors.BOLD + Colors.CYAN))
    print(ctext("=" * 72, Colors.CYAN))


def info(text):
    print(ctext(f"[INFO] {text}", Colors.BLUE))


def success(text):
    print(ctext(f"[PASS] {text}", Colors.GREEN))


def warning(text):
    print(ctext(f"[WARN] {text}", Colors.YELLOW))


def error(text):
    print(ctext(f"[FAIL] {text}", Colors.RED))


# ============================================================
# LOAD
# ============================================================

def load_split(path, split_name):
    if not path.exists():
        raise FileNotFoundError(
            f"{split_name} split not found: {path}"
        )

    info(f"Loading {split_name}: {path}")

    df = pd.read_parquet(path)

    success(
        f"{split_name}: {len(df):,} rows × "
        f"{len(df.columns)} columns"
    )

    return df


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input_columns(df, split_name):
    required = {
        TARGET_COLUMN,
        PROVENANCE_COLUMN,
        *CONSTANT_FEATURES,
        *IMPUTE_COLUMNS,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{split_name} is missing required columns: {missing}"
        )

    success(
        f"{split_name}: all required preprocessing columns are present"
    )


# ============================================================
# CREATE MISSING INDICATORS
# ============================================================

def create_missing_indicators(df, split_name):
    info(
        f"{split_name}: creating missing-value indicators"
    )

    for column in MISSING_INDICATOR_COLUMNS:

        indicator_name = f"{column}_missing"

        if indicator_name in df.columns:
            raise ValueError(
                f"{split_name}: indicator already exists: "
                f"{indicator_name}"
            )

        df[indicator_name] = (
            df[column]
            .isna()
            .astype(np.int8)
        )

        missing_count = int(df[column].isna().sum())

        info(
            f"{split_name}: {indicator_name} = "
            f"{missing_count:,} missing values"
        )

    warning(
        "Flow Packets/s_missing is intentionally NOT created "
        "because its missingness pattern is identical to "
        "Flow Bytes/s_missing."
    )

    return df


# ============================================================
# REMOVE CONSTANT MODEL FEATURES
# ============================================================

def remove_constant_features(df, split_name):
    info(
        f"{split_name}: removing 8 verified constant-zero "
        f"features from MODEL INPUT"
    )

    missing = [
        column
        for column in CONSTANT_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{split_name}: expected constant features are missing "
            f"before removal: {missing}"
        )

    # Verify they really are constant zero before removing them.
    for column in CONSTANT_FEATURES:

        unique_values = df[column].dropna().unique()

        if len(unique_values) != 1 or unique_values[0] != 0:
            raise ValueError(
                f"{split_name}: expected constant-zero feature "
                f"is no longer constant zero: {column}. "
                f"Values found: {unique_values[:10]}"
            )

    df = df.drop(
        columns=CONSTANT_FEATURES
    )

    success(
        f"{split_name}: removed {len(CONSTANT_FEATURES)} "
        f"constant-zero features"
    )

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(df):
    excluded = {
        TARGET_COLUMN,
        PROVENANCE_COLUMN,
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# ============================================================
# TRAIN-ONLY MEDIANS
# ============================================================

def calculate_train_medians(train):
    header("CALCULATING TRAIN-ONLY MEDIANS")

    medians = {}

    for column in IMPUTE_COLUMNS:

        if column not in train.columns:
            raise ValueError(
                f"TRAIN is missing imputation column: {column}"
            )

        median_value = train[column].median()

        if pd.isna(median_value):
            raise ValueError(
                f"TRAIN median is NaN for column: {column}"
            )

        medians[column] = float(median_value)

        print(
            ctext(
                f"{column:<25} {median_value}",
                Colors.WHITE
            )
        )

    success(
        "All medians calculated from TRAIN only."
    )

    return medians


# ============================================================
# APPLY MEDIAN IMPUTATION
# ============================================================

def apply_medians(df, medians, split_name):

    info(
        f"{split_name}: applying TRAIN-derived medians"
    )

    for column, median_value in medians.items():

        if column not in df.columns:
            raise ValueError(
                f"{split_name}: missing imputation column: {column}"
            )

        missing_before = int(df[column].isna().sum())

        if missing_before > 0:
            df[column] = df[column].fillna(
                median_value
            )

        missing_after = int(df[column].isna().sum())

        if missing_after != 0:
            raise RuntimeError(
                f"{split_name}: imputation failed for {column}"
            )

        if missing_before > 0:
            info(
                f"{split_name}: {column}: "
                f"{missing_before:,} values imputed"
            )

    return df


# ============================================================
# NUMERIC VALIDATION
# ============================================================

def validate_numeric_features(df, split_name):

    feature_columns = get_model_features(df)

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    if non_numeric:
        raise TypeError(
            f"{split_name}: non-numeric model features found: "
            f"{non_numeric}"
        )

    success(
        f"{split_name}: all {len(feature_columns)} "
        f"model features are numeric"
    )


# ============================================================
# NaN / INFINITY VALIDATION
# ============================================================

def validate_no_nan_inf(df, split_name):

    feature_columns = get_model_features(df)

    numeric = df[feature_columns].select_dtypes(
        include=[np.number]
    )

    nan_count = int(
        numeric.isna().sum().sum()
    )

    values = numeric.to_numpy(
        dtype=np.float64
    )

    inf_count = int(
        np.isinf(values).sum()
    )

    if nan_count != 0:
        raise ValueError(
            f"{split_name}: {nan_count:,} NaN values "
            f"remain in model features"
        )

    if inf_count != 0:
        raise ValueError(
            f"{split_name}: {inf_count:,} infinity values "
            f"remain in model features"
        )

    success(
        f"{split_name}: no NaN or infinity values "
        f"in model features"
    )


# ============================================================
# INDICATOR VALIDATION
# ============================================================

def validate_indicators(df, split_name):

    expected = {
        f"{column}_missing"
        for column in MISSING_INDICATOR_COLUMNS
    }

    actual = {
        column
        for column in df.columns
        if column.endswith("_missing")
    }

    if actual != expected:
        raise ValueError(
            f"{split_name}: incorrect missing indicators.\n"
            f"Expected: {sorted(expected)}\n"
            f"Found: {sorted(actual)}"
        )

    for column in sorted(expected):

        values = set(
            df[column].dropna().unique()
        )

        if not values.issubset({0, 1}):
            raise ValueError(
                f"{split_name}: indicator {column} "
                f"contains values outside 0/1: {values}"
            )

    success(
        f"{split_name}: missing indicators are correct "
        f"and contain only 0/1"
    )


# ============================================================
# FEATURE CONSISTENCY
# ============================================================

def validate_feature_consistency(
    train,
    validation,
    test,
):

    train_features = get_model_features(train)
    validation_features = get_model_features(validation)
    test_features = get_model_features(test)

    if train_features != validation_features:
        raise ValueError(
            "TRAIN and VALIDATION model feature order differs."
        )

    if train_features != test_features:
        raise ValueError(
            "TRAIN and TEST model feature order differs."
        )

    if len(train_features) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_MODEL_FEATURE_COUNT} model features, "
            f"found {len(train_features)}."
        )

    success(
        "TRAIN / VALIDATION / TEST have identical "
        f"{len(train_features)}-feature order"
    )


# ============================================================
# SAVE FEATURE LIST
# ============================================================

def save_feature_columns(train):

    feature_columns = get_model_features(train)

    if len(feature_columns) != EXPECTED_MODEL_FEATURE_COUNT:
        raise ValueError(
            f"Cannot save feature list: expected "
            f"{EXPECTED_MODEL_FEATURE_COUNT} features, "
            f"found {len(feature_columns)}."
        )

    feature_df = pd.DataFrame({
        "feature": feature_columns
    })

    feature_df.to_csv(
        FEATURES_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    success(
        f"Saved feature list: {FEATURES_OUTPUT}"
    )


# ============================================================
# SAVE MEDIANS
# ============================================================

def save_medians(medians):

    median_df = pd.DataFrame({
        "feature": list(medians.keys()),
        "median": list(medians.values()),
    })

    median_df.to_csv(
        MEDIANS_OUTPUT,
        index=False,
        encoding="utf-8",
    )

    success(
        f"Saved TRAIN medians: {MEDIANS_OUTPUT}"
    )


# ============================================================
# SAVE SPLIT
# ============================================================

def save_split(df, path, split_name):

    df.to_parquet(
        path,
        index=False,
        engine="pyarrow",
    )

    success(
        f"Saved {split_name}: {path}"
    )

    info(
        f"{split_name}: {len(df):,} rows × "
        f"{len(df.columns)} total columns"
    )

    info(
        f"{split_name}: "
        f"{len(get_model_features(df))} model features"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    header("CICIDS2017 PREPROCESSING")

    print(
        ctext(
            "This script does NOT modify the original split files.",
            Colors.YELLOW
        )
    )

    print(
        ctext(
            "TRAIN medians are calculated before being applied "
            "to any split.",
            Colors.YELLOW
        )
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load original locked splits
    # --------------------------------------------------------

    header("1. LOADING LOCKED SPLITS")

    train = load_split(
        TRAIN_INPUT,
        "TRAIN",
    )

    validation = load_split(
        VAL_INPUT,
        "VALIDATION",
    )

    test = load_split(
        TEST_INPUT,
        "TEST",
    )

    # --------------------------------------------------------
    # Validate original split structure
    # --------------------------------------------------------

    header("2. VALIDATING INPUT COLUMNS")

    validate_input_columns(
        train,
        "TRAIN",
    )

    validate_input_columns(
        validation,
        "VALIDATION",
    )

    validate_input_columns(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # Create missing indicators
    #
    # IMPORTANT:
    # Do this BEFORE imputation so that the indicators preserve
    # the original missingness information.
    # --------------------------------------------------------

    header("3. CREATING MISSING-VALUE INDICATORS")

    train = create_missing_indicators(
        train,
        "TRAIN",
    )

    validation = create_missing_indicators(
        validation,
        "VALIDATION",
    )

    test = create_missing_indicators(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # Remove the 8 verified constant-zero features
    # --------------------------------------------------------

    header("4. REMOVING CONSTANT-ZERO FEATURES")

    train = remove_constant_features(
        train,
        "TRAIN",
    )

    validation = remove_constant_features(
        validation,
        "VALIDATION",
    )

    test = remove_constant_features(
        test,
        "TEST",
    )

    # --------------------------------------------------------
    # Calculate TRAIN-only medians
    # --------------------------------------------------------

    medians = calculate_train_medians(
        train
    )

    # --------------------------------------------------------
    # Apply TRAIN medians
    # --------------------------------------------------------

    header("5. MEDIAN IMPUTATION")

    train = apply_medians(
        train,
        medians,
        "TRAIN",
    )

    validation = apply_medians(
        validation,
        medians,
        "VALIDATION",
    )

    test = apply_medians(
        test,
        medians,
        "TEST",
    )

    # --------------------------------------------------------
    # Validate model features
    # --------------------------------------------------------

    header("6. PREPROCESSING VALIDATION")

    validate_numeric_features(
        train,
        "TRAIN",
    )

    validate_numeric_features(
        validation,
        "VALIDATION",
    )

    validate_numeric_features(
        test,
        "TEST",
    )

    validate_no_nan_inf(
        train,
        "TRAIN",
    )

    validate_no_nan_inf(
        validation,
        "VALIDATION",
    )

    validate_no_nan_inf(
        test,
        "TEST",
    )

    validate_indicators(
        train,
        "TRAIN",
    )

    validate_indicators(
        validation,
        "VALIDATION",
    )

    validate_indicators(
        test,
        "TEST",
    )

    validate_feature_consistency(
        train,
        validation,
        test,
    )

    # --------------------------------------------------------
    # Verify targets
    # --------------------------------------------------------

    header("7. TARGET VALIDATION")

    for df, split_name in [
        (train, "TRAIN"),
        (validation, "VALIDATION"),
        (test, "TEST"),
    ]:

        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"{split_name}: target column '{TARGET_COLUMN}' "
                f"is missing."
            )

        if df[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"{split_name}: missing target labels detected."
            )

        success(
            f"{split_name}: target labels present"
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    header("8. SAVING PREPROCESSED DATA")

    save_split(
        train,
        TRAIN_OUTPUT,
        "TRAIN",
    )

    save_split(
        validation,
        VAL_OUTPUT,
        "VALIDATION",
    )

    save_split(
        test,
        TEST_OUTPUT,
        "TEST",
    )

    save_medians(
        medians
    )

    save_feature_columns(
        train
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    header("9. FINAL PREPROCESSING SUMMARY")

    train_features = get_model_features(train)
    validation_features = get_model_features(validation)
    test_features = get_model_features(test)

    print(
        ctext(
            f"TRAIN:      {len(train):,} rows × "
            f"{len(train.columns)} columns",
            Colors.WHITE
        )
    )

    print(
        ctext(
            f"VALIDATION: {len(validation):,} rows × "
            f"{len(validation.columns)} columns",
            Colors.WHITE
        )
    )

    print(
        ctext(
            f"TEST:       {len(test):,} rows × "
            f"{len(test.columns)} columns",
            Colors.WHITE
        )
    )

    print()

    print(
        ctext(
            f"Model features: {len(train_features)}",
            Colors.WHITE
        )
    )

    print(
        ctext(
            "Constant-zero features removed: 8",
            Colors.WHITE
        )
    )

    print(
        ctext(
            "Missing indicators created: 3",
            Colors.WHITE
        )
    )

    print(
        ctext(
            "Missing indicators NOT created: Flow Packets/s_missing",
            Colors.WHITE
        )
    )

    print(
        ctext(
            "Imputation statistics: TRAIN only",
            Colors.WHITE
        )
    )

    print()

    if (
        len(train_features) == EXPECTED_MODEL_FEATURE_COUNT
        and train_features == validation_features
        and train_features == test_features
    ):
        print(
            ctext(
                "████████████████████████████████████████████████",
                Colors.GREEN
            )
        )

        print(
            ctext(
                "        PREPROCESSING COMPLETED SUCCESSFULLY",
                Colors.BOLD + Colors.GREEN
            )
        )

        print(
            ctext(
                f"        FINAL MODEL INPUT = "
                f"{EXPECTED_MODEL_FEATURE_COUNT} FEATURES",
                Colors.BOLD + Colors.GREEN
            )
        )

        print(
            ctext(
                "████████████████████████████████████████████████",
                Colors.GREEN
            )
        )

    else:
        raise RuntimeError(
            "Final preprocessing validation failed."
        )


if __name__ == "__main__":
    main()
