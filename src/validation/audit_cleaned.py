from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"

EXPECTED_ROWS = 2_823_573
EXPECTED_COLUMNS = 75
EXPECTED_LABEL_COUNT = 15

EXPECTED_SOURCE_FILES = {
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
}

DUPLICATE_COLUMNS_THAT_SHOULD_BE_GONE = {
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "CWE Flag Count",
    "SYN Flag Count",
}

CONSTANT_ZERO_COLUMNS = {
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}

NEGATIVE_VALUES_NOT_ALLOWED = {
    "Flow Duration",
    "Bwd Header Length",
    "Fwd Header Length",
    "min_seg_size_forward",
}


# ============================================================
# HELPERS
# ============================================================

def fail(message):
    print(f"  [FAIL] {message}")
    return False


def passed(message):
    print(f"  [PASS] {message}")
    return True


# ============================================================
# MAIN AUDIT
# ============================================================

def main():
    print("=" * 70)
    print("CICIDS2017 POST-CLEAN AUDIT")
    print("=" * 70)

    if not CLEANED_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found:\n{CLEANED_PATH}"
        )

    print(f"\nLoading:\n{CLEANED_PATH}")

    df = pd.read_parquet(CLEANED_PATH)

    all_passed = True

    # --------------------------------------------------------
    # 1. Shape
    # --------------------------------------------------------

    print("\n[1/10] Checking dataset shape...")

    rows, columns = df.shape

    print(f"  Rows    : {rows:,}")
    print(f"  Columns : {columns}")

    if rows == EXPECTED_ROWS:
        passed(f"Row count is {rows:,}")
    else:
        all_passed = fail(
            f"Expected {EXPECTED_ROWS:,} rows, found {rows:,}"
        ) or all_passed

    if columns == EXPECTED_COLUMNS:
        passed(f"Column count is {columns}")
    else:
        all_passed = fail(
            f"Expected {EXPECTED_COLUMNS} columns, found {columns}"
        ) or all_passed

    # --------------------------------------------------------
    # 2. Required columns
    # --------------------------------------------------------

    print("\n[2/10] Checking required columns...")

    required_columns = {"Label", "source_file"}

    missing_required = required_columns - set(df.columns)

    if not missing_required:
        passed("Label and source_file are present")
    else:
        all_passed = fail(
            f"Missing required columns: {sorted(missing_required)}"
        ) or all_passed

    # --------------------------------------------------------
    # 3. Labels
    # --------------------------------------------------------

    print("\n[3/10] Checking labels...")

    if "Label" in df.columns:

        missing_labels = int(df["Label"].isna().sum())

        unique_labels = df["Label"].nunique(dropna=True)

        print(f"  Unique labels : {unique_labels}")
        print(f"  Missing labels: {missing_labels}")

        if unique_labels == EXPECTED_LABEL_COUNT:
            passed(f"Exactly {EXPECTED_LABEL_COUNT} labels remain")
        else:
            all_passed = fail(
                f"Expected {EXPECTED_LABEL_COUNT} labels, found {unique_labels}"
            ) or all_passed

        if missing_labels == 0:
            passed("No missing labels")
        else:
            all_passed = fail(
                f"{missing_labels:,} missing labels found"
            ) or all_passed

        print("\n  Label distribution:")
        label_counts = df["Label"].value_counts()

        for label, count in label_counts.items():
            percentage = count / len(df) * 100
            print(f"    {label:<35} {count:>10,} ({percentage:>7.3f}%)")

    # --------------------------------------------------------
    # 4. Infinity
    # --------------------------------------------------------

    print("\n[4/10] Checking infinity values...")

    numeric_df = df.select_dtypes(include=[np.number])

    positive_inf = int(np.isposinf(numeric_df).sum().sum())
    negative_inf = int(np.isneginf(numeric_df).sum().sum())

    print(f"  +Infinity : {positive_inf:,}")
    print(f"  -Infinity : {negative_inf:,}")

    if positive_inf == 0 and negative_inf == 0:
        passed("No infinity values remain")
    else:
        all_passed = fail(
            f"Infinity values remain: +inf={positive_inf:,}, "
            f"-inf={negative_inf:,}"
        ) or all_passed

    # --------------------------------------------------------
    # 5. NaN audit
    # --------------------------------------------------------

    print("\n[5/10] Checking NaN values...")

    nan_counts = df.isna().sum()
    total_nan = int(nan_counts.sum())

    print(f"  Total NaN cells: {total_nan:,}")

    nonzero_nan = nan_counts[nan_counts > 0]

    if len(nonzero_nan) == 0:
        print("  No NaN values remain.")
    else:
        print("  NaN-containing columns:")

        for column, count in nonzero_nan.sort_values(
            ascending=False
        ).items():
            print(f"    {column:<35} {count:>10,}")

    # NaNs are expected after the cell-level cleaning.
    # We do NOT fail merely because NaNs exist.

    # --------------------------------------------------------
    # 6. Invalid negative values
    # --------------------------------------------------------

    print("\n[6/10] Checking confirmed-invalid negative values...")

    negative_problem_found = False

    for column in NEGATIVE_VALUES_NOT_ALLOWED:

        if column not in df.columns:
            continue

        series = pd.to_numeric(df[column], errors="coerce")

        negative_count = int((series < 0).sum())

        print(f"  {column:<30} {negative_count:,}")

        if negative_count > 0:
            negative_problem_found = True

    if not negative_problem_found:
        passed("No confirmed-invalid negative values remain")
    else:
        all_passed = fail(
            "Confirmed-invalid negative values remain"
        ) or all_passed

    # --------------------------------------------------------
    # 7. Duplicate columns
    # --------------------------------------------------------

    print("\n[7/10] Checking removed duplicate columns...")

    remaining_duplicate_columns = (
        DUPLICATE_COLUMNS_THAT_SHOULD_BE_GONE
        & set(df.columns)
    )

    if not remaining_duplicate_columns:
        passed("All 5 verified duplicate columns are absent")
    else:
        all_passed = fail(
            "Duplicate columns still present: "
            f"{sorted(remaining_duplicate_columns)}"
        ) or all_passed

    # --------------------------------------------------------
    # 8. Constant-zero columns
    # --------------------------------------------------------

    print("\n[8/10] Checking retained constant-zero columns...")

    missing_constant_columns = (
        CONSTANT_ZERO_COLUMNS - set(df.columns)
    )

    bad_constant_columns = []

    for column in CONSTANT_ZERO_COLUMNS:
        if column in df.columns:

            series = pd.to_numeric(df[column], errors="coerce")

            # Ignore NaN when checking whether all actual values are zero.
            non_nan = series.dropna()

            if len(non_nan) > 0 and not (non_nan == 0).all():
                bad_constant_columns.append(column)

    if not missing_constant_columns and not bad_constant_columns:
        passed("All 8 expected constant-zero columns remain and are zero")
    else:
        if missing_constant_columns:
            all_passed = fail(
                f"Missing constant-zero columns: "
                f"{sorted(missing_constant_columns)}"
            ) or all_passed

        if bad_constant_columns:
            all_passed = fail(
                f"Columns expected to be all-zero are not: "
                f"{bad_constant_columns}"
            ) or all_passed

    # --------------------------------------------------------
    # 9. source_file audit
    # --------------------------------------------------------

    print("\n[9/10] Checking source_file provenance...")

    if "source_file" in df.columns:

        missing_source = int(df["source_file"].isna().sum())

        source_values = set(df["source_file"].dropna().unique())

        unexpected_sources = source_values - EXPECTED_SOURCE_FILES
        missing_sources = EXPECTED_SOURCE_FILES - source_values

        print(f"  Unique source files : {len(source_values)}")
        print(f"  Missing source_file : {missing_source}")

        if missing_source == 0:
            passed("No missing source_file values")
        else:
            all_passed = fail(
                f"{missing_source:,} missing source_file values"
            ) or all_passed

        if not unexpected_sources:
            passed("No unexpected source files")
        else:
            all_passed = fail(
                f"Unexpected source files: {sorted(unexpected_sources)}"
            ) or all_passed

        if not missing_sources:
            passed("All 8 original source files are represented")
        else:
            all_passed = fail(
                f"Missing source files: {sorted(missing_sources)}"
            ) or all_passed

    # --------------------------------------------------------
    # 10. Conflicting feature vectors
    # --------------------------------------------------------

    print("\n[10/10] Checking conflicting feature-vector groups...")

    feature_columns = [
        column
        for column in df.columns
        if column not in {"Label", "source_file"}
    ]

    print(f"  Feature columns checked: {len(feature_columns)}")

    # Use exact feature values. NaN is normalized to a sentinel so
    # identical missing-value patterns are treated consistently.
    conflict_groups = 0
    conflict_rows = 0

    feature_subset = df[feature_columns].copy()

    # Make NaN comparable during grouping.
    feature_subset = feature_subset.astype(object).where(
        pd.notna(feature_subset),
        "__NAN__"
    )

    temp = feature_subset.copy()
    temp["__Label__"] = df["Label"].values

    grouped = temp.groupby(
        feature_columns,
        dropna=False,
        sort=False
    )["__Label__"].nunique()

    conflicting_keys = grouped[grouped > 1]

    conflict_groups = len(conflicting_keys)

    if conflict_groups > 0:
        # Re-check rows belonging to conflicting groups.
        conflict_group_sizes = grouped[grouped > 1]
        conflict_rows = int(
            temp.merge(
                conflict_group_sizes.rename("__nlabels__").reset_index(),
                on=feature_columns,
                how="inner"
            ).shape[0]
        )

    print(f"  Conflicting groups: {conflict_groups:,}")
    print(f"  Rows involved     : {conflict_rows:,}")

    if conflict_groups == 0:
        passed("No conflicting feature-vector groups remain")
    else:
        all_passed = fail(
            f"{conflict_groups:,} conflicting feature-vector groups remain"
        ) or all_passed

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if all_passed:
        print("POST-CLEAN AUDIT PASSED")
        print("=" * 70)
        print("\nThe cleaned dataset is ready for the splitting stage.")
    else:
        print("POST-CLEAN AUDIT FOUND PROBLEMS")
        print("=" * 70)
        print("\nDo NOT create split.py yet. Review the failed checks first.")

    print("\nDataset:")
    print(f"  {CLEANED_PATH}")
    print(f"  Shape: {rows:,} rows × {columns} columns")


if __name__ == "__main__":
    main()