from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned.parquet"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = OUTPUT_DIR / "train.parquet"
VAL_PATH = OUTPUT_DIR / "validation.parquet"
TEST_PATH = OUTPUT_DIR / "test.parquet"

REPORT_PATH = PROJECT_ROOT / "reports" / "split_summary.md"

RANDOM_STATE = 42

TRAIN_TARGET = 0.70
VAL_TARGET = 0.15
TEST_TARGET = 0.15

TARGET = "Label"
SOURCE_COLUMN = "source_file"


# ============================================================
# CREATE EXACT FEATURE-VECTOR GROUP IDS
# ============================================================

def make_group_ids(df, feature_columns):
    """
    Create one ID for every exact feature vector.

    NaN values are treated consistently so that rows with
    identical feature values, including NaNs, belong to the
    same group.
    """

    features = df[feature_columns].astype(object).where(
        pd.notna(df[feature_columns]),
        "__NAN__"
    )

    return pd.util.hash_pandas_object(
        features,
        index=False
    )


# ============================================================
# SIZE-AWARE GROUP ASSIGNMENT
# ============================================================

def assign_groups_for_class(
    class_groups,
    train_target,
    val_target,
    test_target,
    rng,
):
    """
    Assign whole groups belonging to ONE class to
    train / validation / test.

    The objective is to keep the ROW counts as close as
    possible to the desired 70/15/15 targets.

    Groups are never split.
    """

    class_groups = class_groups.copy()

    # Random tie-breaker.
    class_groups["_random"] = rng.random(
        len(class_groups)
    )

    # Process larger groups first because they have the
    # greatest effect on the final distribution.
    class_groups = class_groups.sort_values(
        ["group_size", "_random"],
        ascending=[False, True]
    )

    total_rows = int(
        class_groups["group_size"].sum()
    )

    target_rows = {
        "train": total_rows * train_target,
        "validation": total_rows * val_target,
        "test": total_rows * test_target,
    }

    current_rows = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }

    assignments = {}

    split_names = [
        "train",
        "validation",
        "test",
    ]

    # --------------------------------------------------------
    # Assign each whole group.
    # --------------------------------------------------------

    for _, row in class_groups.iterrows():

        group_id = row["_group_id"]
        group_size = int(row["group_size"])

        candidate_scores = {}

        for split in split_names:

            # Pretend this group goes into this split.
            hypothetical = current_rows.copy()

            hypothetical[split] += group_size

            # Calculate normalized squared error from the
            # desired row counts.
            error = 0.0

            for candidate in split_names:

                difference = (
                    hypothetical[candidate]
                    - target_rows[candidate]
                )

                normalized_difference = (
                    difference
                    / max(target_rows[candidate], 1)
                )

                error += normalized_difference ** 2

            candidate_scores[split] = error

        # Choose the split that gives the smallest total
        # deviation from the desired class distribution.
        best_split = min(
            candidate_scores,
            key=candidate_scores.get
        )

        assignments[group_id] = best_split

        current_rows[best_split] += group_size

    return assignments


# ============================================================
# CREATE COMPLETE SPLIT ASSIGNMENTS
# ============================================================

def create_group_assignments(groups, random_state=42):

    rng = np.random.default_rng(random_state)

    assignments = {}

    # Process every class separately.
    for label in sorted(groups[TARGET].unique()):

        class_groups = groups[
            groups[TARGET] == label
        ]

        class_assignments = assign_groups_for_class(
            class_groups=class_groups,
            train_target=TRAIN_TARGET,
            val_target=VAL_TARGET,
            test_target=TEST_TARGET,
            rng=rng,
        )

        assignments.update(class_assignments)

    return assignments


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 SIZE-AWARE GROUP-STRATIFIED SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------------

    print("\n[1/8] Loading cleaned dataset...")

    df = pd.read_parquet(INPUT_PATH)

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns):,}")

    # --------------------------------------------------------
    # 2. Identify feature columns
    # --------------------------------------------------------

    print("\n[2/8] Identifying feature columns...")

    feature_columns = [
        column
        for column in df.columns
        if column not in {
            TARGET,
            SOURCE_COLUMN,
        }
    ]

    print(
        f"  Feature columns: {len(feature_columns)}"
    )

    if len(feature_columns) != 73:
        raise RuntimeError(
            f"Expected 73 feature columns, "
            f"found {len(feature_columns)}."
        )

    # --------------------------------------------------------
    # 3. Create exact feature-vector groups
    # --------------------------------------------------------

    print(
        "\n[3/8] Creating exact feature-vector groups..."
    )

    df["_group_id"] = make_group_ids(
        df,
        feature_columns,
    )

    unique_groups = df["_group_id"].nunique()

    print(
        f"  Rows          : {len(df):,}"
    )

    print(
        f"  Unique groups : {unique_groups:,}"
    )

    print(
        f"  Duplicate rows beyond groups: "
        f"{len(df) - unique_groups:,}"
    )

    # --------------------------------------------------------
    # 4. Build group-level information
    # --------------------------------------------------------

    print(
        "\n[4/8] Building group-level information..."
    )

    group_label_counts = (
        df.groupby("_group_id")[TARGET]
        .nunique()
    )

    conflicting_groups = int(
        (group_label_counts > 1).sum()
    )

    print(
        f"  Conflicting groups: "
        f"{conflicting_groups}"
    )

    if conflicting_groups != 0:
        raise RuntimeError(
            "Conflicting feature-vector groups detected. "
            "The cleaned dataset should contain none."
        )

    groups = (
        df.groupby(
            "_group_id",
            sort=False,
        )
        .agg(
            **{
                TARGET: (TARGET, "first"),
                "group_size": (
                    TARGET,
                    "size",
                ),
            }
        )
        .reset_index()
    )

    print(
        f"  Groups available for splitting: "
        f"{len(groups):,}"
    )

    # --------------------------------------------------------
    # 5. Create size-aware assignments
    # --------------------------------------------------------

    print(
        "\n[5/8] Creating size-aware "
        "class-stratified assignments..."
    )

    assignments = create_group_assignments(
        groups,
        random_state=RANDOM_STATE,
    )

    groups["split"] = groups["_group_id"].map(
        assignments
    )

    if groups["split"].isna().any():
        raise RuntimeError(
            "Some groups were not assigned "
            "to a split."
        )

    # --------------------------------------------------------
    # Show group assignment counts
    # --------------------------------------------------------

    group_counts = (
        groups["split"]
        .value_counts()
        .reindex(
            ["train", "validation", "test"],
            fill_value=0,
        )
    )

    print(
        "\n  Groups assigned:"
    )

    print(
        f"    TRAIN      : "
        f"{group_counts['train']:,}"
    )

    print(
        f"    VALIDATION : "
        f"{group_counts['validation']:,}"
    )

    print(
        f"    TEST       : "
        f"{group_counts['test']:,}"
    )

    # --------------------------------------------------------
    # 6. Map groups back to rows
    # --------------------------------------------------------

    print(
        "\n[6/8] Assigning rows to splits..."
    )

    split_map = groups.set_index(
        "_group_id"
    )["split"]

    df["_split"] = df["_group_id"].map(
        split_map
    )

    train = df[
        df["_split"] == "train"
    ].copy()

    val = df[
        df["_split"] == "validation"
    ].copy()

    test = df[
        df["_split"] == "test"
    ].copy()

    # Save group IDs temporarily for the leakage check.
    train_group_ids = set(
        train["_group_id"]
    )

    val_group_ids = set(
        val["_group_id"]
    )

    test_group_ids = set(
        test["_group_id"]
    )

    # Remove internal columns.
    train.drop(
        columns=[
            "_group_id",
            "_split",
        ],
        inplace=True,
    )

    val.drop(
        columns=[
            "_group_id",
            "_split",
        ],
        inplace=True,
    )

    test.drop(
        columns=[
            "_group_id",
            "_split",
        ],
        inplace=True,
    )

    total_rows = (
        len(train)
        + len(val)
        + len(test)
    )

    train_pct = (
        len(train)
        / total_rows
        * 100
    )

    val_pct = (
        len(val)
        / total_rows
        * 100
    )

    test_pct = (
        len(test)
        / total_rows
        * 100
    )

    print(
        f"  Train rows      : {len(train):,}"
    )

    print(
        f"  Validation rows : {len(val):,}"
    )

    print(
        f"  Test rows       : {len(test):,}"
    )

    print(
        "\n  Actual overall proportions:"
    )

    print(
        f"    TRAIN      : {train_pct:.3f}%"
    )

    print(
        f"    VALIDATION : {val_pct:.3f}%"
    )

    print(
        f"    TEST       : {test_pct:.3f}%"
    )

    # --------------------------------------------------------
    # 7. Validate before saving
    # --------------------------------------------------------

    print(
        "\n[7/8] Running split checks..."
    )

    train_val_overlap = (
        train_group_ids
        & val_group_ids
    )

    train_test_overlap = (
        train_group_ids
        & test_group_ids
    )

    val_test_overlap = (
        val_group_ids
        & test_group_ids
    )

    print(
        f"  Train ∩ Validation groups: "
        f"{len(train_val_overlap)}"
    )

    print(
        f"  Train ∩ Test groups: "
        f"{len(train_test_overlap)}"
    )

    print(
        f"  Validation ∩ Test groups: "
        f"{len(val_test_overlap)}"
    )

    if (
        train_val_overlap
        or train_test_overlap
        or val_test_overlap
    ):
        raise RuntimeError(
            "GROUP LEAKAGE DETECTED!"
        )

    print(
        "  [PASS] No exact feature-vector "
        "group crosses splits."
    )

    # --------------------------------------------------------
    # Check all labels exist
    # --------------------------------------------------------

    all_labels = set(
        df[TARGET].unique()
    )

    for name, split_df in [
        ("TRAIN", train),
        ("VALIDATION", val),
        ("TEST", test),
    ]:

        split_labels = set(
            split_df[TARGET].unique()
        )

        missing = (
            all_labels
            - split_labels
        )

        if missing:
            raise RuntimeError(
                f"{name} is missing labels: "
                f"{sorted(missing)}"
            )

        print(
            f"  [PASS] {name} contains "
            f"all {len(all_labels)} labels."
        )

    # --------------------------------------------------------
    # Safety check:
    #
    # The total split should be close to 70/15/15.
    # We allow a small tolerance because groups cannot be
    # divided.
    # --------------------------------------------------------

    if abs(train_pct - 70.0) > 2.0:
        raise RuntimeError(
            f"TRAIN proportion is {train_pct:.3f}%. "
            f"This is too far from the 70% target. "
            f"Splits will NOT be saved."
        )

    if abs(val_pct - 15.0) > 2.0:
        raise RuntimeError(
            f"VALIDATION proportion is {val_pct:.3f}%. "
            f"This is too far from the 15% target. "
            f"Splits will NOT be saved."
        )

    if abs(test_pct - 15.0) > 2.0:
        raise RuntimeError(
            f"TEST proportion is {test_pct:.3f}%. "
            f"This is too far from the 15% target. "
            f"Splits will NOT be saved."
        )

    print(
        "  [PASS] Overall split proportions "
        "are within tolerance."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\n  Saving split datasets..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        TRAIN_PATH,
        index=False,
    )

    val.to_parquet(
        VAL_PATH,
        index=False,
    )

    test.to_parquet(
        TEST_PATH,
        index=False,
    )

    print(
        f"    TRAIN      : {TRAIN_PATH}"
    )

    print(
        f"    VALIDATION : {VAL_PATH}"
    )

    print(
        f"    TEST       : {TEST_PATH}"
    )

    # --------------------------------------------------------
    # 8. Generate report
    # --------------------------------------------------------

    print(
        "\n[8/8] Generating split report..."
    )

    train_counts = (
        train[TARGET]
        .value_counts()
        .sort_index()
    )

    val_counts = (
        val[TARGET]
        .value_counts()
        .sort_index()
    )

    test_counts = (
        test[TARGET]
        .value_counts()
        .sort_index()
    )

    report_lines = []

    report_lines.append(
        "# CICIDS2017 Size-Aware "
        "Group-Stratified Split\n\n"
    )

    report_lines.append(
        "## Split sizes\n\n"
    )

    report_lines.append(
        "| Split | Rows | Percentage |\n"
    )

    report_lines.append(
        "|---|---:|---:|\n"
    )

    report_lines.append(
        f"| Train | {len(train):,} | "
        f"{train_pct:.3f}% |\n"
    )

    report_lines.append(
        f"| Validation | {len(val):,} | "
        f"{val_pct:.3f}% |\n"
    )

    report_lines.append(
        f"| Test | {len(test):,} | "
        f"{test_pct:.3f}% |\n"
    )

    report_lines.append(
        f"| Total | {total_rows:,} | 100.000% |\n\n"
    )

    report_lines.append(
        "## Class distribution\n\n"
    )

    report_lines.append(
        "| Label | Train | Train % | "
        "Validation | Val % | Test | Test % |\n"
    )

    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )

    for label in sorted(all_labels):

        train_count = int(
            train_counts.get(label, 0)
        )

        val_count = int(
            val_counts.get(label, 0)
        )

        test_count = int(
            test_counts.get(label, 0)
        )

        report_lines.append(
            f"| {label} | "
            f"{train_count:,} | "
            f"{train_count / len(train) * 100:.3f}% | "
            f"{val_count:,} | "
            f"{val_count / len(val) * 100:.3f}% | "
            f"{test_count:,} | "
            f"{test_count / len(test) * 100:.3f}% |\n"
        )

    report_lines.append(
        "\n## Method\n\n"
        "- Exact identical feature vectors were grouped together.\n"
        "- Groups were never divided between splits.\n"
        "- Groups were assigned separately within each class.\n"
        "- The target row distribution is approximately "
        "70% train / 15% validation / 15% test.\n"
        "- `source_file` was excluded from grouping and model features.\n"
        "- NaN values were not imputed during splitting.\n"
        f"- Random state: {RANDOM_STATE}.\n"
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(report_lines)

    print(
        f"  Report saved: {REPORT_PATH}"
    )

    print("\n" + "=" * 70)
    print("SIZE-AWARE SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"Train      : {len(train):,} rows "
        f"({train_pct:.3f}%)"
    )

    print(
        f"Validation : {len(val):,} rows "
        f"({val_pct:.3f}%)"
    )

    print(
        f"Test       : {len(test):,} rows "
        f"({test_pct:.3f}%)"
    )

    print(
        "\n[PASS] Group leakage check passed."
    )

    print(
        "[PASS] Split proportions passed."
    )

    print(
        "[PASS] Splits saved successfully."
    )

    print(
        "\nNext step:"
        "\n  py src\\audit_splits.py"
    )


if __name__ == "__main__":
    main()