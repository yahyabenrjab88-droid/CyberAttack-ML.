from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"

REPORT_PATH = PROJECT_ROOT / "reports" / "split_distribution_audit.md"

TARGET = "Label"
SOURCE_COLUMN = "source_file"


# ============================================================
# HELPERS
# ============================================================

def load_split(path, name):
    print(f"\nLoading {name}...")
    df = pd.read_parquet(path)

    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")

    if TARGET not in df.columns:
        raise RuntimeError(f"{TARGET} column missing from {name}")

    if SOURCE_COLUMN not in df.columns:
        raise RuntimeError(f"{SOURCE_COLUMN} column missing from {name}")

    return df


def get_label_distribution(df):
    counts = df[TARGET].value_counts().sort_index()
    percentages = df[TARGET].value_counts(normalize=True).sort_index() * 100

    result = pd.DataFrame({
        "rows": counts,
        "percentage": percentages,
    })

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CICIDS2017 SPLIT DISTRIBUTION AUDIT")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load splits
    # --------------------------------------------------------

    print("\n[1/5] Loading split datasets...")

    train = load_split(TRAIN_PATH, "TRAIN")
    val = load_split(VAL_PATH, "VALIDATION")
    test = load_split(TEST_PATH, "TEST")

    # --------------------------------------------------------
    # 2. Basic row-count check
    # --------------------------------------------------------

    print("\n[2/5] Checking row counts...")

    total_rows = len(train) + len(val) + len(test)

    print(f"  TRAIN      : {len(train):,}")
    print(f"  VALIDATION : {len(val):,}")
    print(f"  TEST       : {len(test):,}")
    print(f"  TOTAL      : {total_rows:,}")

    # --------------------------------------------------------
    # 3. Class distributions
    # --------------------------------------------------------

    print("\n[3/5] Checking class distributions...")

    train_dist = get_label_distribution(train)
    val_dist = get_label_distribution(val)
    test_dist = get_label_distribution(test)

    all_labels = sorted(
        set(train[TARGET])
        | set(val[TARGET])
        | set(test[TARGET])
    )

    distribution = pd.DataFrame(index=all_labels)

    distribution["train_rows"] = train_dist["rows"].reindex(all_labels, fill_value=0)
    distribution["train_%"] = train_dist["percentage"].reindex(all_labels, fill_value=0)

    distribution["val_rows"] = val_dist["rows"].reindex(all_labels, fill_value=0)
    distribution["val_%"] = val_dist["percentage"].reindex(all_labels, fill_value=0)

    distribution["test_rows"] = test_dist["rows"].reindex(all_labels, fill_value=0)
    distribution["test_%"] = test_dist["percentage"].reindex(all_labels, fill_value=0)

    print("\n  Label distribution:")
    print()

    for label, row in distribution.iterrows():
        print(
            f"  {label:<35} "
            f"Train: {int(row['train_rows']):>8,} ({row['train_%']:>7.3f}%) | "
            f"Val: {int(row['val_rows']):>8,} ({row['val_%']:>7.3f}%) | "
            f"Test: {int(row['test_rows']):>8,} ({row['test_%']:>7.3f}%)"
        )

    # --------------------------------------------------------
    # 4. Source-file distribution
    # --------------------------------------------------------

    print("\n[4/5] Checking source-file distribution...")

    train_sources = train[SOURCE_COLUMN].value_counts(normalize=True).sort_index() * 100
    val_sources = val[SOURCE_COLUMN].value_counts(normalize=True).sort_index() * 100
    test_sources = test[SOURCE_COLUMN].value_counts(normalize=True).sort_index() * 100

    all_sources = sorted(
        set(train[SOURCE_COLUMN])
        | set(val[SOURCE_COLUMN])
        | set(test[SOURCE_COLUMN])
    )

    source_distribution = pd.DataFrame(index=all_sources)

    source_distribution["train_%"] = train_sources.reindex(
        all_sources, fill_value=0
    )

    source_distribution["val_%"] = val_sources.reindex(
        all_sources, fill_value=0
    )

    source_distribution["test_%"] = test_sources.reindex(
        all_sources, fill_value=0
    )

    print()

    for source, row in source_distribution.iterrows():
        print(
            f"  {source:<60} "
            f"Train: {row['train_%']:>7.3f}% | "
            f"Val: {row['val_%']:>7.3f}% | "
            f"Test: {row['test_%']:>7.3f}%"
        )

    # --------------------------------------------------------
    # 5. Generate report
    # --------------------------------------------------------

    print("\n[5/5] Generating audit report...")

    report_lines = []

    report_lines.append("# CICIDS2017 Split Distribution Audit\n")
    report_lines.append("## Dataset sizes\n")
    report_lines.append(
        f"- Train: {len(train):,} rows\n"
        f"- Validation: {len(val):,} rows\n"
        f"- Test: {len(test):,} rows\n"
        f"- Total: {total_rows:,} rows\n"
    )

    report_lines.append("\n## Class distribution\n")

    report_lines.append(
        "| Label | Train rows | Train % | Validation rows | Validation % | Test rows | Test % |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n"
    )

    for label, row in distribution.iterrows():
        report_lines.append(
            f"| {label} | "
            f"{int(row['train_rows']):,} | "
            f"{row['train_%']:.3f}% | "
            f"{int(row['val_rows']):,} | "
            f"{row['val_%']:.3f}% | "
            f"{int(row['test_rows']):,} | "
            f"{row['test_%']:.3f}% |\n"
        )

    report_lines.append("\n## Source-file distribution\n")

    report_lines.append(
        "| Source file | Train % | Validation % | Test % |\n"
        "|---|---:|---:|---:|\n"
    )

    for source, row in source_distribution.iterrows():
        report_lines.append(
            f"| {source} | "
            f"{row['train_%']:.3f}% | "
            f"{row['val_%']:.3f}% | "
            f"{row['test_%']:.3f}% |\n"
        )

    report_lines.append(
        "\n## Notes\n\n"
        "- The primary split is group-aware to prevent identical feature vectors "
        "from crossing train/validation/test.\n"
        "- Class stratification was performed at the group level.\n"
        "- Therefore, row-level class percentages are not expected to be "
        "perfectly identical across splits.\n"
        "- `source_file` is provenance information and is not a model feature.\n"
        "- This audit does not modify any dataset.\n"
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"  Report saved: {REPORT_PATH}")

    print("\n" + "=" * 70)
    print("SPLIT DISTRIBUTION AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()