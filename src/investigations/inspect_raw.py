"""
inspect_raw.py

Read-only audit of the 8 raw CICIDS2017 CSV files.

This script:
- Reads each CSV in chunks to control RAM usage.
- Does NOT modify the raw files.
- Checks column consistency.
- Audits labels.
- Audits missing values.
- Audits numeric conversion failures.
- Audits +/- infinity.
- Audits negative values.
- Finds constant features.
- Counts exact duplicate complete rows.
- Counts duplicate feature vectors.
- Detects feature vectors associated with conflicting labels.
- Writes audit reports to reports/.

It does NOT:
- clean the dataset
- drop rows
- encode labels
- scale features
- split the dataset
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RAW FILES
# ============================================================

RAW_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000

# Number of suspicious raw values to keep as examples per column.
MAX_BAD_VALUE_SAMPLES = 5


# ============================================================
# HELPERS
# ============================================================

def read_csv_chunks(path: Path):
    """
    Read a CSV file in chunks.

    UTF-8 is attempted first.
    latin1 is used as a fallback if decoding fails.
    """

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            chunksize=CHUNK_SIZE,
            low_memory=True,
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            chunksize=CHUNK_SIZE,
            encoding="latin1",
            low_memory=True,
        )


def clean_column_names(columns):
    """Strip whitespace from column names without modifying raw files."""
    return [str(col).strip() for col in columns]


def safe_float_series(series: pd.Series) -> pd.Series:
    """
    Convert a string series to numeric values.

    Invalid values become NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def add_sample(samples_dict, column, values):
    """
    Store a few examples of suspicious values per column.
    """
    if column not in samples_dict:
        samples_dict[column] = []

    remaining = MAX_BAD_VALUE_SAMPLES - len(samples_dict[column])

    if remaining <= 0:
        return

    for value in values.astype(str).tolist()[:remaining]:
        if value not in samples_dict[column]:
            samples_dict[column].append(value)


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 80)
    print("CICIDS2017 RAW DATA AUDIT")
    print("=" * 80)
    print()

    # --------------------------------------------------------
    # Verify raw directory
    # --------------------------------------------------------

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory does not exist:\n{RAW_DIR}"
        )

    missing_files = [
        filename
        for filename in RAW_FILES
        if not (RAW_DIR / filename).exists()
    ]

    if missing_files:
        print("ERROR: The following expected files are missing:")
        for filename in missing_files:
            print(f"  - {filename}")
        raise SystemExit(1)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Raw directory: {RAW_DIR}")
    print(f"Chunk size   : {CHUNK_SIZE:,}")
    print()


    # --------------------------------------------------------
    # Global containers
    # --------------------------------------------------------

    reference_columns = None
    feature_columns = None
    label_column = None

    total_rows = 0

    rows_per_file = {}
    labels_global = Counter()
    labels_per_file = {}

    missing_counts = Counter()
    numeric_failure_counts = Counter()
    positive_inf_counts = Counter()
    negative_inf_counts = Counter()
    negative_counts = Counter()

    min_values = {}
    max_values = {}

    constant_values = {}

    bad_value_samples = {}

    # Hash -> number of occurrences
    exact_row_hash_counts = Counter()

    # Feature-vector hash -> number of occurrences
    feature_hash_counts = Counter()

    # Feature-vector hash -> labels
    feature_hash_labels = defaultdict(set)

    # --------------------------------------------------------
    # Process each file
    # --------------------------------------------------------

    for file_index, filename in enumerate(RAW_FILES, start=1):

        path = RAW_DIR / filename

        print("-" * 80)
        print(f"[{file_index}/{len(RAW_FILES)}] Processing:")
        print(filename)
        print("-" * 80)

        file_rows = 0
        file_labels = Counter()

        chunk_reader = read_csv_chunks(path)

        for chunk_number, chunk in enumerate(chunk_reader, start=1):

            # ------------------------------------------------
            # Normalize column names IN MEMORY ONLY
            # ------------------------------------------------

            chunk.columns = clean_column_names(chunk.columns)

            # ------------------------------------------------
            # First chunk establishes the schema
            # ------------------------------------------------

            if reference_columns is None:
                reference_columns = list(chunk.columns)

                if "Label" not in reference_columns:
                    raise ValueError(
                        "Could not find a 'Label' column after stripping "
                        "column-name whitespace."
                    )

                label_column = "Label"

                feature_columns = [
                    col for col in reference_columns
                    if col != label_column
                ]

                print()
                print("Detected columns:")
                print(f"  Total columns : {len(reference_columns)}")
                print(f"  Features      : {len(feature_columns)}")
                print(f"  Target        : {label_column}")
                print()

                print("Column list:")
                for index, column in enumerate(reference_columns, start=1):
                    print(f"  {index:02d}. {column}")

                print()

            else:
                # ------------------------------------------------
                # Check schema consistency
                # ------------------------------------------------

                current_columns = list(chunk.columns)

                if current_columns != reference_columns:
                    raise ValueError(
                        f"Column mismatch detected in:\n{filename}\n"
                        f"Chunk: {chunk_number}"
                    )

            # ------------------------------------------------
            # Row counts
            # ------------------------------------------------

            current_rows = len(chunk)

            file_rows += current_rows
            total_rows += current_rows

            # ------------------------------------------------
            # Labels
            # ------------------------------------------------

            labels = chunk[label_column].fillna("<MISSING_LABEL>")

            label_counts = labels.value_counts(dropna=False)

            for label, count in label_counts.items():
                label = str(label)

                labels_global[label] += int(count)
                file_labels[label] += int(count)

            # ------------------------------------------------
            # Missing values
            # ------------------------------------------------

            chunk_missing = chunk.isna().sum()

            for column, count in chunk_missing.items():
                if count:
                    missing_counts[column] += int(count)

            # ------------------------------------------------
            # Numeric feature auditing
            # ------------------------------------------------

            for column in feature_columns:

                numeric = safe_float_series(chunk[column])

                original_non_missing = chunk[column].notna()

                conversion_failures = (
                    original_non_missing
                    & numeric.isna()
                )

                failure_count = int(conversion_failures.sum())

                if failure_count:
                    numeric_failure_counts[column] += failure_count

                    add_sample(
                        bad_value_samples,
                        column,
                        chunk.loc[conversion_failures, column]
                    )

                # --------------------------------------------
                # Infinity
                # --------------------------------------------

                positive_inf = np.isposinf(numeric)
                negative_inf = np.isneginf(numeric)

                positive_inf_count = int(positive_inf.sum())
                negative_inf_count = int(negative_inf.sum())

                if positive_inf_count:
                    positive_inf_counts[column] += positive_inf_count

                if negative_inf_count:
                    negative_inf_counts[column] += negative_inf_count

                # --------------------------------------------
                # Negative values
                # --------------------------------------------

                negative_mask = numeric < 0
                negative_count = int(negative_mask.sum())

                if negative_count:
                    negative_counts[column] += negative_count

                # --------------------------------------------
                # Min / max
                # --------------------------------------------

                finite = numeric[np.isfinite(numeric)]

                if not finite.empty:

                    chunk_min = float(finite.min())
                    chunk_max = float(finite.max())

                    if (
                        column not in min_values
                        or chunk_min < min_values[column]
                    ):
                        min_values[column] = chunk_min

                    if (
                        column not in max_values
                        or chunk_max > max_values[column]
                    ):
                        max_values[column] = chunk_max

                # --------------------------------------------
                # Constant-value detection
                # --------------------------------------------

                if column not in constant_values:
                    constant_values[column] = set()

                if len(constant_values[column]) <= 1:

                    unique_values = (
                        chunk[column]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    for value in unique_values:
                        constant_values[column].add(value)

                        if len(constant_values[column]) > 1:
                            break

            # ------------------------------------------------
            # Exact duplicate complete rows
            # ------------------------------------------------

            complete_hashes = pd.util.hash_pandas_object(
                chunk,
                index=False,
            )

            exact_row_hash_counts.update(
                complete_hashes.tolist()
            )

            # ------------------------------------------------
            # Duplicate feature vectors
            # ------------------------------------------------

            feature_chunk = chunk[feature_columns]

            feature_hashes = pd.util.hash_pandas_object(
                feature_chunk,
                index=False,
            )

            feature_hash_counts.update(
                feature_hashes.tolist()
            )

            # ------------------------------------------------
            # Feature-vector -> label relationship
            # ------------------------------------------------

            paired = pd.DataFrame(
                {
                    "feature_hash": feature_hashes,
                    "label": labels.astype(str),
                }
            )

            grouped = paired.groupby("feature_hash")["label"]

            for feature_hash, group_labels in grouped:
                feature_hash_labels[feature_hash].update(
                    group_labels.tolist()
                )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if chunk_number == 1 or chunk_number % 5 == 0:
                print(
                    f"  Chunk {chunk_number:,} | "
                    f"Rows processed in file: {file_rows:,}"
                )

        rows_per_file[filename] = file_rows
        labels_per_file[filename] = file_labels

        print(
            f"  Finished: {file_rows:,} rows"
        )
        print()


    # ========================================================
    # POST-PROCESS DUPLICATE INFORMATION
    # ========================================================

    exact_duplicate_rows = sum(
        count - 1
        for count in exact_row_hash_counts.values()
        if count > 1
    )

    duplicate_feature_rows = sum(
        count - 1
        for count in feature_hash_counts.values()
        if count > 1
    )

    duplicate_feature_groups = sum(
        1
        for count in feature_hash_counts.values()
        if count > 1
    )

    conflicting_feature_groups = sum(
        1
        for labels in feature_hash_labels.values()
        if len(labels) > 1
    )

    conflicting_feature_rows = sum(
        feature_hash_counts[feature_hash]
        for feature_hash, labels in feature_hash_labels.items()
        if len(labels) > 1
    )


    # ========================================================
    # CONSTANT FEATURES
    # ========================================================

    constant_features = [
        column
        for column, values in constant_values.items()
        if len(values) <= 1
    ]


    # ========================================================
    # REPORT 1: raw_file_summary.csv
    # ========================================================

    file_summary_rows = []

    for filename in RAW_FILES:

        file_labels = labels_per_file[filename]

        row = {
            "file": filename,
            "rows": rows_per_file[filename],
        }

        for label, count in sorted(file_labels.items()):
            row[f"label__{label}"] = count

        file_summary_rows.append(row)

    file_summary_df = pd.DataFrame(file_summary_rows)

    file_summary_path = REPORTS_DIR / "raw_file_summary.csv"

    file_summary_df.to_csv(
        file_summary_path,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # REPORT 2: raw_label_counts.csv
    # ========================================================

    label_rows = []

    for label, count in labels_global.most_common():

        label_rows.append(
            {
                "label": label,
                "count": count,
                "percentage": (
                    100.0 * count / total_rows
                    if total_rows
                    else 0.0
                ),
            }
        )

    label_df = pd.DataFrame(label_rows)

    label_path = REPORTS_DIR / "raw_label_counts.csv"

    label_df.to_csv(
        label_path,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # REPORT 3: raw_column_summary.csv
    # ========================================================

    column_rows = []

    for column in reference_columns:

        if column == label_column:
            continue

        column_rows.append(
            {
                "column": column,
                "missing_count": missing_counts.get(column, 0),
                "numeric_conversion_failures": numeric_failure_counts.get(
                    column, 0
                ),
                "positive_inf_count": positive_inf_counts.get(
                    column, 0
                ),
                "negative_inf_count": negative_inf_counts.get(
                    column, 0
                ),
                "negative_value_count": negative_counts.get(
                    column, 0
                ),
                "min_finite_value": min_values.get(
                    column, np.nan
                ),
                "max_finite_value": max_values.get(
                    column, np.nan
                ),
                "constant_feature": column in constant_features,
                "bad_value_examples": " | ".join(
                    bad_value_samples.get(column, [])
                ),
            }
        )

    column_df = pd.DataFrame(column_rows)

    column_path = REPORTS_DIR / "raw_column_summary.csv"

    column_df.to_csv(
        column_path,
        index=False,
        encoding="utf-8",
    )


    # ========================================================
    # REPORT 4: raw_data_audit.md
    # ========================================================

    report_path = REPORTS_DIR / "raw_data_audit.md"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as report:

        report.write("# CICIDS2017 Raw Data Audit\n\n")

        report.write("## Dataset overview\n\n")

        report.write(
            f"- Files audited: {len(RAW_FILES)}\n"
        )

        report.write(
            f"- Total rows: {total_rows:,}\n"
        )

        report.write(
            f"- Total columns: {len(reference_columns)}\n"
        )

        report.write(
            f"- Feature columns: {len(feature_columns)}\n"
        )

        report.write(
            f"- Target column: `{label_column}`\n"
        )

        report.write(
            f"- Chunk size: {CHUNK_SIZE:,}\n"
        )

        report.write("\n")

        # ----------------------------------------------------
        # Files
        # ----------------------------------------------------

        report.write("## Rows per file\n\n")

        report.write("| File | Rows |\n")
        report.write("|---|---:|\n")

        for filename in RAW_FILES:
            report.write(
                f"| `{filename}` | "
                f"{rows_per_file[filename]:,} |\n"
            )

        report.write("\n")

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        report.write("## Label distribution\n\n")

        report.write("| Label | Count | Percentage |\n")
        report.write("|---|---:|---:|\n")

        for label, count in labels_global.most_common():

            percentage = (
                100.0 * count / total_rows
                if total_rows
                else 0.0
            )

            report.write(
                f"| `{label}` | "
                f"{count:,} | "
                f"{percentage:.4f}% |\n"
            )

        report.write("\n")

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        report.write("## Missing values\n\n")

        missing_total = sum(missing_counts.values())

        report.write(
            f"Total missing cells: **{missing_total:,}**\n\n"
        )

        if missing_total:

            report.write("| Column | Missing |\n")
            report.write("|---|---:|\n")

            for column, count in sorted(
                missing_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            ):

                if count:
                    report.write(
                        f"| `{column}` | {count:,} |\n"
                    )

            report.write("\n")

        # ----------------------------------------------------
        # Numeric conversion failures
        # ----------------------------------------------------

        report.write("## Numeric conversion failures\n\n")

        conversion_total = sum(
            numeric_failure_counts.values()
        )

        report.write(
            f"Total conversion failures: "
            f"**{conversion_total:,}**\n\n"
        )

        if conversion_total:

            report.write("| Column | Failures | Examples |\n")
            report.write("|---|---:|---|\n")

            for column, count in sorted(
                numeric_failure_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            ):

                examples = " | ".join(
                    bad_value_samples.get(column, [])
                )

                report.write(
                    f"| `{column}` | "
                    f"{count:,} | "
                    f"{examples} |\n"
                )

            report.write("\n")

        # ----------------------------------------------------
        # Infinity
        # ----------------------------------------------------

        report.write("## Infinity values\n\n")

        total_positive_inf = sum(
            positive_inf_counts.values()
        )

        total_negative_inf = sum(
            negative_inf_counts.values()
        )

        report.write(
            f"- Positive infinity: "
            f"**{total_positive_inf:,}**\n"
        )

        report.write(
            f"- Negative infinity: "
            f"**{total_negative_inf:,}**\n\n"
        )

        # ----------------------------------------------------
        # Negative values
        # ----------------------------------------------------

        report.write("## Negative values\n\n")

        total_negative = sum(
            negative_counts.values()
        )

        report.write(
            f"Total negative numeric values: "
            f"**{total_negative:,}**\n\n"
        )

        if total_negative:

            report.write("| Column | Negative values |\n")
            report.write("|---|---:|\n")

            for column, count in sorted(
                negative_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            ):

                report.write(
                    f"| `{column}` | {count:,} |\n"
                )

            report.write("\n")

        # ----------------------------------------------------
        # Constant features
        # ----------------------------------------------------

        report.write("## Constant features\n\n")

        if constant_features:

            for column in constant_features:
                report.write(
                    f"- `{column}`\n"
                )

        else:
            report.write("None detected.\n")

        report.write("\n")

        # ----------------------------------------------------
        # Duplicates
        # ----------------------------------------------------

        report.write("## Duplicate analysis\n\n")

        report.write(
            f"- Exact duplicate complete rows: "
            f"**{exact_duplicate_rows:,}**\n"
        )

        report.write(
            f"- Duplicate feature-vector groups: "
            f"**{duplicate_feature_groups:,}**\n"
        )

        report.write(
            f"- Duplicate feature-vector rows beyond first occurrence: "
            f"**{duplicate_feature_rows:,}**\n"
        )

        report.write(
            f"- Conflicting-label feature-vector groups: "
            f"**{conflicting_feature_groups:,}**\n"
        )

        report.write(
            f"- Rows belonging to conflicting-label feature-vector groups: "
            f"**{conflicting_feature_rows:,}**\n"
        )

        report.write("\n")

        # ----------------------------------------------------
        # Columns
        # ----------------------------------------------------

        report.write("## Feature columns\n\n")

        report.write(
            f"Total feature columns: **{len(feature_columns)}**\n\n"
        )

        for column in feature_columns:
            report.write(
                f"- `{column}`\n"
            )

        report.write("\n")

        # ----------------------------------------------------
        # Methodology notes
        # ----------------------------------------------------

        report.write("## Methodology notes\n\n")

        report.write(
            "- Raw CSV files were read only; no raw file was modified.\n"
        )

        report.write(
            "- CSVs were processed in chunks to reduce memory usage.\n"
        )

        report.write(
            "- Column names were stripped in memory only.\n"
        )

        report.write(
            "- Numeric conversion was used for auditing only.\n"
        )

        report.write(
            "- No rows were dropped.\n"
        )

        report.write(
            "- No labels were changed.\n"
        )

        report.write(
            "- No scaling or normalization was performed.\n"
        )

        report.write(
            "- No train/validation/test split was performed.\n"
        )

        report.write(
            "- Duplicate detection uses pandas 64-bit row hashes.\n"
        )

        report.write(
            "- Conflicting-label feature vectors are reported for "
            "later cleaning decisions; they are not removed by this audit.\n"
        )


    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

    print(f"Total rows              : {total_rows:,}")
    print(f"Total columns           : {len(reference_columns)}")
    print(f"Feature columns         : {len(feature_columns)}")
    print(f"Labels                  : {len(labels_global)}")

    print()
    print("Duplicate analysis:")
    print(
        f"  Exact duplicate rows  : "
        f"{exact_duplicate_rows:,}"
    )
    print(
        f"  Duplicate feature rows: "
        f"{duplicate_feature_rows:,}"
    )
    print(
        f"  Conflicting groups    : "
        f"{conflicting_feature_groups:,}"
    )

    print()
    print("Missing values:")
    print(
        f"  Total missing cells   : "
        f"{sum(missing_counts.values()):,}"
    )

    print()
    print("Infinity:")
    print(
        f"  +inf                  : "
        f"{sum(positive_inf_counts.values()):,}"
    )
    print(
        f"  -inf                  : "
        f"{sum(negative_inf_counts.values()):,}"
    )

    print()
    print("Reports created:")

    print(
        f"  {report_path}"
    )

    print(
        f"  {file_summary_path}"
    )

    print(
        f"  {column_path}"
    )

    print(
        f"  {label_path}"
    )

    print()
    print("Raw files were NOT modified.")
    print("=" * 80)


if __name__ == "__main__":
    main()