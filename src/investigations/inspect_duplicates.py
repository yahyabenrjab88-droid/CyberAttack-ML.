from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import hashlib


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


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

CHUNK_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def read_csv_chunks(path):
    try:
        return pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
        )


def make_hashes(chunk, columns):
    """
    Create hashes for rows using their string representation.

    We deliberately do NOT modify values.
    """

    data = chunk[columns].fillna("<NA>").astype(str)

    return pd.util.hash_pandas_object(
        data,
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("DUPLICATE DATA INSPECTION")
    print("=" * 80)

    expected_columns = None
    label_column = None
    feature_columns = None

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_rows = 0

    exact_hash_counts = Counter()
    feature_hash_counts = Counter()

    # How many files contain each feature vector?
    feature_hash_files = defaultdict(set)

    # Labels associated with each feature vector.
    feature_hash_labels = defaultdict(Counter)

    # Exact duplicate rows by file.
    file_row_counts = Counter()

    # --------------------------------------------------------
    # PASS 1
    # --------------------------------------------------------

    for filename in RAW_FILES:

        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        print(f"\nProcessing: {filename}")

        for chunk in read_csv_chunks(path):

            chunk.columns = [str(c).strip() for c in chunk.columns]

            if expected_columns is None:

                expected_columns = list(chunk.columns)

                label_matches = [
                    c for c in expected_columns
                    if c.lower() == "label"
                ]

                if len(label_matches) != 1:
                    raise ValueError(
                        f"Could not identify Label column: {label_matches}"
                    )

                label_column = label_matches[0]

                feature_columns = [
                    c for c in expected_columns
                    if c != label_column
                ]

                print(f"Columns: {len(expected_columns)}")
                print(f"Features: {len(feature_columns)}")
                print(f"Label: {label_column}")

            # ------------------------------------------------
            # Exact full-row hash
            # Includes Label.
            # ------------------------------------------------

            exact_hashes = make_hashes(
                chunk,
                expected_columns,
            )

            for h in exact_hashes:
                exact_hash_counts[int(h)] += 1

            # ------------------------------------------------
            # Feature-only hash
            # Excludes Label.
            # ------------------------------------------------

            feature_hashes = make_hashes(
                chunk,
                feature_columns,
            )

            labels = chunk[label_column].fillna("<MISSING>").astype(str)

            for h, label in zip(feature_hashes, labels):

                h = int(h)

                feature_hash_counts[h] += 1
                feature_hash_files[h].add(filename)
                feature_hash_labels[h][label] += 1

            file_row_counts[filename] += len(chunk)
            total_rows += len(chunk)

    # ========================================================
    # BASIC COUNTS
    # ========================================================

    exact_duplicate_rows = sum(
        count - 1
        for count in exact_hash_counts.values()
        if count > 1
    )

    duplicate_feature_rows = sum(
        count - 1
        for count in feature_hash_counts.values()
        if count > 1
    )

    exact_duplicate_groups = sum(
        1
        for count in exact_hash_counts.values()
        if count > 1
    )

    feature_duplicate_groups = sum(
        1
        for count in feature_hash_counts.values()
        if count > 1
    )

    # ========================================================
    # CROSS-FILE FEATURE DUPLICATES
    # ========================================================

    cross_file_groups = {
        h: files
        for h, files in feature_hash_files.items()
        if len(files) > 1
    }

    cross_file_duplicate_rows = sum(
        feature_hash_counts[h] - 1
        for h in cross_file_groups
    )

    # ========================================================
    # CONFLICTING LABEL GROUPS
    # ========================================================

    conflicting_groups = {
        h: labels
        for h, labels in feature_hash_labels.items()
        if len(labels) > 1
    }

    conflicting_rows = sum(
        feature_hash_counts[h]
        for h in conflicting_groups
    )

    # ========================================================
    # LABEL DISTRIBUTION INSIDE DUPLICATE GROUPS
    # ========================================================

    duplicate_label_counts = Counter()

    for h, count in feature_hash_counts.items():

        if count > 1:

            for label, label_count in feature_hash_labels[h].items():
                duplicate_label_counts[label] += label_count

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("DUPLICATE AUDIT RESULTS")
    print("=" * 80)

    print(f"\nTotal rows: {total_rows:,}")

    print("\n--- EXACT FULL-ROW DUPLICATES ---")

    print(f"Duplicate groups: {exact_duplicate_groups:,}")
    print(f"Duplicate rows beyond first: {exact_duplicate_rows:,}")

    print("\n--- FEATURE-VECTOR DUPLICATES ---")

    print(f"Duplicate feature groups: {feature_duplicate_groups:,}")
    print(f"Duplicate feature rows beyond first: {duplicate_feature_rows:,}")

    print("\n--- CROSS-FILE FEATURE DUPLICATES ---")

    print(f"Groups appearing in multiple files: {len(cross_file_groups):,}")
    print(f"Rows beyond first occurrence: {cross_file_duplicate_rows:,}")

    print("\n--- CONFLICTING FEATURE-VECTOR LABELS ---")

    print(f"Conflicting groups: {len(conflicting_groups):,}")
    print(f"Rows belonging to conflicting groups: {conflicting_rows:,}")

    print("\n--- LABEL DISTRIBUTION INSIDE DUPLICATE FEATURE GROUPS ---")

    for label, count in duplicate_label_counts.most_common():
        print(f"{label!r:45} {count:10,d}")

    # ========================================================
    # CROSS-FILE REPORT
    # ========================================================

    cross_file_rows = []

    for h, files in cross_file_groups.items():

        labels = feature_hash_labels[h]

        cross_file_rows.append(
            {
                "feature_hash": h,
                "row_count": feature_hash_counts[h],
                "file_count": len(files),
                "files": " | ".join(sorted(files)),
                "labels": " | ".join(
                    sorted(labels.keys())
                ),
            }
        )

    cross_file_df = pd.DataFrame(cross_file_rows)

    if not cross_file_df.empty:
        cross_file_df = cross_file_df.sort_values(
            ["row_count", "file_count"],
            ascending=[False, False],
        )

    cross_file_df.to_csv(
        REPORTS_DIR / "cross_file_duplicate_features.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # DUPLICATE LABEL REPORT
    # ========================================================

    duplicate_label_df = pd.DataFrame(
        [
            {
                "label": label,
                "rows_in_duplicate_feature_groups": count,
            }
            for label, count in duplicate_label_counts.most_common()
        ]
    )

    duplicate_label_df.to_csv(
        REPORTS_DIR / "duplicate_label_distribution.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "exact_duplicate_groups": exact_duplicate_groups,
                "exact_duplicate_rows_beyond_first": exact_duplicate_rows,
                "feature_duplicate_groups": feature_duplicate_groups,
                "feature_duplicate_rows_beyond_first": duplicate_feature_rows,
                "cross_file_feature_groups": len(cross_file_groups),
                "cross_file_duplicate_rows_beyond_first": cross_file_duplicate_rows,
                "conflicting_feature_groups": len(conflicting_groups),
                "conflicting_rows": conflicting_rows,
            }
        ]
    )

    summary.to_csv(
        REPORTS_DIR / "duplicate_audit_summary.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n" + "=" * 80)
    print("DUPLICATE INSPECTION COMPLETE")
    print("=" * 80)

    print("\nReports written:")

    print(
        f"  {REPORTS_DIR / 'duplicate_audit_summary.csv'}"
    )

    print(
        f"  {REPORTS_DIR / 'cross_file_duplicate_features.csv'}"
    )

    print(
        f"  {REPORTS_DIR / 'duplicate_label_distribution.csv'}"
    )

    print("\nIMPORTANT:")
    print("No raw data was modified.")


if __name__ == "__main__":
    main()