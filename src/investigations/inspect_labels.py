from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd


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
    """
    Read CSV in chunks without modifying the source data.
    """
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


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 80)
    print("LABEL QUALITY INSPECTION")
    print("=" * 80)

    total_rows = 0

    raw_labels = Counter()
    stripped_labels = Counter()

    whitespace_labels = Counter()
    empty_labels = Counter()

    labels_by_file = defaultdict(Counter)

    suspicious_label_values = Counter()

    expected_columns = None
    label_column = None

    for filename in RAW_FILES:

        path = RAW_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        print(f"\nProcessing: {filename}")

        file_rows = 0

        for chunk in read_csv_chunks(path):

            # Clean column names ONLY in memory.
            chunk.columns = [str(c).strip() for c in chunk.columns]

            if expected_columns is None:
                expected_columns = list(chunk.columns)

                label_matches = [
                    c for c in expected_columns
                    if c.lower() == "label"
                ]

                if len(label_matches) != 1:
                    raise ValueError(
                        f"Could not identify exactly one Label column. "
                        f"Found: {label_matches}"
                    )

                label_column = label_matches[0]

            labels = chunk[label_column]

            for value in labels:

                if pd.isna(value):
                    empty_labels["<MISSING>"] += 1
                    continue

                value = str(value)

                raw_labels[value] += 1
                labels_by_file[filename][value] += 1

                stripped = value.strip()
                stripped_labels[stripped] += 1

                if value != stripped:
                    whitespace_labels[value] += 1

                if stripped == "":
                    empty_labels["<EMPTY_AFTER_STRIP>"] += 1

                # Detect replacement character often associated
                # with corrupted Unicode.
                if "\ufffd" in value:
                    suspicious_label_values[value] += 1

                # Detect unusual control characters.
                if any(ord(ch) < 32 and ch not in "\t\r\n" for ch in value):
                    suspicious_label_values[value] += 1

            file_rows += len(chunk)

        total_rows += file_rows

        print(f"Rows: {file_rows}")

    # ========================================================
    # RESULTS
    # ========================================================

    print("\n" + "=" * 80)
    print("LABEL AUDIT RESULTS")
    print("=" * 80)

    print(f"\nTotal rows: {total_rows}")
    print(f"Unique raw label values: {len(raw_labels)}")
    print(f"Unique stripped label values: {len(stripped_labels)}")

    print("\n--- RAW LABEL COUNTS ---")

    for label, count in raw_labels.most_common():
        percentage = count / total_rows * 100
        print(f"{label!r:45} {count:10,d}  ({percentage:8.4f}%)")

    print("\n--- LABELS WITH LEADING/TRAILING WHITESPACE ---")

    if whitespace_labels:
        for label, count in whitespace_labels.items():
            print(f"{label!r:45} {count:10,d}")
    else:
        print("None found.")

    print("\n--- MISSING / EMPTY LABELS ---")

    if empty_labels:
        for label, count in empty_labels.items():
            print(f"{label:30} {count:,}")
    else:
        print("None found.")

    print("\n--- SUSPICIOUS LABEL VALUES ---")

    if suspicious_label_values:
        for label, count in suspicious_label_values.items():
            print(f"{label!r:45} {count:,}")
    else:
        print("None found.")

    # ========================================================
    # SAVE LABEL COUNTS
    # ========================================================

    label_df = pd.DataFrame(
        [
            {
                "label": label,
                "count": count,
                "percentage": count / total_rows * 100,
            }
            for label, count in raw_labels.most_common()
        ]
    )

    label_df.to_csv(
        REPORTS_DIR / "label_quality_counts.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # LABEL DISTRIBUTION BY FILE
    # ========================================================

    rows = []

    for filename in RAW_FILES:

        file_total = sum(labels_by_file[filename].values())

        for label, count in labels_by_file[filename].most_common():

            rows.append(
                {
                    "file": filename,
                    "label": label,
                    "count": count,
                    "percentage_of_file": count / file_total * 100,
                }
            )

    file_label_df = pd.DataFrame(rows)

    file_label_df.to_csv(
        REPORTS_DIR / "label_distribution_by_file.csv",
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # NORMALIZED LABEL DIFFERENCES
    # ========================================================

    normalized_groups = defaultdict(list)

    for raw_label in raw_labels:
        normalized_groups[raw_label.strip()].append(raw_label)

    print("\n--- LABELS THAT DIFFER ONLY BY WHITESPACE ---")

    whitespace_variants_found = False

    for normalized, variants in normalized_groups.items():

        if len(variants) > 1:
            whitespace_variants_found = True

            print(f"\nNormalized: {normalized!r}")

            for variant in variants:
                print(
                    f"    {variant!r}: "
                    f"{raw_labels[variant]:,} rows"
                )

    if not whitespace_variants_found:
        print("None found.")

    print("\n" + "=" * 80)
    print("LABEL INSPECTION COMPLETE")
    print("=" * 80)

    print("\nReports written:")
    print(f"  {REPORTS_DIR / 'label_quality_counts.csv'}")
    print(f"  {REPORTS_DIR / 'label_distribution_by_file.csv'}")

    print("\nIMPORTANT:")
    print("No raw data was modified.")


if __name__ == "__main__":
    main()