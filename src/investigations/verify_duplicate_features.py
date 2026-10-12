
from pathlib import Path
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
# SUSPECTED DUPLICATE FEATURE PAIRS
# ============================================================

FEATURE_PAIRS = [
    ("Fwd Header Length", "Fwd Header Length.1"),
    ("Total Fwd Packets", "Subflow Fwd Packets"),
    ("Total Backward Packets", "Subflow Bwd Packets"),
    ("Total Length of Fwd Packets", "Subflow Fwd Bytes"),
    ("Total Length of Bwd Packets", "Subflow Bwd Bytes"),
    ("Fwd Packet Length Mean", "Avg Fwd Segment Size"),
    ("Bwd Packet Length Mean", "Avg Bwd Segment Size"),
    ("RST Flag Count", "ECE Flag Count"),
    ("Fwd URG Flags", "CWE Flag Count"),
    ("Fwd PSH Flags", "SYN Flag Count"),
]


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# HELPERS
# ============================================================

def read_csv_chunks(file_path):
    """
    Read a raw CICIDS CSV in chunks.

    We keep values as strings so that this audit checks
    equality of the original exported values rather than
    introducing numeric conversion effects.
    """

    try:
        return pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="utf-8",
        )

    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
        )


def normalize_columns(columns):
    """
    Normalize column names in memory only.
    The raw files are never modified.
    """
    return [str(col).strip() for col in columns]


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    print("=" * 80)
    print("FULL-DATASET DUPLICATE FEATURE VERIFICATION")
    print("=" * 80)

    print(f"Raw directory : {RAW_DIR}")
    print(f"Chunk size    : {CHUNK_SIZE}")
    print(f"Files         : {len(RAW_FILES)}")
    print()

    # --------------------------------------------------------
    # Initialize results
    # --------------------------------------------------------

    results = []

    # For each pair we track:
    # - total rows checked
    # - rows where both values are equal
    # - rows where values differ
    # - rows where both are missing
    # - rows where only one is missing
    #
    # We also track the first mismatch for debugging.

    pair_stats = {}

    for feature_a, feature_b in FEATURE_PAIRS:
        pair_stats[(feature_a, feature_b)] = {
            "rows_checked": 0,
            "equal_rows": 0,
            "different_rows": 0,
            "both_missing": 0,
            "only_a_missing": 0,
            "only_b_missing": 0,
            "first_mismatch_file": "",
            "first_mismatch_row": "",
            "first_mismatch_a": "",
            "first_mismatch_b": "",
        }

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    total_dataset_rows = 0

    for file_index, filename in enumerate(RAW_FILES, start=1):

        file_path = RAW_DIR / filename

        print(f"[{file_index}/{len(RAW_FILES)}] Processing: {filename}")

        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw file not found:\n{file_path}"
            )

        file_rows = 0

        chunks = read_csv_chunks(file_path)

        for chunk in chunks:

            # Normalize column names in memory.
            chunk.columns = normalize_columns(chunk.columns)

            current_chunk_rows = len(chunk)
            total_dataset_rows += current_chunk_rows
            file_rows += current_chunk_rows

            # ------------------------------------------------
            # Check every suspected pair
            # ------------------------------------------------

            for feature_a, feature_b in FEATURE_PAIRS:

                key = (feature_a, feature_b)
                stats = pair_stats[key]

                if feature_a not in chunk.columns:
                    raise KeyError(
                        f"Missing feature '{feature_a}' "
                        f"in file '{filename}'"
                    )

                if feature_b not in chunk.columns:
                    raise KeyError(
                        f"Missing feature '{feature_b}' "
                        f"in file '{filename}'"
                    )

                a = chunk[feature_a]
                b = chunk[feature_b]

                a_missing = a.isna()
                b_missing = b.isna()

                both_missing = a_missing & b_missing
                only_a_missing = a_missing & ~b_missing
                only_b_missing = ~a_missing & b_missing

                # Two values are considered equal if their original
                # string representations are identical.
                #
                # Missing/missing is handled separately.
                both_present = ~a_missing & ~b_missing

                equal = both_present & (a == b)
                different = both_present & (a != b)

                equal_count = int(equal.sum())
                different_count = int(different.sum())
                both_missing_count = int(both_missing.sum())
                only_a_missing_count = int(only_a_missing.sum())
                only_b_missing_count = int(only_b_missing.sum())

                stats["rows_checked"] += current_chunk_rows
                stats["equal_rows"] += equal_count
                stats["different_rows"] += different_count
                stats["both_missing"] += both_missing_count
                stats["only_a_missing"] += only_a_missing_count
                stats["only_b_missing"] += only_b_missing_count

                # ------------------------------------------------
                # Record first mismatch
                # ------------------------------------------------

                mismatch_mask = different | only_a_missing | only_b_missing

                if (
                    mismatch_mask.any()
                    and stats["first_mismatch_file"] == ""
                ):

                    mismatch_indices = chunk.index[mismatch_mask]

                    first_index = mismatch_indices[0]

                    value_a = chunk.loc[first_index, feature_a]
                    value_b = chunk.loc[first_index, feature_b]

                    stats["first_mismatch_file"] = filename
                    stats["first_mismatch_row"] = str(
                        int(first_index) + 2
                    )
                    stats["first_mismatch_a"] = str(value_a)
                    stats["first_mismatch_b"] = str(value_b)

        print(f"    Rows checked: {file_rows:,}")

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    for feature_a, feature_b in FEATURE_PAIRS:

        stats = pair_stats[(feature_a, feature_b)]

        rows_checked = stats["rows_checked"]
        different = stats["different_rows"]
        both_missing = stats["both_missing"]
        only_a_missing = stats["only_a_missing"]
        only_b_missing = stats["only_b_missing"]

        comparable_rows = (
            rows_checked
            - both_missing
            - only_a_missing
            - only_b_missing
        )

        exact_match = (
            comparable_rows > 0
            and different == 0
        )

        if exact_match:
            status = "IDENTICAL"
        else:
            status = "NOT IDENTICAL"

        print()
        print(f"{feature_a}")
        print(f"  <-> {feature_b}")
        print(f"  Status              : {status}")
        print(f"  Rows checked        : {rows_checked:,}")
        print(f"  Equal values        : {stats['equal_rows']:,}")
        print(f"  Different values    : {different:,}")
        print(f"  Both missing        : {both_missing:,}")
        print(f"  Only first missing  : {only_a_missing:,}")
        print(f"  Only second missing : {only_b_missing:,}")

        if comparable_rows > 0:
            equality_rate = (
                stats["equal_rows"] / comparable_rows
            ) * 100

            print(
                f"  Equality rate       : "
                f"{equality_rate:.6f}%"
            )
        else:
            equality_rate = 0.0
            print("  Equality rate       : N/A")

        if stats["first_mismatch_file"]:
            print(
                f"  First mismatch file : "
                f"{stats['first_mismatch_file']}"
            )
            print(
                f"  First mismatch row  : "
                f"{stats['first_mismatch_row']}"
            )
            print(
                f"  First value         : "
                f"{stats['first_mismatch_a']}"
            )
            print(
                f"  Second value        : "
                f"{stats['first_mismatch_b']}"
            )

        results.append({
            "feature_a": feature_a,
            "feature_b": feature_b,
            "status": status,
            "rows_checked": rows_checked,
            "equal_rows": stats["equal_rows"],
            "different_rows": different,
            "both_missing": both_missing,
            "only_a_missing": only_a_missing,
            "only_b_missing": only_b_missing,
            "comparable_rows": comparable_rows,
            "equality_rate_percent": equality_rate,
            "first_mismatch_file": stats["first_mismatch_file"],
            "first_mismatch_row": stats["first_mismatch_row"],
            "first_mismatch_value_a": stats["first_mismatch_a"],
            "first_mismatch_value_b": stats["first_mismatch_b"],
        })

    # ========================================================
    # SAVE REPORT
    # ========================================================

    output_path = REPORTS_DIR / "duplicate_feature_verification.csv"

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    identical_count = sum(
        1
        for result in results
        if result["status"] == "IDENTICAL"
    )

    not_identical_count = len(results) - identical_count

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total dataset rows checked : {total_dataset_rows:,}")
    print(f"Feature pairs checked      : {len(FEATURE_PAIRS)}")
    print(f"Identical pairs             : {identical_count}")
    print(f"Non-identical pairs         : {not_identical_count}")
    print()
    print(f"Report saved to:")
    print(output_path)
    print()
    print("RAW DATA WAS NOT MODIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
