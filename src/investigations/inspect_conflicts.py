from pathlib import Path
from collections import defaultdict
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
# REPRESENTATIVE CONFLICT GROUPS
# ============================================================

# We inspect representative groups instead of all 698 groups.
#
# The script will automatically select:
#   - largest conflict groups
#   - medium-sized conflict groups
#   - small conflict groups
#   - rare label combinations
#
# Maximum number of groups inspected:
MAX_GROUPS = 15


# ============================================================
# CSV READER
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CONFLICTING FEATURE-VECTOR INSPECTION")
    print("=" * 80)

    expected_columns = None
    label_column = None
    feature_columns = None

    # --------------------------------------------------------
    # First pass:
    # Find every feature-vector hash and its labels.
    # --------------------------------------------------------

    feature_groups = defaultdict(lambda: {
        "count": 0,
        "labels": defaultdict(int),
        "files": defaultdict(int),
    })

    total_rows = 0

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

            feature_data = (
                chunk[feature_columns]
                .fillna("<NA>")
                .astype(str)
            )

            hashes = pd.util.hash_pandas_object(
                feature_data,
                index=False,
            )

            labels = (
                chunk[label_column]
                .fillna("<MISSING>")
                .astype(str)
            )

            for h, label in zip(hashes, labels):

                h = int(h)

                feature_groups[h]["count"] += 1
                feature_groups[h]["labels"][label] += 1
                feature_groups[h]["files"][filename] += 1

            total_rows += len(chunk)

    # --------------------------------------------------------
    # Identify conflicting groups
    # --------------------------------------------------------

    conflicts = []

    for feature_hash, info in feature_groups.items():

        if len(info["labels"]) > 1:

            conflicts.append(
                {
                    "feature_hash": feature_hash,
                    "row_count": info["count"],
                    "labels": dict(info["labels"]),
                    "files": dict(info["files"]),
                }
            )

    conflicts.sort(
        key=lambda x: x["row_count"],
        reverse=True,
    )

    print("\n" + "=" * 80)
    print("CONFLICT SUMMARY")
    print("=" * 80)

    print(f"\nTotal dataset rows: {total_rows:,}")
    print(f"Conflicting groups found: {len(conflicts):,}")

    print("\nLargest conflicting groups:")

    for i, conflict in enumerate(conflicts[:10], start=1):

        label_text = " | ".join(
            f"{label} ({count:,})"
            for label, count
            in sorted(
                conflict["labels"].items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        print(
            f"{i:2}. "
            f"Rows={conflict['row_count']:,} | "
            f"{label_text}"
        )

    # ========================================================
    # SELECT REPRESENTATIVE GROUPS
    # ========================================================

    selected = []

    # Largest groups
    selected.extend(conflicts[:5])

    # Medium-sized groups
    if len(conflicts) > 5:
        middle_start = len(conflicts) // 2
        selected.extend(
            conflicts[middle_start:middle_start + 5]
        )

    # Small groups
    small_groups = sorted(
        conflicts,
        key=lambda x: x["row_count"]
    )

    selected.extend(small_groups[:5])

    # Remove duplicates while preserving order
    unique_selected = []
    seen = set()

    for conflict in selected:

        h = conflict["feature_hash"]

        if h not in seen:
            seen.add(h)
            unique_selected.append(conflict)

    selected = unique_selected[:MAX_GROUPS]

    selected_hashes = {
        conflict["feature_hash"]
        for conflict in selected
    }

    print(
        f"\nRepresentative groups selected: "
        f"{len(selected)}"
    )

    # ========================================================
    # SECOND PASS:
    # Collect actual rows belonging to selected groups.
    # ========================================================

    collected_rows = defaultdict(list)

    for filename in RAW_FILES:

        path = RAW_DIR / filename

        print(
            f"\nSearching selected conflicts in: "
            f"{filename}"
        )

        for chunk in read_csv_chunks(path):

            chunk.columns = [str(c).strip() for c in chunk.columns]

            feature_data = (
                chunk[feature_columns]
                .fillna("<NA>")
                .astype(str)
            )

            hashes = pd.util.hash_pandas_object(
                feature_data,
                index=False,
            )

            for idx, h in enumerate(hashes):

                h = int(h)

                if h in selected_hashes:

                    row = chunk.iloc[idx].copy()
                    row["Source_File"] = filename
                    row["Feature_Hash"] = h

                    collected_rows[h].append(row)

    # ========================================================
    # WRITE DETAILED CSV
    # ========================================================

    detailed_rows = []

    for conflict in selected:

        h = conflict["feature_hash"]

        for row in collected_rows[h]:

            output = row.to_dict()

            output["Conflict_Row_Count"] = conflict["row_count"]
            output["Conflict_Labels"] = " | ".join(
                sorted(conflict["labels"].keys())
            )

            detailed_rows.append(output)

    detailed_df = pd.DataFrame(detailed_rows)

    if not detailed_df.empty:

        # Put useful identification columns first.
        priority_columns = [
            "Feature_Hash",
            "Conflict_Row_Count",
            "Conflict_Labels",
            "Source_File",
            label_column,
        ]

        remaining = [
            c for c in detailed_df.columns
            if c not in priority_columns
        ]

        detailed_df = detailed_df[
            priority_columns + remaining
        ]

    output_path = REPORTS_DIR / "selected_conflict_rows.csv"

    detailed_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # WRITE GROUP SUMMARY
    # ========================================================

    summary_rows = []

    for number, conflict in enumerate(selected, start=1):

        summary_rows.append(
            {
                "inspection_id": number,
                "feature_hash": conflict["feature_hash"],
                "row_count": conflict["row_count"],
                "labels": " | ".join(
                    f"{label}: {count}"
                    for label, count
                    in sorted(
                        conflict["labels"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ),
                "files": " | ".join(
                    f"{filename}: {count}"
                    for filename, count
                    in sorted(
                        conflict["files"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        REPORTS_DIR / "selected_conflict_groups.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # PRINT SELECTED GROUPS
    # ========================================================

    print("\n" + "=" * 80)
    print("SELECTED CONFLICT GROUPS")
    print("=" * 80)

    for number, conflict in enumerate(selected, start=1):

        print(f"\n--- GROUP {number} ---")

        print(
            f"Feature hash : "
            f"{conflict['feature_hash']}"
        )

        print(
            f"Rows         : "
            f"{conflict['row_count']:,}"
        )

        print("Labels:")

        for label, count in sorted(
            conflict["labels"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(
                f"  {label!r}: {count:,}"
            )

        print("Files:")

        for filename, count in sorted(
            conflict["files"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            print(
                f"  {filename}: {count:,}"
            )

    # ========================================================
    # FINISH
    # ========================================================

    print("\n" + "=" * 80)
    print("CONFLICT INSPECTION COMPLETE")
    print("=" * 80)

    print("\nReports written:")

    print(
        f"  {output_path}"
    )

    print(
        f"  {summary_path}"
    )

    print("\nIMPORTANT:")
    print("No raw data was modified.")
    print(
        "Only representative conflicting groups "
        "were extracted for investigation."
    )


if __name__ == "__main__":
    main()