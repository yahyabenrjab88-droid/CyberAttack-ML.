from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


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
# Counters
# ============================================================

total_rows = 0

missing_flow_bytes = 0
inf_flow_bytes = 0
inf_flow_packets = 0

negative_duration = 0
negative_iat = Counter()
negative_header = Counter()

negative_init_window = Counter()

label_counts = Counter()

# Conflicting feature-vector information
feature_labels = defaultdict(set)
feature_counts = Counter()
feature_examples = {}

# Duplicate complete rows
complete_row_counts = Counter()

# Keep examples for unusual values
examples = {
    "missing_flow_bytes": [],
    "inf_rates": [],
    "negative_duration": [],
    "negative_iat": [],
    "negative_headers": [],
    "negative_init_windows": [],
}


# ============================================================
# Columns needed
# ============================================================

CHECK_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
]


# ============================================================
# Read dataset
# ============================================================

for filename in RAW_FILES:

    path = RAW_DIR / filename

    print(f"Processing: {filename}")

    for chunk in pd.read_csv(
        path,
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):

        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        label_counts.update(
            chunk["Label"].fillna("<MISSING_LABEL>")
        )

        # ----------------------------------------------------
        # Numeric conversion
        # ----------------------------------------------------

        numeric = chunk[CHECK_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # ----------------------------------------------------
        # Missing / infinity
        # ----------------------------------------------------

        missing_mask = chunk["Flow Bytes/s"].isna()

        inf_bytes_mask = np.isposinf(
            numeric["Flow Bytes/s"]
        )

        inf_packets_mask = np.isposinf(
            numeric["Flow Packets/s"]
        )

        missing_flow_bytes += int(missing_mask.sum())
        inf_flow_bytes += int(inf_bytes_mask.sum())
        inf_flow_packets += int(inf_packets_mask.sum())

        # ----------------------------------------------------
        # Negative duration
        # ----------------------------------------------------

        mask = numeric["Flow Duration"] < 0

        negative_duration += int(mask.sum())

        if len(examples["negative_duration"]) < 10:
            rows = chunk.loc[mask]

            for _, row in rows.head(
                10 - len(examples["negative_duration"])
            ).iterrows():

                examples["negative_duration"].append(
                    {
                        "file": filename,
                        "Label": row["Label"],
                        "Flow Duration": row["Flow Duration"],
                    }
                )

        # ----------------------------------------------------
        # Negative IAT values
        # ----------------------------------------------------

        for column in [
            "Flow IAT Mean",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Min",
        ]:

            mask = numeric[column] < 0

            count = int(mask.sum())

            negative_iat[column] += count

            if len(examples["negative_iat"]) < 20:

                rows = chunk.loc[mask]

                for _, row in rows.head(
                    20 - len(examples["negative_iat"])
                ).iterrows():

                    examples["negative_iat"].append(
                        {
                            "file": filename,
                            "Label": row["Label"],
                            "column": column,
                            "value": row[column],
                        }
                    )

        # ----------------------------------------------------
        # Negative header values
        # ----------------------------------------------------

        for column in [
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Header Length.1",
            "min_seg_size_forward",
        ]:

            mask = numeric[column] < 0

            count = int(mask.sum())

            negative_header[column] += count

            if len(examples["negative_headers"]) < 20:

                rows = chunk.loc[mask]

                for _, row in rows.head(
                    20 - len(examples["negative_headers"])
                ).iterrows():

                    examples["negative_headers"].append(
                        {
                            "file": filename,
                            "Label": row["Label"],
                            "column": column,
                            "value": row[column],
                        }
                    )

        # ----------------------------------------------------
        # Negative initial window values
        # ----------------------------------------------------

        for column in [
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
        ]:

            mask = numeric[column] < 0

            count = int(mask.sum())

            negative_init_window[column] += count

            if len(examples["negative_init_windows"]) < 20:

                rows = chunk.loc[mask]

                for _, row in rows.head(
                    20 - len(examples["negative_init_windows"])
                ).iterrows():

                    examples["negative_init_windows"].append(
                        {
                            "file": filename,
                            "Label": row["Label"],
                            "column": column,
                            "value": row[column],
                        }
                    )

        # ----------------------------------------------------
        # Feature-vector groups
        # ----------------------------------------------------

        feature_columns = [
            column
            for column in chunk.columns
            if column != "Label"
        ]

        # Convert feature values to strings so the exact raw
        # representation is used for grouping.
        feature_frame = chunk[feature_columns].fillna(
            "<MISSING>"
        )

        for index, row in feature_frame.iterrows():

            values = tuple(row.tolist())

            label = chunk.at[index, "Label"]

            feature_counts[values] += 1
            feature_labels[values].add(label)

            if values not in feature_examples:
                feature_examples[values] = {
                    "file": filename,
                    "label": label,
                }

        # ----------------------------------------------------
        # Complete-row duplicates
        # ----------------------------------------------------

        for index, row in chunk.iterrows():

            values = tuple(row.tolist())

            complete_row_counts[values] += 1


# ============================================================
# Build report
# ============================================================

report = []

report.append("# CICIDS2017 Problem Investigation")
report.append("")
report.append("This report is READ-ONLY. Raw CSV files were not modified.")
report.append("")

report.append("## 1. Dataset size")
report.append("")
report.append(f"- Total rows: {total_rows:,}")
report.append("")

report.append("## 2. Missing / Infinity")
report.append("")
report.append(
    f"- Missing Flow Bytes/s: {missing_flow_bytes:,}"
)
report.append(
    f"- +Infinity Flow Bytes/s: {inf_flow_bytes:,}"
)
report.append(
    f"- +Infinity Flow Packets/s: {inf_flow_packets:,}"
)
report.append("")

report.append("## 3. Negative duration")
report.append("")
report.append(
    f"- Negative Flow Duration rows: {negative_duration:,}"
)
report.append("")

report.append("## 4. Negative IAT values")
report.append("")

for column, count in negative_iat.items():
    report.append(f"- {column}: {count:,}")

report.append("")

report.append("## 5. Negative header values")
report.append("")

for column, count in negative_header.items():
    report.append(f"- {column}: {count:,}")

report.append("")

report.append("## 6. Negative initial-window values")
report.append("")

for column, count in negative_init_window.items():
    report.append(f"- {column}: {count:,}")

report.append("")

report.append("## 7. Conflicting feature-vector groups")
report.append("")

conflicting_groups = [
    (features, labels, feature_counts[features])
    for features, labels in feature_labels.items()
    if len(labels) > 1
]

conflicting_rows = sum(
    count
    for _, _, count in conflicting_groups
)

report.append(
    f"- Conflicting groups: {len(conflicting_groups):,}"
)
report.append(
    f"- Rows belonging to conflicting groups: {conflicting_rows:,}"
)
report.append("")

conflict_pairs = Counter()

for _, labels, _ in conflicting_groups:

    labels = sorted(labels)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):

            conflict_pairs[
                f"{labels[i]} <-> {labels[j]}"
            ] += 1

report.append("### Label conflict pairs")
report.append("")

for pair, count in conflict_pairs.most_common():

    report.append(
        f"- {pair}: {count:,} groups"
    )

report.append("")

report.append("## 8. Duplicate feature vectors")
report.append("")

duplicate_feature_groups = sum(
    1
    for count in feature_counts.values()
    if count > 1
)

duplicate_feature_rows_beyond_first = sum(
    count - 1
    for count in feature_counts.values()
    if count > 1
)

report.append(
    f"- Duplicate feature-vector groups: "
    f"{duplicate_feature_groups:,}"
)

report.append(
    f"- Duplicate rows beyond first occurrence: "
    f"{duplicate_feature_rows_beyond_first:,}"
)

report.append("")

report.append("## 9. Exact complete-row duplicates")
report.append("")

exact_duplicate_groups = sum(
    1
    for count in complete_row_counts.values()
    if count > 1
)

exact_duplicate_rows_beyond_first = sum(
    count - 1
    for count in complete_row_counts.values()
    if count > 1
)

report.append(
    f"- Exact duplicate groups: "
    f"{exact_duplicate_groups:,}"
)

report.append(
    f"- Exact duplicate rows beyond first occurrence: "
    f"{exact_duplicate_rows_beyond_first:,}"
)

report.append("")

report.append("## 10. Examples")
report.append("")

report.append("### Negative duration examples")

for example in examples["negative_duration"]:
    report.append(str(example))

report.append("")

report.append("### Negative IAT examples")

for example in examples["negative_iat"]:
    report.append(str(example))

report.append("")

report.append("### Negative header examples")

for example in examples["negative_headers"]:
    report.append(str(example))

report.append("")

report.append("### Negative initial-window examples")

for example in examples["negative_init_windows"]:
    report.append(str(example))

report.append("")


output = REPORTS_DIR / "problem_investigation.md"

output.write_text(
    "\n".join(report),
    encoding="utf-8",
)


# ============================================================
# Save conflict summary
# ============================================================

conflict_summary = []

for features, labels, count in conflicting_groups:

    conflict_summary.append(
        {
            "row_count": count,
            "labels": " | ".join(sorted(labels)),
            "example_file": feature_examples[features]["file"],
        }
    )

pd.DataFrame(conflict_summary).sort_values(
    "row_count",
    ascending=False,
).to_csv(
    REPORTS_DIR / "conflicting_feature_vectors.csv",
    index=False,
    encoding="utf-8",
)


print()
print("=" * 80)
print("PROBLEM INVESTIGATION COMPLETE")
print("=" * 80)
print(f"Total rows inspected: {total_rows:,}")
print()
print(f"Report: {output}")
print(
    "Conflicts: "
    f"{REPORTS_DIR / 'conflicting_feature_vectors.csv'}"
)
print()