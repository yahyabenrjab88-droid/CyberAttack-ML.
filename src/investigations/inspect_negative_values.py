from pathlib import Path
from collections import Counter, defaultdict
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

FEATURES = [
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
]

# Exact negative-value counts
value_counters = {
    feature: Counter()
    for feature in FEATURES
}

# Negative values by label
label_counters = {
    feature: defaultdict(Counter)
    for feature in FEATURES
}

# Negative values by file
file_counters = {
    feature: defaultdict(Counter)
    for feature in FEATURES
}

total_negative_rows = {
    feature: 0
    for feature in FEATURES
}


for filename in FILES:
    path = RAW_DIR / filename
    print(f"Processing: {filename}")

    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk.columns = chunk.columns.str.strip()

        for feature in FEATURES:
            values = pd.to_numeric(chunk[feature], errors="coerce")
            mask = values < 0

            if not mask.any():
                continue

            negative_values = values[mask]

            # Exact negative-value frequencies
            for value, count in negative_values.value_counts().items():
                value_counters[feature][float(value)] += int(count)

            total_negative_rows[feature] += int(mask.sum())

            # Negative values by label
            negative_chunk = chunk.loc[mask, ["Label", feature]].copy()
            negative_chunk[feature] = pd.to_numeric(
                negative_chunk[feature],
                errors="coerce"
            )

            for (label, value), count in (
                negative_chunk.groupby(["Label", feature], dropna=False)
                .size()
                .items()
            ):
                label_counters[feature][label][float(value)] += int(count)

            # Negative values by source file
            for value, count in negative_values.value_counts().items():
                file_counters[feature][filename][float(value)] += int(count)


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("NEGATIVE VALUE BREAKDOWN")
print("=" * 80)

for feature in FEATURES:
    print(f"\n{feature}")
    print("-" * len(feature))
    print(f"Total negative rows: {total_negative_rows[feature]:,}")

    if not value_counters[feature]:
        print("No negative values.")
        continue

    for value, count in sorted(value_counters[feature].items()):
        print(f"  {value:g} -> {count:,} rows")


# ============================================================
# SAVE EXACT VALUE COUNTS
# ============================================================

rows = []

for feature in FEATURES:
    for value, count in sorted(value_counters[feature].items()):
        rows.append({
            "feature": feature,
            "negative_value": value,
            "row_count": count,
        })

pd.DataFrame(rows).to_csv(
    REPORT_DIR / "negative_value_counts.csv",
    index=False,
)


# ============================================================
# SAVE VALUE + LABEL COUNTS
# ============================================================

label_rows = []

for feature in FEATURES:
    for label, counter in label_counters[feature].items():
        for value, count in sorted(counter.items()):
            label_rows.append({
                "feature": feature,
                "label": label,
                "negative_value": value,
                "row_count": count,
            })

pd.DataFrame(label_rows).to_csv(
    REPORT_DIR / "negative_value_counts_by_label.csv",
    index=False,
)


# ============================================================
# SAVE VALUE + FILE COUNTS
# ============================================================

file_rows = []

for feature in FEATURES:
    for filename, counter in file_counters[feature].items():
        for value, count in sorted(counter.items()):
            file_rows.append({
                "feature": feature,
                "file": filename,
                "negative_value": value,
                "row_count": count,
            })

pd.DataFrame(file_rows).to_csv(
    REPORT_DIR / "negative_value_counts_by_file.csv",
    index=False,
)


print("\nReports created:")
print(f"  {REPORT_DIR / 'negative_value_counts.csv'}")
print(f"  {REPORT_DIR / 'negative_value_counts_by_label.csv'}")
print(f"  {REPORT_DIR / 'negative_value_counts_by_file.csv'}")