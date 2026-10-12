from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

CHUNK_SIZE = 100_000

TARGET_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Label",
]


# --------------------------------------------------
# Counters
# --------------------------------------------------

total_rows = 0

missing_bytes = 0
inf_bytes = 0
inf_packets = 0

duration_zero = 0

missing_bytes_duration_zero = 0
inf_bytes_duration_zero = 0
inf_packets_duration_zero = 0

missing_bytes_not_duration_zero = 0
inf_bytes_not_duration_zero = 0
inf_packets_not_duration_zero = 0

overlap_any_rate_problem = 0

label_counts = Counter()
file_counts = Counter()

examples = []


# --------------------------------------------------
# Process files
# --------------------------------------------------

for filename in FILES:

    path = RAW_DIR / filename

    print(f"Processing: {filename}")

    try:
        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="utf-8",
        )
    except UnicodeDecodeError:
        reader = pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
        )

    file_problem_count = 0

    for chunk in reader:

        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # Convert only columns needed for this audit
        numeric_cols = [
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Flow Bytes/s",
            "Flow Packets/s",
        ]

        numeric = chunk[numeric_cols].apply(
            pd.to_numeric,
            errors="coerce"
        )

        duration = numeric["Flow Duration"]

        bytes_rate = numeric["Flow Bytes/s"]
        packets_rate = numeric["Flow Packets/s"]

        missing_b = bytes_rate.isna()
        inf_b = bytes_rate == float("inf")
        inf_p = packets_rate == float("inf")

        zero_duration = duration == 0

        # Basic counts
        missing_count = int(missing_b.sum())
        inf_b_count = int(inf_b.sum())
        inf_p_count = int(inf_p.sum())

        missing_bytes += missing_count
        inf_bytes += inf_b_count
        inf_packets += inf_p_count

        duration_zero += int(zero_duration.sum())

        # Overlap with duration == 0
        missing_bytes_duration_zero += int(
            (missing_b & zero_duration).sum()
        )

        inf_bytes_duration_zero += int(
            (inf_b & zero_duration).sum()
        )

        inf_packets_duration_zero += int(
            (inf_p & zero_duration).sum()
        )

        missing_bytes_not_duration_zero += int(
            (missing_b & ~zero_duration).sum()
        )

        inf_bytes_not_duration_zero += int(
            (inf_b & ~zero_duration).sum()
        )

        inf_packets_not_duration_zero += int(
            (inf_p & ~zero_duration).sum()
        )

        problem_mask = missing_b | inf_b | inf_p

        problem_rows = chunk.loc[problem_mask].copy()

        if len(problem_rows) > 0:

            file_problem_count += len(problem_rows)

            for label, count in problem_rows["Label"].value_counts().items():
                label_counts[label] += int(count)

            # Save a few representative examples
            if len(examples) < 30:
                for idx in problem_rows.index[:30 - len(examples)]:

                    row = problem_rows.loc[idx]

                    examples.append({
                        "file": filename,
                        "row_index": int(idx),
                        "Flow Duration": row["Flow Duration"],
                        "Total Fwd Packets": row["Total Fwd Packets"],
                        "Total Backward Packets": row["Total Backward Packets"],
                        "Total Length of Fwd Packets": row[
                            "Total Length of Fwd Packets"
                        ],
                        "Total Length of Bwd Packets": row[
                            "Total Length of Bwd Packets"
                        ],
                        "Flow Bytes/s": row["Flow Bytes/s"],
                        "Flow Packets/s": row["Flow Packets/s"],
                        "Label": row["Label"],
                    })

    file_counts[filename] = file_problem_count


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("INFINITY / MISSING RATE AUDIT")
print("=" * 60)

print(f"Total rows: {total_rows}")
print()

print("PROBLEM COUNTS")
print(f"Missing Flow Bytes/s       : {missing_bytes}")
print(f"+Infinity Flow Bytes/s     : {inf_bytes}")
print(f"+Infinity Flow Packets/s   : {inf_packets}")
print(f"Flow Duration == 0         : {duration_zero}")
print()

print("OVERLAP WITH FLOW DURATION == 0")
print(f"Missing Flow Bytes/s       : {missing_bytes_duration_zero}")
print(f"+Infinity Flow Bytes/s     : {inf_bytes_duration_zero}")
print(f"+Infinity Flow Packets/s   : {inf_packets_duration_zero}")
print()

print("NOT CAUSED BY FLOW DURATION == 0")
print(f"Missing Flow Bytes/s       : {missing_bytes_not_duration_zero}")
print(f"+Infinity Flow Bytes/s     : {inf_bytes_not_duration_zero}")
print(f"+Infinity Flow Packets/s   : {inf_packets_not_duration_zero}")
print()

print("LABEL DISTRIBUTION")
for label, count in label_counts.most_common():
    print(f"{label:<40} {count}")

print()
print("PROBLEM ROWS BY FILE")
for filename, count in file_counts.items():
    print(f"{filename:<65} {count}")

# --------------------------------------------------
# Save representative examples
# --------------------------------------------------

examples_df = pd.DataFrame(examples)

examples_path = REPORT_DIR / "infinity_rate_examples.csv"
examples_df.to_csv(examples_path, index=False)

print()
print(f"Examples saved to: {examples_path}")