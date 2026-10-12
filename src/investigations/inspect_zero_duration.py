from pathlib import Path
import pandas as pd
from collections import Counter


# --------------------------------------------------
# Paths
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


# --------------------------------------------------
# Columns needed for the audit
# --------------------------------------------------

NUMERIC_COLUMNS = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "min_seg_size_forward",
    "Flow Bytes/s",
    "Flow Packets/s",
]


# --------------------------------------------------
# Counters
# --------------------------------------------------

total_zero_duration = 0

labels = Counter()
files = Counter()

zero_with_fwd_packets = 0
zero_with_bwd_packets = 0
zero_with_bytes = 0

zero_with_negative_iat = 0
zero_with_negative_headers = 0
zero_with_negative_min_seg = 0

zero_with_negative_duration = 0

examples = []


# --------------------------------------------------
# Process raw files
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

    file_zero_count = 0

    for chunk in reader:

        chunk.columns = chunk.columns.str.strip()

        numeric = chunk[NUMERIC_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        zero = numeric["Flow Duration"] == 0

        if not zero.any():
            continue

        z = numeric.loc[zero]
        zero_rows = chunk.loc[zero]

        count = len(z)

        total_zero_duration += count
        file_zero_count += count

        # ------------------------------------------
        # Labels
        # ------------------------------------------

        for label, n in zero_rows["Label"].value_counts().items():
            labels[label] += int(n)

        # ------------------------------------------
        # Basic flow content
        # ------------------------------------------

        zero_with_fwd_packets += int(
            (z["Total Fwd Packets"] > 0).sum()
        )

        zero_with_bwd_packets += int(
            (z["Total Backward Packets"] > 0).sum()
        )

        zero_with_bytes += int(
            (
                (z["Total Length of Fwd Packets"] > 0)
                | (z["Total Length of Bwd Packets"] > 0)
            ).sum()
        )

        # ------------------------------------------
        # Negative IAT values
        # ------------------------------------------

        negative_iat = (
            (z["Flow IAT Mean"] < 0)
            | (z["Flow IAT Max"] < 0)
            | (z["Flow IAT Min"] < 0)
            | (z["Fwd IAT Total"] < 0)
            | (z["Fwd IAT Mean"] < 0)
            | (z["Fwd IAT Max"] < 0)
            | (z["Fwd IAT Min"] < 0)
            | (z["Bwd IAT Total"] < 0)
            | (z["Bwd IAT Mean"] < 0)
            | (z["Bwd IAT Max"] < 0)
            | (z["Bwd IAT Min"] < 0)
        )

        zero_with_negative_iat += int(negative_iat.sum())

        # ------------------------------------------
        # Negative headers / segment size
        # ------------------------------------------

        negative_headers = (
            (z["Fwd Header Length"] < 0)
            | (z["Bwd Header Length"] < 0)
        )

        negative_min_seg = z["min_seg_size_forward"] < 0

        zero_with_negative_headers += int(
            negative_headers.sum()
        )

        zero_with_negative_min_seg += int(
            negative_min_seg.sum()
        )

        # ------------------------------------------
        # Examples
        # ------------------------------------------

        if len(examples) < 40:

            selected = zero_rows.iloc[
                :40 - len(examples)
            ]

            for idx in selected.index:

                row = zero_rows.loc[idx]

                examples.append({
                    "file": filename,
                    "row_index": int(idx),
                    "Flow Duration": row["Flow Duration"],
                    "Total Fwd Packets": row[
                        "Total Fwd Packets"
                    ],
                    "Total Backward Packets": row[
                        "Total Backward Packets"
                    ],
                    "Total Length of Fwd Packets": row[
                        "Total Length of Fwd Packets"
                    ],
                    "Total Length of Bwd Packets": row[
                        "Total Length of Bwd Packets"
                    ],
                    "Flow IAT Mean": row["Flow IAT Mean"],
                    "Flow IAT Max": row["Flow IAT Max"],
                    "Flow IAT Min": row["Flow IAT Min"],
                    "Fwd IAT Min": row["Fwd IAT Min"],
                    "Bwd IAT Min": row["Bwd IAT Min"],
                    "Fwd Header Length": row[
                        "Fwd Header Length"
                    ],
                    "Bwd Header Length": row[
                        "Bwd Header Length"
                    ],
                    "min_seg_size_forward": row[
                        "min_seg_size_forward"
                    ],
                    "Flow Bytes/s": row["Flow Bytes/s"],
                    "Flow Packets/s": row["Flow Packets/s"],
                    "Label": row["Label"],
                })

    files[filename] = file_zero_count


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("ZERO-DURATION FLOW AUDIT")
print("=" * 60)

print(f"Total Flow Duration == 0 rows : {total_zero_duration}")
print()

print("FLOW CONTENT")
print(
    f"Rows with forward packets     : "
    f"{zero_with_fwd_packets}"
)
print(
    f"Rows with backward packets    : "
    f"{zero_with_bwd_packets}"
)
print(
    f"Rows with any packet bytes    : "
    f"{zero_with_bytes}"
)

print()
print("NEGATIVE / SUSPICIOUS VALUES")
print(
    f"Rows with negative IAT values : "
    f"{zero_with_negative_iat}"
)
print(
    f"Rows with negative headers    : "
    f"{zero_with_negative_headers}"
)
print(
    f"Rows with negative min-seg    : "
    f"{zero_with_negative_min_seg}"
)

print()
print("LABEL DISTRIBUTION")

for label, count in labels.most_common():
    print(f"{label:<40} {count}")

print()
print("BY FILE")

for filename, count in files.items():
    print(f"{filename:<65} {count}")


# --------------------------------------------------
# Save examples
# --------------------------------------------------

examples_df = pd.DataFrame(examples)

output = REPORT_DIR / "zero_duration_examples.csv"

examples_df.to_csv(output, index=False)

print()
print(f"Examples saved to: {output}")