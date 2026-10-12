from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# FEATURES WE STILL WANT TO CHECK
# ============================================================

CHECK_COLUMNS = [
    "Flow Duration",

    "Total Fwd Packets",
    "Total Backward Packets",

    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",

    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow IAT Min",

    "Fwd IAT Min",
    "Fwd IAT Mean",
    "Fwd IAT Max",

    "Bwd IAT Min",
    "Bwd IAT Mean",
    "Bwd IAT Max",

    "Fwd Header Length",
    "Bwd Header Length",

    "min_seg_size_forward",

    "Flow Bytes/s",
    "Flow Packets/s",

    "Destination Port",

    "Fwd Packet Length Min",
    "Fwd Packet Length Max",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",

    "Bwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",

    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",

    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",

    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
]


# ============================================================
# COUNTERS
# ============================================================

total_rows = 0

missing_counts = Counter()
pos_inf_counts = Counter()
neg_inf_counts = Counter()
negative_counts = Counter()

zero_duration = 0
negative_duration = 0

invalid_ports = 0

negative_packet_counts = 0
negative_byte_counts = 0
negative_packet_lengths = 0

iat_below_minus_one = 0
fwd_iat_below_minus_one = 0

zero_duration_negative_iat = 0

problem_examples = []


# ============================================================
# PROCESS
# ============================================================

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

    for chunk in reader:

        chunk.columns = chunk.columns.str.strip()

        total_rows += len(chunk)

        # ----------------------------------------------------
        # Only use columns that actually exist
        # ----------------------------------------------------

        available = [
            c for c in CHECK_COLUMNS
            if c in chunk.columns
        ]

        numeric = chunk[available].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        for column in available:

            count = int(
                numeric[column].isna().sum()
            )

            # Only report columns where missing values exist
            if count:
                missing_counts[column] += count

        # ----------------------------------------------------
        # Infinity
        # ----------------------------------------------------

        for column in available:

            pos = int(
                np.isposinf(numeric[column]).sum()
            )

            neg = int(
                np.isneginf(numeric[column]).sum()
            )

            if pos:
                pos_inf_counts[column] += pos

            if neg:
                neg_inf_counts[column] += neg

        # ----------------------------------------------------
        # Negative values
        # ----------------------------------------------------

        for column in available:

            count = int(
                (numeric[column] < 0).sum()
            )

            if count:
                negative_counts[column] += count

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration = numeric["Flow Duration"]

        zero_mask = duration == 0
        negative_mask = duration < 0

        zero_duration += int(
            zero_mask.sum()
        )

        negative_duration += int(
            negative_mask.sum()
        )

        # ----------------------------------------------------
        # IAT anomalies
        # ----------------------------------------------------

        iat_bad = (
            numeric["Flow IAT Min"] < -1
        )

        fwd_iat_bad = (
            numeric["Fwd IAT Min"] < -1
        )

        iat_below_minus_one += int(
            iat_bad.sum()
        )

        fwd_iat_below_minus_one += int(
            fwd_iat_bad.sum()
        )

        zero_duration_negative_iat += int(
            (
                zero_mask
                &
                (
                    (numeric["Flow IAT Min"] < 0)
                    |
                    (numeric["Fwd IAT Min"] < 0)
                )
            ).sum()
        )

        # ----------------------------------------------------
        # Ports
        # ----------------------------------------------------

        ports = numeric["Destination Port"]

        invalid_ports += int(
            (
                (ports < 0)
                |
                (ports > 65535)
            ).sum()
        )

        # ----------------------------------------------------
        # Packet counts
        # ----------------------------------------------------

        packet_columns = [
            "Total Fwd Packets",
            "Total Backward Packets",
            "Subflow Fwd Packets",
            "Subflow Bwd Packets",
        ]

        for column in packet_columns:

            negative_packet_counts += int(
                (numeric[column] < 0).sum()
            )

        # ----------------------------------------------------
        # Byte counts
        # ----------------------------------------------------

        byte_columns = [
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Bytes",
        ]

        for column in byte_columns:

            negative_byte_counts += int(
                (numeric[column] < 0).sum()
            )

        # ----------------------------------------------------
        # Packet lengths
        # ----------------------------------------------------

        packet_length_columns = [
            "Fwd Packet Length Min",
            "Fwd Packet Length Max",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Min",
            "Bwd Packet Length Max",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
        ]

        for column in packet_length_columns:

            negative_packet_lengths += int(
                (numeric[column] < 0).sum()
            )

        # ----------------------------------------------------
        # Save examples of remaining suspicious rows
        # ----------------------------------------------------

        suspicious = (
            (numeric["Flow Duration"] < 0)
            |
            (numeric["Flow IAT Min"] < -1)
            |
            (numeric["Fwd IAT Min"] < -1)
            |
            (numeric["Fwd Header Length"] < 0)
            |
            (numeric["Bwd Header Length"] < 0)
            |
            (numeric["min_seg_size_forward"] < 0)
        )

        selected = chunk.loc[suspicious]

        if len(problem_examples) < 30:

            for idx in selected.index[
                :30 - len(problem_examples)
            ]:

                row = chunk.loc[idx]

                problem_examples.append({
                    "file": filename,
                    "row_index": int(idx),
                    "Flow Duration": row["Flow Duration"],
                    "Flow IAT Min": row["Flow IAT Min"],
                    "Fwd IAT Min": row["Fwd IAT Min"],
                    "Fwd Header Length": row[
                        "Fwd Header Length"
                    ],
                    "Bwd Header Length": row[
                        "Bwd Header Length"
                    ],
                    "min_seg_size_forward": row[
                        "min_seg_size_forward"
                    ],
                    "Label": row["Label"],
                })


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL REMAINING AUDIT")
print("=" * 70)

print()
print("DATASET")
print(f"Rows: {total_rows}")

print()
print("MISSING VALUES")

if missing_counts:

    for column, count in missing_counts.most_common():
        print(f"{column:<45} {count}")

else:
    print("None")


print()
print("INFINITY")

print("Positive infinity:")

if pos_inf_counts:

    for column, count in pos_inf_counts.most_common():
        print(f"  {column:<43} {count}")

else:
    print("  None")

print("Negative infinity:")

if neg_inf_counts:

    for column, count in neg_inf_counts.most_common():
        print(f"  {column:<43} {count}")

else:
    print("  None")


print()
print("NEGATIVE VALUES")

for column, count in negative_counts.most_common():

    print(f"{column:<45} {count}")


print()
print("FLOW VALIDITY")

print(
    f"Flow Duration == 0        : "
    f"{zero_duration}"
)

print(
    f"Flow Duration < 0         : "
    f"{negative_duration}"
)

print(
    f"Flow IAT Min < -1         : "
    f"{iat_below_minus_one}"
)

print(
    f"Fwd IAT Min < -1          : "
    f"{fwd_iat_below_minus_one}"
)

print(
    f"Zero-duration + neg IAT   : "
    f"{zero_duration_negative_iat}"
)


print()
print("OTHER VALIDITY")

print(
    f"Invalid destination ports : "
    f"{invalid_ports}"
)

print(
    f"Negative packet counts    : "
    f"{negative_packet_counts}"
)

print(
    f"Negative byte counts      : "
    f"{negative_byte_counts}"
)

print(
    f"Negative packet lengths   : "
    f"{negative_packet_lengths}"
)


# ============================================================
# SAVE EXAMPLES
# ============================================================

examples_df = pd.DataFrame(
    problem_examples
)

examples_path = (
    REPORT_DIR /
    "final_remaining_suspicious_examples.csv"
)

examples_df.to_csv(
    examples_path,
    index=False
)

print()
print(
    f"Suspicious examples saved to: "
    f"{examples_path}"
)

print()
print("=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)