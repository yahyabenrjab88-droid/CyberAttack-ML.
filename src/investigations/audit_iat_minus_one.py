from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

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
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Flow Bytes/s",
    "Flow Packets/s",
]

print("=" * 80)
print("AUDIT: IAT MIN == -1")
print("=" * 80)
print()

total_rows = 0
flow_iat_minus_one = 0
fwd_iat_minus_one = 0
both_minus_one = 0

label_flow_iat = {}
label_fwd_iat = {}

file_flow_iat = {}
file_fwd_iat = {}

representative_rows = []

for filename in FILES:
    path = ROOT / "data" / "raw" / filename

    print(f"Reading: {filename}")

    row_offset = 0

    try:
        reader = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="utf-8",
            chunksize=100_000,
        )
    except UnicodeDecodeError:
        reader = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
            chunksize=100_000,
        )

    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()

        numeric = chunk[FEATURES].apply(
            pd.to_numeric,
            errors="coerce"
        )

        flow_mask = numeric["Flow IAT Min"] == -1
        fwd_mask = numeric["Fwd IAT Min"] == -1
        both_mask = flow_mask & fwd_mask

        total_rows += len(chunk)

        flow_count = int(flow_mask.sum())
        fwd_count = int(fwd_mask.sum())
        both_count = int(both_mask.sum())

        flow_iat_minus_one += flow_count
        fwd_iat_minus_one += fwd_count
        both_minus_one += both_count

        # ------------------------------------------------------------
        # LABEL COUNTS
        # ------------------------------------------------------------

        for label, count in chunk.loc[flow_mask, "Label"].value_counts().items():
            label_flow_iat[label] = label_flow_iat.get(label, 0) + int(count)

        for label, count in chunk.loc[fwd_mask, "Label"].value_counts().items():
            label_fwd_iat[label] = label_fwd_iat.get(label, 0) + int(count)

        # ------------------------------------------------------------
        # FILE COUNTS
        # ------------------------------------------------------------

        file_flow_iat[filename] = file_flow_iat.get(filename, 0) + flow_count
        file_fwd_iat[filename] = file_fwd_iat.get(filename, 0) + fwd_count

        # ------------------------------------------------------------
        # REPRESENTATIVE ROWS
        # ------------------------------------------------------------

        selected_mask = flow_mask | fwd_mask

        selected = chunk.loc[selected_mask].copy()

        if not selected.empty:
            selected.insert(
                0,
                "source_file",
                filename
            )

            selected.insert(
                1,
                "source_row",
                selected.index + row_offset + 2
            )

            representative_rows.append(selected)

        row_offset += len(chunk)

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("GLOBAL SUMMARY")
print("=" * 80)
print()

print(f"Total dataset rows              : {total_rows:,}")
print(f"Flow IAT Min == -1              : {flow_iat_minus_one:,}")
print(f"Fwd IAT Min == -1               : {fwd_iat_minus_one:,}")
print(f"Both == -1                       : {both_minus_one:,}")
print()

# ============================================================================
# LABEL DISTRIBUTION
# ============================================================================

print("=" * 80)
print("FLOW IAT MIN == -1 — BY LABEL")
print("=" * 80)
print()

for label, count in sorted(
    label_flow_iat.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label:<35} {count:>8,}")

print()

print("=" * 80)
print("FWD IAT MIN == -1 — BY LABEL")
print("=" * 80)
print()

for label, count in sorted(
    label_fwd_iat.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{label:<35} {count:>8,}")

print()

# ============================================================================
# FILE DISTRIBUTION
# ============================================================================

print("=" * 80)
print("FLOW IAT MIN == -1 — BY FILE")
print("=" * 80)
print()

for filename, count in file_flow_iat.items():
    print(f"{filename:<70} {count:>8,}")

print()

print("=" * 80)
print("FWD IAT MIN == -1 — BY FILE")
print("=" * 80)
print()

for filename, count in file_fwd_iat.items():
    print(f"{filename:<70} {count:>8,}")

print()

# ============================================================================
# SAVE ALL MATCHING ROWS
# ============================================================================

if representative_rows:
    result = pd.concat(
        representative_rows,
        ignore_index=True
    )

    output = ROOT / "reports" / "iat_minus_one_rows.csv"

    result.to_csv(
        output,
        index=False,
        encoding="utf-8"
    )

    print("=" * 80)
    print("MATCHING ROWS SAVED")
    print("=" * 80)
    print()

    print(f"Rows saved : {len(result):,}")
    print(f"Output     : {output}")
    print()

    # =========================================================================
    # REPRESENTATIVE SAMPLE
    # =========================================================================

    print("=" * 80)
    print("REPRESENTATIVE SAMPLE")
    print("=" * 80)
    print()

    sample_columns = [
        "source_file",
        "source_row",
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
        "Bwd IAT Mean",
        "Bwd IAT Max",
        "Bwd IAT Min",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Label",
    ]

    print(
        result[sample_columns]
        .sample(
            n=min(30, len(result)),
            random_state=42
        )
        .to_string(index=False)
    )

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)