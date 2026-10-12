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

TARGETS = [
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

rows = []

print("=" * 80)
print("AUDITING REMAINING FLOW IAT MIN OVERLAPS")
print("=" * 80)
print()

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

        numeric = chunk[TARGETS].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Flow IAT Min < -1
        iat_anomaly = numeric["Flow IAT Min"] < -1

        # Exclude rows already explained by negative Flow Duration
        valid_duration = numeric["Flow Duration"] >= 0

        # Other suspicious negative features
        other_bad = (
            (numeric["Flow Bytes/s"] < 0)
            | (numeric["Flow Packets/s"] < 0)
            | (numeric["Flow IAT Mean"] < 0)
            | (numeric["Flow IAT Max"] < 0)
            | (numeric["Fwd IAT Min"] < -1)
            | (numeric["Fwd Header Length"] < 0)
            | (numeric["Bwd Header Length"] < 0)
            | (numeric["Fwd Header Length.1"] < 0)
            | (numeric["min_seg_size_forward"] < 0)
        )

        mask = iat_anomaly & valid_duration & other_bad

        selected = chunk.loc[mask].copy()

        if not selected.empty:
            # Original CSV row number.
            # +2 because row 1 is the CSV header.
            source_rows = selected.index + row_offset + 2

            selected.insert(0, "source_file", filename)
            selected.insert(1, "source_row", source_rows)

            rows.append(selected)

        row_offset += len(chunk)

if rows:
    result = pd.concat(rows, ignore_index=True)

    output = ROOT / "reports" / "remaining_iat_overlap.csv"

    result.to_csv(
        output,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print()

    print(f"Rows found: {len(result)}")
    print()

    display_columns = [
        "source_file",
        "source_row",
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Min",
        "Fwd Header Length",
        "Bwd Header Length",
        "min_seg_size_forward",
        "Label",
    ]

    print(result[display_columns].to_string(index=False))

    print()
    print(f"Full output: {output}")

else:
    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print()
    print("No remaining overlapping rows found.")