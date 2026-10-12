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
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Flow IAT Min",
    "Flow IAT Mean",
    "Flow IAT Max",
    "Flow Bytes/s",
    "Flow Packets/s",
]

all_rows = []

total_rows = 0

negative_header_rows = 0
negative_fwd_header = 0
negative_bwd_header = 0
negative_min_seg = 0

overlap_duration = 0
overlap_any_header = 0

print("=" * 80)
print("AUDIT: NEGATIVE HEADER LENGTHS / MIN SEGMENT SIZE")
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

        numeric = chunk[FEATURES].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # ------------------------------------------------------------
        # NEGATIVE HEADER / MIN SEGMENT MASKS
        # ------------------------------------------------------------

        fwd_header_bad = numeric["Fwd Header Length"] < 0
        bwd_header_bad = numeric["Bwd Header Length"] < 0
        fwd_header_duplicate_bad = numeric["Fwd Header Length.1"] < 0
        min_seg_bad = numeric["min_seg_size_forward"] < 0

        any_header_bad = (
            fwd_header_bad
            | bwd_header_bad
            | fwd_header_duplicate_bad
            | min_seg_bad
        )

        duration_bad = numeric["Flow Duration"] < 0

        # ------------------------------------------------------------
        # GLOBAL COUNTS
        # ------------------------------------------------------------

        negative_fwd_header += int(fwd_header_bad.sum())
        negative_bwd_header += int(bwd_header_bad.sum())
        negative_min_seg += int(min_seg_bad.sum())

        negative_header_rows += int(any_header_bad.sum())

        overlap_duration += int(
            (any_header_bad & duration_bad).sum()
        )

        overlap_any_header += int(
            any_header_bad.sum()
        )

        # ------------------------------------------------------------
        # SAVE ALL UNIQUE SUSPICIOUS ROWS
        # ------------------------------------------------------------

        selected = chunk.loc[any_header_bad].copy()

        if not selected.empty:

            source_rows = selected.index + row_offset + 2

            selected.insert(
                0,
                "source_file",
                filename
            )

            selected.insert(
                1,
                "source_row",
                source_rows
            )

            all_rows.append(selected)

        total_rows += len(chunk)
        row_offset += len(chunk)

# ============================================================================
# RESULTS
# ============================================================================

print()
print("=" * 80)
print("GLOBAL SUMMARY")
print("=" * 80)
print()

print(f"Total dataset rows                 : {total_rows:,}")
print(f"Rows with negative Fwd Header     : {negative_fwd_header:,}")
print(f"Rows with negative Bwd Header     : {negative_bwd_header:,}")
print(f"Rows with negative min_seg_size   : {negative_min_seg:,}")
print(f"Unique rows with any of these     : {negative_header_rows:,}")
print()
print(
    f"Header/min-segment rows also with "
    f"negative Flow Duration            : {overlap_duration:,}"
)

# ============================================================================
# SAVE RESULTS
# ============================================================================

if all_rows:

    result = pd.concat(
        all_rows,
        ignore_index=True
    )

    output = ROOT / "reports" / "negative_header_rows.csv"

    result.to_csv(
        output,
        index=False,
        encoding="utf-8"
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print()

    print(f"Rows saved : {len(result):,}")
    print(f"Output     : {output}")

    # ========================================================================
    # LABEL DISTRIBUTION
    # ========================================================================

    print()
    print("=" * 80)
    print("BY LABEL")
    print("=" * 80)
    print()

    print(
        result["Label"]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # FILE DISTRIBUTION
    # ========================================================================

    print()
    print("=" * 80)
    print("BY FILE")
    print("=" * 80)
    print()

    print(
        result["source_file"]
        .value_counts()
        .to_string()
    )

    # ========================================================================
    # REPRESENTATIVE ROWS
    # ========================================================================

    print()
    print("=" * 80)
    print("REPRESENTATIVE ROWS")
    print("=" * 80)
    print()

    display_columns = [
        "source_file",
        "source_row",
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Fwd Header Length",
        "Bwd Header Length",
        "Fwd Header Length.1",
        "min_seg_size_forward",
        "Flow IAT Min",
        "Flow IAT Mean",
        "Flow IAT Max",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Label",
    ]

    print(
        result[display_columns]
        .to_string(index=False)
    )

else:

    print()
    print("No negative header/min-segment rows found.")

print()
print("=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)