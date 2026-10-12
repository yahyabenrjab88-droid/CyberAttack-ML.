from pathlib import Path
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

# Columns useful for understanding whether a row is actually corrupt.
CONTEXT_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
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
    "Label",
]

# Exact situations we want to inspect.
CONDITIONS = {
    "negative_flow_duration": lambda df: df["Flow Duration"] < 0,

    "flow_iat_min_below_minus_one": lambda df: df["Flow IAT Min"] < -1,

    "negative_fwd_iat_min_below_minus_one": lambda df: df["Fwd IAT Min"] < -1,

    "negative_header_lengths": lambda df: (
        (df["Fwd Header Length"] < 0)
        | (df["Bwd Header Length"] < 0)
    ),

    "negative_min_seg_size": lambda df: df["min_seg_size_forward"] < 0,

    "negative_flow_bytes": lambda df: df["Flow Bytes/s"] < 0,
}


all_results = {
    name: []
    for name in CONDITIONS
}


for filename in FILES:
    path = RAW_DIR / filename

    print(f"Processing: {filename}")

    for chunk_start, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=100_000,
            low_memory=False,
            encoding="utf-8",
        )
    ):
        chunk.columns = chunk.columns.str.strip()

        # Convert only the columns we need.
        for column in CONTEXT_COLUMNS:
            if column != "Label":
                chunk[column] = pd.to_numeric(
                    chunk[column],
                    errors="coerce",
                )

        for condition_name, condition in CONDITIONS.items():
            mask = condition(chunk)

            if not mask.any():
                continue

            selected = chunk.loc[
                mask,
                CONTEXT_COLUMNS
            ].copy()

            selected.insert(0, "source_file", filename)
            selected.insert(
                1,
                "source_row",
                range(
                    chunk_start * 100_000 + 2,
                    chunk_start * 100_000 + 2 + len(selected)
                ),
            )

            all_results[condition_name].append(selected)


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 80)
print("TARGETED NEGATIVE-ROW AUDIT")
print("=" * 80)

for condition_name, pieces in all_results.items():

    if not pieces:
        print(f"\n{condition_name}: 0 rows")
        continue

    result = pd.concat(pieces, ignore_index=True)

    output = REPORT_DIR / f"{condition_name}.csv"
    result.to_csv(output, index=False)

    print(f"\n{condition_name}")
    print("-" * len(condition_name))
    print(f"Rows: {len(result):,}")
    print(f"Output: {output}")

    print("\nBy label:")
    print(
        result["Label"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nBy file:")
    print(
        result["source_file"]
        .value_counts()
        .to_string()
    )


print("\nReports created in:")
print(REPORT_DIR)