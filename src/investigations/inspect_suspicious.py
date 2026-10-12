from pathlib import Path
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


# Columns containing values that need closer inspection
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


all_suspicious = []


for filename in RAW_FILES:
    path = RAW_DIR / filename

    print(f"Processing: {filename}")

    for chunk in pd.read_csv(
        path,
        dtype=str,
        chunksize=CHUNK_SIZE,
        low_memory=True,
    ):
        # Clean column names in memory only.
        # The original CSV files are NOT modified.
        chunk.columns = chunk.columns.str.strip()

        # Convert only the columns we are inspecting to numeric.
        numeric = chunk[CHECK_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )

        # Start with no suspicious rows.
        mask = pd.Series(False, index=chunk.index)

        # Missing Flow Bytes/s
        mask |= chunk["Flow Bytes/s"].isna()

        # Positive infinity
        mask |= np.isposinf(numeric["Flow Bytes/s"])
        mask |= np.isposinf(numeric["Flow Packets/s"])

        # Negative timing values
        for column in [
            "Flow Duration",
            "Flow IAT Mean",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Min",
        ]:
            mask |= numeric[column] < 0

        # Negative header-related values
        mask |= numeric["Fwd Header Length"] < 0
        mask |= numeric["Bwd Header Length"] < 0
        mask |= numeric["Fwd Header Length.1"] < 0
        mask |= numeric["min_seg_size_forward"] < 0

        suspicious = chunk.loc[mask].copy()

        if not suspicious.empty:
            suspicious.insert(0, "source_file", filename)

            # Add numeric versions so we can inspect the actual values easily.
            for column in CHECK_COLUMNS:
                suspicious[f"NUM__{column}"] = pd.to_numeric(
                    suspicious[column],
                    errors="coerce",
                )

            all_suspicious.append(suspicious)


# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------

if not all_suspicious:
    print("No suspicious rows found.")
    raise SystemExit(0)


result = pd.concat(
    all_suspicious,
    ignore_index=True,
)


output = REPORTS_DIR / "suspicious_rows.csv"

result.to_csv(
    output,
    index=False,
    encoding="utf-8",
)


print()
print("=" * 80)
print("SUSPICIOUS ROW INSPECTION COMPLETE")
print("=" * 80)
print(f"Rows collected: {len(result):,}")
print(f"Output: {output}")
print()