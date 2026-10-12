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

CHECK_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Min",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",
    "min_seg_size_forward",
    "Label",
]

results = []


for filename in FILES:
    path = RAW_DIR / filename

    print(f"Processing: {filename}")

    row_offset = 2

    for chunk in pd.read_csv(
        path,
        chunksize=100_000,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk.columns = chunk.columns.str.strip()

        for col in CHECK_COLUMNS:
            if col != "Label":
                chunk[col] = pd.to_numeric(
                    chunk[col],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Conditions
        # ----------------------------------------------------

        duration_bad = chunk["Flow Duration"] < 0
        flow_bytes_bad = chunk["Flow Bytes/s"] < 0

        header_bad = (
            (chunk["Fwd Header Length"] < 0)
            | (chunk["Bwd Header Length"] < 0)
            | (chunk["min_seg_size_forward"] < 0)
        )

        iat_min_bad = chunk["Flow IAT Min"] < -1
        fwd_iat_bad = chunk["Fwd IAT Min"] < -1

        # ----------------------------------------------------
        # Overlap counts
        # ----------------------------------------------------

        results.append({
            "file": filename,
            "duration_bad": int(duration_bad.sum()),
            "flow_bytes_bad": int(flow_bytes_bad.sum()),
            "both_duration_and_bytes": int(
                (duration_bad & flow_bytes_bad).sum()
            ),
            "header_bad": int(header_bad.sum()),
            "iat_min_below_minus_one": int(iat_min_bad.sum()),
            "fwd_iat_below_minus_one": int(fwd_iat_bad.sum()),
        })

        row_offset += len(chunk)


summary = pd.DataFrame(results)

print("\n" + "=" * 80)
print("OVERLAP SUMMARY BY FILE")
print("=" * 80)

print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("GLOBAL TOTALS")
print("=" * 80)

print(
    f"Negative Flow Duration: "
    f"{summary['duration_bad'].sum():,}"
)

print(
    f"Negative Flow Bytes/s: "
    f"{summary['flow_bytes_bad'].sum():,}"
)

print(
    f"Both Duration AND Bytes/s negative: "
    f"{summary['both_duration_and_bytes'].sum():,}"
)

print(
    f"Negative header/min-segment rows: "
    f"{summary['header_bad'].sum():,}"
)

print(
    f"Flow IAT Min < -1: "
    f"{summary['iat_min_below_minus_one'].sum():,}"
)

print(
    f"Fwd IAT Min < -1: "
    f"{summary['fwd_iat_below_minus_one'].sum():,}"
)

summary.to_csv(
    REPORT_DIR / "negative_overlap_summary.csv",
    index=False,
)

print(
    f"\nReport created: "
    f"{REPORT_DIR / 'negative_overlap_summary.csv'}"
)