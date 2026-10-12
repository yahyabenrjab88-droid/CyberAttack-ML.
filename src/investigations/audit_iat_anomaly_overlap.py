from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"

path = REPORT_DIR / "flow_iat_min_below_minus_one.csv"

df = pd.read_csv(path)

numeric_cols = [
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

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


checks = {
    "negative_flow_duration": df["Flow Duration"] < 0,
    "negative_flow_bytes": df["Flow Bytes/s"] < 0,
    "negative_flow_packets": df["Flow Packets/s"] < 0,
    "negative_flow_iat_mean": df["Flow IAT Mean"] < 0,
    "negative_flow_iat_max": df["Flow IAT Max"] < 0,
    "negative_fwd_iat_min": df["Fwd IAT Min"] < 0,
    "negative_fwd_header": df["Fwd Header Length"] < 0,
    "negative_bwd_header": df["Bwd Header Length"] < 0,
    "negative_min_seg": df["min_seg_size_forward"] < 0,
}

print("\n" + "=" * 80)
print("FLOW IAT MIN < -1 : ANOMALY OVERLAP")
print("=" * 80)

print(f"\nTotal rows: {len(df):,}")

for name, mask in checks.items():
    print(f"{name:35s}: {int(mask.sum()):,}")

# Rows having ANY other suspicious negative feature
other_bad = pd.Series(False, index=df.index)

for name, mask in checks.items():
    if name != "negative_flow_iat_mean":
        other_bad |= mask

# Exclude the Flow IAT Min condition itself.
other_bad = (
    (df["Flow Duration"] < 0)
    | (df["Flow Bytes/s"] < 0)
    | (df["Flow Packets/s"] < 0)
    | (df["Flow IAT Mean"] < 0)
    | (df["Flow IAT Max"] < 0)
    | (df["Fwd IAT Min"] < 0)
    | (df["Fwd Header Length"] < 0)
    | (df["Bwd Header Length"] < 0)
    | (df["min_seg_size_forward"] < 0)
)

print(
    f"\nRows with another negative suspicious feature: "
    f"{int(other_bad.sum()):,}"
)

print(
    f"Rows with ONLY Flow IAT Min < -1 as suspicious negative: "
    f"{int((~other_bad).sum()):,}"
)

print("\nBy label:")
print(df["Label"].value_counts().to_string())

print("\nExact Flow IAT Min values:")
print(
    df["Flow IAT Min"]
    .value_counts()
    .sort_index()
    .to_string()
)