from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

# Used only for correlation / quantiles.
# We do NOT need every 2.8 million rows for those.
SAMPLE_PER_CHUNK = 2_500
RANDOM_SEED = 42


# ============================================================
# FIRST PASS: EXACT STATISTICS OVER ALL ROWS
# ============================================================

stats = {}
feature_columns = None
total_rows = 0

rng = np.random.default_rng(RANDOM_SEED)

sample_parts = []


for filename in RAW_FILES:

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

        # Clean column names in memory only.
        chunk.columns = chunk.columns.str.strip()

        if feature_columns is None:
            if "Label" not in chunk.columns:
                raise ValueError("Label column not found.")

            feature_columns = [
                col for col in chunk.columns
                if col != "Label"
            ]

            for col in feature_columns:
                stats[col] = {
                    "missing": 0,
                    "plus_inf": 0,
                    "minus_inf": 0,
                    "negative": 0,
                    "zero": 0,
                    "minus_one": 0,
                    "numeric_failures": 0,
                    "finite_count": 0,
                    "min": np.inf,
                    "max": -np.inf,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                }

        total_rows += len(chunk)

        numeric = chunk[feature_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # ----------------------------------------------------
        # Exact statistics for every row
        # ----------------------------------------------------

        for col in feature_columns:

            values = numeric[col]

            missing = values.isna()
            plus_inf = np.isposinf(values)
            minus_inf = np.isneginf(values)
            finite = np.isfinite(values)

            s = stats[col]

            s["missing"] += int(missing.sum())
            s["plus_inf"] += int(plus_inf.sum())
            s["minus_inf"] += int(minus_inf.sum())

            finite_values = values[finite]

            if len(finite_values) > 0:

                s["negative"] += int((finite_values < 0).sum())
                s["zero"] += int((finite_values == 0).sum())
                s["minus_one"] += int((finite_values == -1).sum())

                s["finite_count"] += len(finite_values)

                current_min = finite_values.min()
                current_max = finite_values.max()

                if current_min < s["min"]:
                    s["min"] = float(current_min)

                if current_max > s["max"]:
                    s["max"] = float(current_max)

                s["sum"] += float(finite_values.sum())
                s["sum_sq"] += float((finite_values ** 2).sum())

        # ----------------------------------------------------
        # Deterministic random sample for distribution /
        # correlation analysis
        # ----------------------------------------------------

        sample_size = min(SAMPLE_PER_CHUNK, len(chunk))

        if sample_size > 0:
            sampled = numeric.sample(
                n=sample_size,
                random_state=RANDOM_SEED
            )
            sample_parts.append(sampled)


# ============================================================
# BUILD SUMMARY
# ============================================================

summary_rows = []

for col in feature_columns:

    s = stats[col]

    finite_count = s["finite_count"]

    if finite_count > 0:

        mean = s["sum"] / finite_count

        variance = (
            s["sum_sq"] / finite_count
            - mean ** 2
        )

        variance = max(variance, 0.0)

        std = np.sqrt(variance)

    else:
        mean = np.nan
        std = np.nan

    summary_rows.append({
        "feature": col,
        "missing": s["missing"],
        "plus_inf": s["plus_inf"],
        "minus_inf": s["minus_inf"],
        "negative": s["negative"],
        "zero": s["zero"],
        "minus_one": s["minus_one"],
        "numeric_failures": s["numeric_failures"],
        "finite_count": finite_count,
        "min": s["min"] if finite_count else np.nan,
        "max": s["max"] if finite_count else np.nan,
        "mean": mean,
        "std": std,
        "zero_pct": 100 * s["zero"] / total_rows,
        "negative_pct": 100 * s["negative"] / total_rows,
        "minus_one_pct": 100 * s["minus_one"] / total_rows,
        "missing_pct": 100 * s["missing"] / total_rows,
        "plus_inf_pct": 100 * s["plus_inf"] / total_rows,
    })


summary = pd.DataFrame(summary_rows)


# ============================================================
# SAMPLE QUANTILES
# ============================================================

sample_df = pd.concat(sample_parts, ignore_index=True)

quantiles = sample_df.quantile(
    [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
)

summary["q01"] = summary["feature"].map(
    quantiles.loc[0.01].to_dict()
)

summary["q05"] = summary["feature"].map(
    quantiles.loc[0.05].to_dict()
)

summary["q25"] = summary["feature"].map(
    quantiles.loc[0.25].to_dict()
)

summary["q50"] = summary["feature"].map(
    quantiles.loc[0.50].to_dict()
)

summary["q75"] = summary["feature"].map(
    quantiles.loc[0.75].to_dict()
)

summary["q95"] = summary["feature"].map(
    quantiles.loc[0.95].to_dict()
)

summary["q99"] = summary["feature"].map(
    quantiles.loc[0.99].to_dict()
)


# ============================================================
# ZERO-HEAVY / NEAR-CONSTANT FEATURES
# ============================================================

summary["zero_heavy"] = summary["zero_pct"] >= 99.0
summary["extremely_zero_heavy"] = summary["zero_pct"] >= 99.9


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

print("\nCalculating correlations from deterministic sample...")

corr = sample_df.corr()

correlation_rows = []

for i, col1 in enumerate(feature_columns):

    for j in range(i + 1, len(feature_columns)):

        col2 = feature_columns[j]

        value = corr.loc[col1, col2]

        if pd.notna(value):

            correlation_rows.append({
                "feature_1": col1,
                "feature_2": col2,
                "correlation": float(value),
                "absolute_correlation": abs(float(value)),
            })


correlation_df = pd.DataFrame(correlation_rows)

if not correlation_df.empty:
    correlation_df = correlation_df.sort_values(
        "absolute_correlation",
        ascending=False
    )


# ============================================================
# IDENTICAL-ON-SAMPLE FEATURES
# ============================================================

identical_sample_pairs = []

for i, col1 in enumerate(feature_columns):

    for j in range(i + 1, len(feature_columns)):

        col2 = feature_columns[j]

        a = sample_df[col1]
        b = sample_df[col2]

        mask = a.notna() & b.notna()

        if mask.sum() == 0:
            continue

        if (a[mask] == b[mask]).all():

            identical_sample_pairs.append({
                "feature_1": col1,
                "feature_2": col2,
                "matching_sample_values": int(mask.sum()),
            })


identical_sample_df = pd.DataFrame(identical_sample_pairs)


# ============================================================
# SAVE REPORTS
# ============================================================

summary_path = REPORTS_DIR / "feature_quality_summary.csv"
corr_path = REPORTS_DIR / "feature_correlation.csv"
identical_path = REPORTS_DIR / "feature_identical_sample.csv"

summary.to_csv(summary_path, index=False)
correlation_df.to_csv(corr_path, index=False)
identical_sample_df.to_csv(identical_path, index=False)


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FEATURE QUALITY AUDIT")
print("=" * 80)

print(f"Total rows inspected : {total_rows:,}")
print(f"Feature columns      : {len(feature_columns)}")
print(f"Correlation sample   : {len(sample_df):,} rows")


print("\n--- ZERO-HEAVY FEATURES (>=99% zeros) ---")

zero_heavy = summary[
    summary["zero_heavy"]
].sort_values("zero_pct", ascending=False)

if zero_heavy.empty:
    print("None")
else:
    for _, row in zero_heavy.iterrows():
        print(
            f"{row['feature']:<35} "
            f"{row['zero_pct']:.4f}% zeros"
        )


print("\n--- EXTREMELY ZERO-HEAVY FEATURES (>=99.9% zeros) ---")

extreme_zero = summary[
    summary["extremely_zero_heavy"]
].sort_values("zero_pct", ascending=False)

if extreme_zero.empty:
    print("None")
else:
    for _, row in extreme_zero.iterrows():
        print(
            f"{row['feature']:<35} "
            f"{row['zero_pct']:.4f}% zeros"
        )


print("\n--- FEATURES WITH NEGATIVE VALUES ---")

negative_features = summary[
    summary["negative"] > 0
].sort_values("negative", ascending=False)

for _, row in negative_features.iterrows():
    print(
        f"{row['feature']:<35} "
        f"{row['negative']:,} negative "
        f"({row['negative_pct']:.4f}%)"
    )


print("\n--- EXTREME MIN/MAX VALUES ---")

for _, row in summary.sort_values(
    "max",
    ascending=False
).head(15).iterrows():

    print(
        f"{row['feature']:<35} "
        f"min={row['min']:.4g} "
        f"max={row['max']:.4g}"
    )


print("\n--- HIGHEST CORRELATIONS ---")

if correlation_df.empty:
    print("No correlations calculated.")
else:

    for _, row in correlation_df.head(20).iterrows():

        print(
            f"{row['feature_1']:<35} <-> "
            f"{row['feature_2']:<35} "
            f"r={row['correlation']:.6f}"
        )


print("\n--- IDENTICAL FEATURES ON AUDIT SAMPLE ---")

if identical_sample_df.empty:
    print("None")
else:

    for _, row in identical_sample_df.iterrows():

        print(
            f"{row['feature_1']} <-> "
            f"{row['feature_2']} "
            f"({row['matching_sample_values']:,} matching values)"
        )


print("\nReports created:")

print(f"  {summary_path}")
print(f"  {corr_path}")
print(f"  {identical_path}")

print("\nFEATURE AUDIT COMPLETE")