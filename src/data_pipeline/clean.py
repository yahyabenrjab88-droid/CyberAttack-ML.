"""
clean.py

Final CICIDS2017 cleaning pipeline based on the completed raw-data audit.

IMPORTANT:
- data/raw/ is NEVER modified.
- This script performs data cleaning only.
- No train/validation/test split.
- No scaling.
- No class balancing.
- No ML training.
- No aggressive outlier removal.
- No arbitrary threshold-based deletion.

Cleaning decisions:

REMOVE ROWS:
1. Flow Duration < 0
2. Negative Fwd Header Length
3. Negative Bwd Header Length
4. Negative min_seg_size_forward
5. Rows belonging to conflicting exact 78-feature-vector groups

CELL-LEVEL CLEANING:
1. Flow IAT Min <= -1 -> NaN
2. Fwd IAT Min <= -1 -> NaN
3. +/- infinity -> NaN

LABEL CLEANING:
1. Normalize corrupted Web Attack labels.

FEATURE COLUMNS:
1. Remove 5 verified exact duplicate columns.
2. Keep 100%-zero columns for later feature engineering.

OUTPUT:
data/processed/cleaned.parquet

REPORTS:
reports/cleaning_summary.csv
reports/cleaning_summary.md
"""

from __future__ import annotations

import re
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

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

LABEL_COL = "Label"

# Invalid-row checks
FLOW_DURATION = "Flow Duration"
FWD_HEADER = "Fwd Header Length"
BWD_HEADER = "Bwd Header Length"
MIN_SEG = "min_seg_size_forward"

# Sentinel / invalid cell checks
FLOW_IAT_MIN = "Flow IAT Min"
FWD_IAT_MIN = "Fwd IAT Min"


# These were verified as EXACT duplicates on the complete raw dataset.
EXACT_DUPLICATE_PAIRS = [
    ("Fwd Header Length", "Fwd Header Length.1"),
    ("Total Fwd Packets", "Subflow Fwd Packets"),
    ("Total Backward Packets", "Subflow Bwd Packets"),
    ("Fwd URG Flags", "CWE Flag Count"),
    ("Fwd PSH Flags", "SYN Flag Count"),
]

REMOVE_EXACT_DUPLICATE_COLUMNS = True

CONFLICT_IMPACT_WARN_THRESHOLD = 0.20


# The audit found these three corrupted label families.
WEB_ATTACK_PATTERN = re.compile(
    r"^Web Attack\s+.*?\s+(Brute Force|XSS|Sql Injection)$"
)


# ============================================================
# LOGGING
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


# ============================================================
# STATISTICS
# ============================================================

class CleaningStats:

    def __init__(self):
        self.reference_columns = None
        self.feature_columns = None

        self.raw_rows = 0

        self.raw_label_counts = Counter()

        self.label_changes = Counter()

        self.numeric_conversion_failures = 0

        self.removed_negative_duration = 0
        self.removed_invalid_headers = 0

        self.conflicting_groups = 0
        self.removed_conflict_rows = 0

        self.conflict_removed_labels = Counter()
        self.conflict_class_impact = []

        self.iat_nan_counts = Counter()
        self.infinity_nan_counts = Counter()

        self.duplicate_columns_removed = []

        self.constant_zero_columns = []

        self.final_rows = 0
        self.final_columns = 0

        self.final_label_counts = Counter()

        self.missing_before = Counter()
        self.missing_after = Counter()


# ============================================================
# RAW FILE READING
# ============================================================

def read_raw_file(path: Path) -> pd.DataFrame:
    """
    Read one raw CSV as strings.

    We intentionally avoid automatic numeric dtype inference here so
    that conversion is controlled explicitly.
    """

    try:
        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
        )

    except UnicodeDecodeError:
        log(
            f"  UTF-8 failed for {path.name}; "
            f"retrying with latin1."
        )

        return pd.read_csv(
            path,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
        )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_labels(
    labels: pd.Series,
    stats: CleaningStats,
) -> pd.Series:

    labels = labels.str.strip()

    mask = labels.str.match(
        WEB_ATTACK_PATTERN,
        na=False,
    )

    extracted = labels[mask].str.extract(
        WEB_ATTACK_PATTERN,
        expand=False,
    )

    for original, attack_type in zip(
        labels[mask],
        extracted,
    ):
        normalized = f"Web Attack - {attack_type}"

        if original != normalized:
            stats.label_changes[(original, normalized)] += 1

    labels.loc[mask] = (
        "Web Attack - " + extracted
    )

    return labels


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_features_to_numeric(
    df: pd.DataFrame,
    feature_columns: list[str],
    stats: CleaningStats,
    filename: str,
) -> pd.DataFrame:

    for col in feature_columns:

        original = df[col]

        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        unexpected = (
            original.notna()
            & numeric.isna()
        )

        count = int(unexpected.sum())

        if count:
            examples = (
                original[unexpected]
                .drop_duplicates()
                .head(5)
                .tolist()
            )

            log(
                f"  WARNING: {count:,} unexpected numeric "
                f"conversion failures in {col} "
                f"({filename}). Examples: {examples}"
            )

            stats.numeric_conversion_failures += count

        df[col] = numeric

    return df


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    filename: str,
    stats: CleaningStats,
) -> pd.DataFrame:

    path = RAW_DIR / filename

    log(f"\nProcessing: {filename}")

    df = read_raw_file(path)

    # Strip column-name whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    # --------------------------------------------------------
    # Verify structure
    # --------------------------------------------------------

    if stats.reference_columns is None:

        stats.reference_columns = list(df.columns)

        if LABEL_COL not in df.columns:
            raise ValueError(
                f"{LABEL_COL!r} not found in {filename}"
            )

        stats.feature_columns = [
            c for c in df.columns
            if c != LABEL_COL
        ]

    else:

        if list(df.columns) != stats.reference_columns:
            raise ValueError(
                f"Column structure mismatch in {filename}.\n"
                f"The raw files must have identical columns."
            )

    required = [
        FLOW_DURATION,
        FWD_HEADER,
        BWD_HEADER,
        MIN_SEG,
        FLOW_IAT_MIN,
        FWD_IAT_MIN,
        LABEL_COL,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {filename}: {missing}"
        )

    stats.raw_rows += len(df)

    # --------------------------------------------------------
    # Raw label counts
    # --------------------------------------------------------

    stats.raw_label_counts.update(
        df[LABEL_COL]
        .str.strip()
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Normalize labels
    # --------------------------------------------------------

    df[LABEL_COL] = normalize_labels(
        df[LABEL_COL],
        stats,
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df = convert_features_to_numeric(
        df,
        stats.feature_columns,
        stats,
        filename,
    )

    # --------------------------------------------------------
    # Missing-value audit before cleaning
    # --------------------------------------------------------

    for col in stats.feature_columns:
        stats.missing_before[col] += int(
            df[col].isna().sum()
        )

    # --------------------------------------------------------
    # INVALID ROW 1:
    # Flow Duration < 0
    # --------------------------------------------------------

    negative_duration = (
        df[FLOW_DURATION] < 0
    )

    n_negative_duration = int(
        negative_duration.sum()
    )

    stats.removed_negative_duration += (
        n_negative_duration
    )

    # --------------------------------------------------------
    # INVALID ROW 2:
    # Impossible header / min-segment values
    # --------------------------------------------------------

    invalid_headers = (
        (df[FWD_HEADER] < 0)
        | (df[BWD_HEADER] < 0)
        | (df[MIN_SEG] < 0)
    )

    # Avoid double-counting if a future dataset contains overlap.
    invalid_headers_only = (
        invalid_headers
        & ~negative_duration
    )

    n_invalid_headers = int(
        invalid_headers_only.sum()
    )

    stats.removed_invalid_headers += (
        n_invalid_headers
    )

    overlap = int(
        (negative_duration & invalid_headers).sum()
    )

    if overlap:
        log(
            f"  WARNING: {overlap} rows matched both "
            f"invalid-row rules. They are counted only once."
        )

    remove_mask = (
        negative_duration
        | invalid_headers
    )

    df = df.loc[
        ~remove_mask
    ].copy()

    log(
        f"  Removed negative-duration rows: "
        f"{n_negative_duration:,}"
    )

    log(
        f"  Removed invalid-header/min-segment rows: "
        f"{n_invalid_headers:,}"
    )

    # Preserve provenance.
    df["source_file"] = filename

    return df


# ============================================================
# EXACT FEATURE-VECTOR CONFLICT DETECTION
# ============================================================

def remove_conflicting_feature_vectors(
    df: pd.DataFrame,
    stats: CleaningStats,
) -> pd.DataFrame:

    """
    Detect exact 78-feature-vector groups whose labels disagree.

    IMPORTANT:

    A 64-bit hash is used ONLY as a candidate grouping mechanism.

    Hash groups containing multiple labels are subsequently verified
    using the actual feature values.

    This avoids treating a theoretical hash collision as a real
    feature-vector conflict.
    """

    feature_columns = stats.feature_columns

    log("\nBuilding feature-vector hashes...")

    feature_df = df[feature_columns]

    hashes = pd.util.hash_pandas_object(
        feature_df,
        index=False,
    )

    temp = pd.DataFrame(
        {
            "_hash": hashes.to_numpy(),
            "_label": df[LABEL_COL].to_numpy(),
        },
        index=df.index,
    )

    # --------------------------------------------------------
    # Find hashes associated with >1 label
    # --------------------------------------------------------

    hash_label_counts = (
        temp.groupby("_hash")["_label"]
        .nunique()
    )

    candidate_hashes = hash_label_counts[
        hash_label_counts > 1
    ].index

    if len(candidate_hashes) == 0:
        log("  No conflicting feature-vector candidates found.")
        return df

    log(
        f"  Candidate hash groups with multiple labels: "
        f"{len(candidate_hashes):,}"
    )

    candidate_mask = hashes.isin(
        candidate_hashes
    )

    candidates = df.loc[
        candidate_mask
    ].copy()

    candidate_hashes_series = hashes.loc[
        candidate_mask
    ]

    candidates["_feature_hash"] = (
        candidate_hashes_series.to_numpy()
    )

    # --------------------------------------------------------
    # Verify actual feature equality
    # --------------------------------------------------------

    log(
        f"  Verifying {len(candidates):,} candidate rows "
        f"against their actual feature values..."
    )

    conflicting_indices = []

    # We only inspect candidate hash groups.
    # For each candidate group, actual feature values are compared.
    for _, group in candidates.groupby(
        "_feature_hash",
        sort=False,
    ):

        if group[LABEL_COL].nunique() <= 1:
            continue

        # Group by the ACTUAL feature columns.
        # This is the important verification step.
        actual_groups = group.groupby(
            feature_columns,
            dropna=False,
            sort=False,
        )

        for _, exact_group in actual_groups:

            if exact_group[LABEL_COL].nunique() > 1:
                conflicting_indices.extend(
                    exact_group.index.tolist()
                )

    if not conflicting_indices:
        log(
            "  Hash candidates existed, but no actual "
            "conflicting feature vectors were verified."
        )
        return df

    conflicting_indices = pd.Index(
        conflicting_indices
    ).unique()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    removed = df.loc[
        conflicting_indices
    ]

    stats.conflicting_groups = 0

    # Count exact conflicting groups.
    exact_conflict_groups = removed.groupby(
        feature_columns,
        dropna=False,
        sort=False,
    )

    for _, group in exact_conflict_groups:

        if group[LABEL_COL].nunique() > 1:
            stats.conflicting_groups += 1

    stats.removed_conflict_rows = len(
        conflicting_indices
    )

    stats.conflict_removed_labels.update(
        removed[LABEL_COL].value_counts().to_dict()
    )

    # --------------------------------------------------------
    # Per-class impact
    # --------------------------------------------------------

    before_counts = (
        df[LABEL_COL]
        .value_counts()
    )

    for label, removed_count in (
        removed[LABEL_COL]
        .value_counts()
        .items()
    ):

        before = int(
            before_counts.get(label, 0)
        )

        fraction = (
            removed_count / before
            if before
            else 0.0
        )

        stats.conflict_class_impact.append(
            {
                "label": label,
                "rows_removed": int(removed_count),
                "rows_before_removal": before,
                "fraction_removed": fraction,
            }
        )

        if fraction >= CONFLICT_IMPACT_WARN_THRESHOLD:
            log(
                f"  !!! WARNING: {label} loses "
                f"{removed_count:,}/{before:,} rows "
                f"({fraction:.2%}) to conflict removal."
            )

    log(
        f"  Verified conflicting groups: "
        f"{stats.conflicting_groups:,}"
    )

    log(
        f"  Rows removed from conflicting groups: "
        f"{stats.removed_conflict_rows:,}"
    )

    return df.drop(
        index=conflicting_indices
    )


# ============================================================
# CELL-LEVEL CLEANING
# ============================================================

def clean_invalid_cells(
    df: pd.DataFrame,
    stats: CleaningStats,
) -> pd.DataFrame:

    log("\nApplying cell-level cleaning...")

    # --------------------------------------------------------
    # Flow IAT Min <= -1
    # --------------------------------------------------------

    mask = (
        df[FLOW_IAT_MIN] <= -1
    )

    count = int(mask.sum())

    stats.iat_nan_counts[
        FLOW_IAT_MIN
    ] += count

    df.loc[
        mask,
        FLOW_IAT_MIN
    ] = np.nan

    log(
        f"  {FLOW_IAT_MIN} <= -1 -> NaN: "
        f"{count:,}"
    )

    # --------------------------------------------------------
    # Fwd IAT Min <= -1
    # --------------------------------------------------------

    mask = (
        df[FWD_IAT_MIN] <= -1
    )

    count = int(mask.sum())

    stats.iat_nan_counts[
        FWD_IAT_MIN
    ] += count

    df.loc[
        mask,
        FWD_IAT_MIN
    ] = np.nan

    log(
        f"  {FWD_IAT_MIN} <= -1 -> NaN: "
        f"{count:,}"
    )

    # --------------------------------------------------------
    # Infinity -> NaN
    # --------------------------------------------------------

    total_inf = 0

    for col in stats.feature_columns:

        mask = np.isinf(
            df[col]
        )

        count = int(mask.sum())

        if count:
            stats.infinity_nan_counts[col] += count

            df.loc[
                mask,
                col
            ] = np.nan

            total_inf += count

    log(
        f"  +/- infinity -> NaN: "
        f"{total_inf:,}"
    )

    return df


# ============================================================
# EXACT DUPLICATE COLUMNS
# ============================================================

def remove_exact_duplicate_columns(
    df: pd.DataFrame,
    stats: CleaningStats,
) -> pd.DataFrame:

    if not REMOVE_EXACT_DUPLICATE_COLUMNS:
        log(
            "\nExact duplicate column removal disabled."
        )
        return df

    log(
        "\nVerifying exact duplicate columns..."
    )

    columns_to_drop = []

    for col_a, col_b in EXACT_DUPLICATE_PAIRS:

        if (
            col_a not in df.columns
            or col_b not in df.columns
        ):
            log(
                f"  WARNING: missing duplicate pair "
                f"{col_a} / {col_b}"
            )
            continue

        # pandas Series.equals handles NaN == NaN
        # for equality purposes.
        equal = df[col_a].equals(
            df[col_b]
        )

        if not equal:

            mismatch = (
                df[col_a] != df[col_b]
            )

            # NaN vs NaN should not count as mismatch.
            mismatch &= ~(
                df[col_a].isna()
                & df[col_b].isna()
            )

            n_mismatch = int(
                mismatch.sum()
            )

            log(
                f"  WARNING: {col_a} and {col_b} "
                f"are no longer identical "
                f"({n_mismatch:,} mismatches). "
                f"Keeping both."
            )

            continue

        # Keep the first column according to original order.
        order = list(df.columns)

        if order.index(col_a) < order.index(col_b):
            keep = col_a
            drop = col_b
        else:
            keep = col_b
            drop = col_a

        columns_to_drop.append(drop)

        stats.duplicate_columns_removed.append(
            {
                "kept": keep,
                "dropped": drop,
            }
        )

        log(
            f"  Verified: kept '{keep}', "
            f"dropped '{drop}'."
        )

    if columns_to_drop:
        df = df.drop(
            columns=columns_to_drop
        )

        stats.feature_columns = [
            c
            for c in stats.feature_columns
            if c not in columns_to_drop
        ]

    return df


# ============================================================
# CONSTANT ZERO FEATURES
# ============================================================

def inspect_constant_zero_columns(
    df: pd.DataFrame,
    stats: CleaningStats,
) -> None:

    log(
        "\nChecking 100%-zero feature columns..."
    )

    for col in stats.feature_columns:

        values = df[col].dropna()

        if (
            len(values) > 0
            and (values == 0).all()
        ):
            stats.constant_zero_columns.append(
                col
            )

    if stats.constant_zero_columns:

        log(
            f"  Found {len(stats.constant_zero_columns)} "
            f"100%-zero columns."
        )

        for col in stats.constant_zero_columns:
            log(f"    - {col}")

        log(
            "  These columns are retained for now."
        )

    else:
        log(
            "  No 100%-zero columns found."
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_final_dataset(
    df: pd.DataFrame,
    stats: CleaningStats,
) -> None:

    log(
        "\nRunning final validation..."
    )

    # --------------------------------------------------------
    # No negative Flow Duration
    # --------------------------------------------------------

    assert not (
        df[FLOW_DURATION] < 0
    ).any(), (
        "ERROR: negative Flow Duration remains."
    )

    # --------------------------------------------------------
    # No impossible header/min-segment values
    # --------------------------------------------------------

    assert not (
        df[FWD_HEADER] < 0
    ).any(), (
        "ERROR: negative Fwd Header Length remains."
    )

    assert not (
        df[BWD_HEADER] < 0
    ).any(), (
        "ERROR: negative Bwd Header Length remains."
    )

    assert not (
        df[MIN_SEG] < 0
    ).any(), (
        "ERROR: negative min_seg_size_forward remains."
    )

    # --------------------------------------------------------
    # No infinities
    # --------------------------------------------------------

    for col in stats.feature_columns:

        assert not np.isinf(
            df[col]
        ).any(), (
            f"ERROR: infinity remains in {col}"
        )

    # --------------------------------------------------------
    # Check required target
    # --------------------------------------------------------

    assert (
        LABEL_COL in df.columns
    ), "ERROR: Label column disappeared."

    assert (
        df[LABEL_COL].notna().all()
    ), "ERROR: missing labels found."

    # --------------------------------------------------------
    # Check row count
    # --------------------------------------------------------

    assert (
        len(df) == stats.final_rows
    )

    log(
        "  Final validation PASSED."
    )


# ============================================================
# REPORT
# ============================================================

def write_reports(
    stats: CleaningStats,
) -> None:

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CSV summary
    # --------------------------------------------------------

    metrics = [
        (
            "raw_rows",
            stats.raw_rows,
        ),
        (
            "removed_negative_duration",
            stats.removed_negative_duration,
        ),
        (
            "removed_invalid_headers",
            stats.removed_invalid_headers,
        ),
        (
            "conflicting_groups",
            stats.conflicting_groups,
        ),
        (
            "removed_conflict_rows",
            stats.removed_conflict_rows,
        ),
        (
            "final_rows",
            stats.final_rows,
        ),
        (
            "final_columns",
            stats.final_columns,
        ),
        (
            "numeric_conversion_failures",
            stats.numeric_conversion_failures,
        ),
        (
            "duplicate_columns_removed",
            len(stats.duplicate_columns_removed),
        ),
        (
            "constant_zero_columns",
            len(stats.constant_zero_columns),
        ),
    ]

    pd.DataFrame(
        metrics,
        columns=[
            "metric",
            "value",
        ],
    ).to_csv(
        REPORTS_DIR
        / "cleaning_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    lines = []

    lines.append(
        "# CICIDS2017 Cleaning Summary"
    )
    lines.append("")

    lines.append(
        "Generated by `src/clean.py`."
    )
    lines.append(
        "`data/raw/` was not modified."
    )
    lines.append("")

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    lines.append(
        "## Row Counts"
    )
    lines.append("")

    lines.append(
        f"- Raw rows: **{stats.raw_rows:,}**"
    )

    lines.append(
        f"- Removed negative Flow Duration: "
        f"**{stats.removed_negative_duration:,}**"
    )

    lines.append(
        f"- Removed impossible header/min-segment rows: "
        f"**{stats.removed_invalid_headers:,}**"
    )

    lines.append(
        f"- Conflicting feature-vector groups: "
        f"**{stats.conflicting_groups:,}**"
    )

    lines.append(
        f"- Rows removed from conflicting groups: "
        f"**{stats.removed_conflict_rows:,}**"
    )

    lines.append(
        f"- Final rows: **{stats.final_rows:,}**"
    )

    lines.append("")

    expected = (
        stats.raw_rows
        - stats.removed_negative_duration
        - stats.removed_invalid_headers
        - stats.removed_conflict_rows
    )

    lines.append(
        f"- Arithmetic check: "
        f"`{stats.raw_rows:,} - "
        f"{stats.removed_negative_duration:,} - "
        f"{stats.removed_invalid_headers:,} - "
        f"{stats.removed_conflict_rows:,} "
        f"= {expected:,}`"
    )

    lines.append("")

    # --------------------------------------------------------
    # Label normalization
    # --------------------------------------------------------

    lines.append(
        "## Label Normalization"
    )
    lines.append("")

    if stats.label_changes:

        lines.append(
            "| Raw label | Normalized label | Rows |"
        )
        lines.append(
            "|---|---|---:|"
        )

        for (
            (raw, normalized),
            count,
        ) in stats.label_changes.items():

            lines.append(
                f"| `{raw}` | `{normalized}` | "
                f"{count:,} |"
            )

    else:
        lines.append(
            "No labels were normalized."
        )

    lines.append("")

    # --------------------------------------------------------
    # Conflicts
    # --------------------------------------------------------

    lines.append(
        "## Conflicting Feature Vectors"
    )
    lines.append("")

    lines.append(
        f"- Verified conflicting groups: "
        f"**{stats.conflicting_groups:,}**"
    )

    lines.append(
        f"- Rows removed: "
        f"**{stats.removed_conflict_rows:,}**"
    )

    lines.append("")

    if stats.conflict_removed_labels:

        lines.append(
            "| Label | Rows removed |"
        )
        lines.append(
            "|---|---:|"
        )

        for label, count in sorted(
            stats.conflict_removed_labels.items(),
            key=lambda x: -x[1],
        ):

            lines.append(
                f"| {label} | {count:,} |"
            )

    lines.append("")

    # --------------------------------------------------------
    # Class impact
    # --------------------------------------------------------

    lines.append(
        "### Conflict Removal Class Impact"
    )
    lines.append("")

    if stats.conflict_class_impact:

        lines.append(
            "| Label | Removed | Before | Fraction |"
        )
        lines.append(
            "|---|---:|---:|---:|"
        )

        for item in sorted(
            stats.conflict_class_impact,
            key=lambda x: -x["fraction_removed"],
        ):

            lines.append(
                f"| {item['label']} | "
                f"{item['rows_removed']:,} | "
                f"{item['rows_before_removal']:,} | "
                f"{item['fraction_removed']:.2%} |"
            )

    lines.append("")

    # --------------------------------------------------------
    # IAT cleaning
    # --------------------------------------------------------

    lines.append(
        "## Sentinel / Invalid IAT Values"
    )
    lines.append("")

    for col, count in (
        stats.iat_nan_counts.items()
    ):

        lines.append(
            f"- `{col} <= -1 -> NaN`: "
            f"**{count:,}**"
        )

    lines.append("")

    # --------------------------------------------------------
    # Infinity
    # --------------------------------------------------------

    lines.append(
        "## Infinity Replacement"
    )
    lines.append("")

    if stats.infinity_nan_counts:

        for col, count in (
            stats.infinity_nan_counts.items()
        ):

            lines.append(
                f"- `{col}`: "
                f"**{count:,}** values -> NaN"
            )

    else:
        lines.append(
            "No infinities were found."
        )

    lines.append("")

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    lines.append(
        "## Exact Duplicate Columns"
    )
    lines.append("")

    if stats.duplicate_columns_removed:

        lines.append(
            "| Kept | Dropped |"
        )
        lines.append(
            "|---|---|"
        )

        for item in (
            stats.duplicate_columns_removed
        ):

            lines.append(
                f"| {item['kept']} | "
                f"{item['dropped']} |"
            )

    else:
        lines.append(
            "No exact duplicate columns were removed."
        )

    lines.append("")

    # --------------------------------------------------------
    # Constant zero columns
    # --------------------------------------------------------

    lines.append(
        "## 100%-Zero Features"
    )
    lines.append("")

    if stats.constant_zero_columns:

        lines.append(
            "These columns were detected but deliberately "
            "retained for later feature engineering:"
        )
        lines.append("")

        for col in (
            stats.constant_zero_columns
        ):
            lines.append(
                f"- `{col}`"
            )

    else:
        lines.append(
            "No 100%-zero features found."
        )

    lines.append("")

    # --------------------------------------------------------
    # Final labels
    # --------------------------------------------------------

    lines.append(
        "## Final Label Distribution"
    )
    lines.append("")

    lines.append(
        "| Label | Raw | Final |"
    )
    lines.append(
        "|---|---:|---:|"
    )

    all_labels = sorted(
        set(stats.raw_label_counts)
        | set(stats.final_label_counts),
        key=lambda x: -stats.raw_label_counts.get(
            x,
            0,
        ),
    )

    for label in all_labels:

        lines.append(
            f"| {label} | "
            f"{stats.raw_label_counts.get(label, 0):,} | "
            f"{stats.final_label_counts.get(label, 0):,} |"
        )

    lines.append("")

    report_path = (
        REPORTS_DIR
        / "cleaning_summary.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    log(
        f"Wrote report: {report_path}"
    )

    log(
        f"Wrote summary: "
        f"{REPORTS_DIR / 'cleaning_summary.csv'}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> int:

    log("=" * 70)
    log("CICIDS2017 CLEANING PIPELINE")
    log("=" * 70)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not RAW_DIR.exists():

        log(
            f"ERROR: raw directory does not exist:\n"
            f"{RAW_DIR}"
        )

        return 1

    missing = [
        filename
        for filename in RAW_FILES
        if not (
            RAW_DIR / filename
        ).exists()
    ]

    if missing:

        log(
            "ERROR: missing raw files:"
        )

        for filename in missing:
            log(f"  - {filename}")

        return 1

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = CleaningStats()

    parts = []

    # --------------------------------------------------------
    # Load and clean each raw file
    # --------------------------------------------------------

    log(
        "\n[1/7] Loading and cleaning raw files..."
    )

    try:

        for filename in RAW_FILES:

            df = process_file(
                filename,
                stats,
            )

            log(
                f"  Retained after physical-validity "
                f"cleaning: {len(df):,}"
            )

            parts.append(df)

    except Exception:

        log(
            "\nFATAL ERROR while processing raw data:"
        )

        log(
            traceback.format_exc()
        )

        return 1

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    log(
        "\n[2/7] Combining files..."
    )

    df = pd.concat(
        parts,
        ignore_index=True,
    )

    del parts

    log(
        f"  Rows before conflict removal: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Conflicts
    # --------------------------------------------------------

    log(
        "\n[3/7] Removing conflicting "
        "feature-vector groups..."
    )

    df = remove_conflicting_feature_vectors(
        df,
        stats,
    )

    log(
        f"  Rows after conflict removal: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Cell cleaning
    # --------------------------------------------------------

    log(
        "\n[4/7] Cleaning invalid/sentinel cells..."
    )

    df = clean_invalid_cells(
        df,
        stats,
    )

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    log(
        "\n[5/7] Handling exact duplicate columns..."
    )

    df = remove_exact_duplicate_columns(
        df,
        stats,
    )

    # --------------------------------------------------------
    # Constant features
    # --------------------------------------------------------

    log(
        "\n[6/7] Inspecting constant-zero features..."
    )

    inspect_constant_zero_columns(
        df,
        stats,
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    stats.final_rows = len(df)
    stats.final_columns = len(df.columns)

    stats.final_label_counts = Counter(
        df[LABEL_COL].value_counts().to_dict()
    )

    for col in stats.feature_columns:

        stats.missing_after[col] = int(
            df[col].isna().sum()
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    validate_final_dataset(
        df,
        stats,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    log(
        "\n[7/7] Writing cleaned dataset..."
    )

    output_path = (
        PROCESSED_DIR
        / "cleaned.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    log(
        f"  Saved: {output_path}"
    )

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    write_reports(
        stats
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    log("\n" + "=" * 70)
    log("CLEANING COMPLETE")
    log("=" * 70)

    log(
        f"Raw rows:             {stats.raw_rows:,}"
    )

    log(
        f"Invalid duration:     "
        f"{stats.removed_negative_duration:,}"
    )

    log(
        f"Invalid headers:      "
        f"{stats.removed_invalid_headers:,}"
    )

    log(
        f"Conflict groups:      "
        f"{stats.conflicting_groups:,}"
    )

    log(
        f"Conflict rows:        "
        f"{stats.removed_conflict_rows:,}"
    )

    log(
        f"Final rows:           "
        f"{stats.final_rows:,}"
    )

    log(
        f"Final columns:        "
        f"{stats.final_columns:,}"
    )

    log(
        "Raw data was NOT modified."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())