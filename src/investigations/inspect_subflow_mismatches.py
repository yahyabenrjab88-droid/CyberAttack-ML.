
from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RAW FILES
# ============================================================

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


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# COLUMNS WE WANT TO INSPECT
# ============================================================

INSPECT_COLUMNS = [
    "Destination Port",
    "Flow Duration",

    "Total Fwd Packets",
    "Subflow Fwd Packets",

    "Total Backward Packets",
    "Subflow Bwd Packets",

    "Total Length of Fwd Packets",
    "Subflow Fwd Bytes",

    "Total Length of Bwd Packets",
    "Subflow Bwd Bytes",

    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",

    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",

    "Flow Bytes/s",
    "Flow Packets/s",

    "Fwd Packets/s",
    "Bwd Packets/s",

    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",

    "Fwd Header Length",
    "Fwd Header Length.1",

    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",

    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",

    "Label",
]


# ============================================================
# HELPERS
# ============================================================

def read_csv_chunks(file_path):
    """
    Read CSV in chunks without modifying the source file.
    """

    try:
        return pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="utf-8",
        )

    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=True,
            na_filter=True,
            encoding="latin1",
        )


def normalize_columns(columns):
    return [str(col).strip() for col in columns]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SUBFLOW MISMATCH INSPECTION")
    print("=" * 80)

    print(f"Raw directory : {RAW_DIR}")
    print(f"Chunk size    : {CHUNK_SIZE}")
    print()

    all_mismatches = []

    total_forward_mismatches = 0
    total_backward_mismatches = 0

    # --------------------------------------------------------
    # Process every raw file
    # --------------------------------------------------------

    for file_index, filename in enumerate(RAW_FILES, start=1):

        file_path = RAW_DIR / filename

        print(
            f"[{file_index}/{len(RAW_FILES)}] "
            f"Processing: {filename}"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Raw file not found:\n{file_path}"
            )

        chunks = read_csv_chunks(file_path)

        file_forward_mismatches = 0
        file_backward_mismatches = 0

        # ----------------------------------------------------
        # Process chunks
        # ----------------------------------------------------

        for chunk in chunks:

            chunk.columns = normalize_columns(chunk.columns)

            # Make sure required columns exist.
            required_columns = set(INSPECT_COLUMNS)

            missing_columns = (
                required_columns - set(chunk.columns)
            )

            if missing_columns:
                raise KeyError(
                    f"Missing columns in {filename}: "
                    f"{sorted(missing_columns)}"
                )

            # ------------------------------------------------
            # Convert only the columns needed for comparison
            # ------------------------------------------------

            fwd_total = pd.to_numeric(
                chunk["Total Length of Fwd Packets"],
                errors="coerce",
            )

            fwd_subflow = pd.to_numeric(
                chunk["Subflow Fwd Bytes"],
                errors="coerce",
            )

            bwd_total = pd.to_numeric(
                chunk["Total Length of Bwd Packets"],
                errors="coerce",
            )

            bwd_subflow = pd.to_numeric(
                chunk["Subflow Bwd Bytes"],
                errors="coerce",
            )

            # ------------------------------------------------
            # Forward mismatch
            # ------------------------------------------------

            fwd_mismatch = (
                fwd_total != fwd_subflow
            )

            # ------------------------------------------------
            # Backward mismatch
            # ------------------------------------------------

            bwd_mismatch = (
                bwd_total != bwd_subflow
            )

            # ------------------------------------------------
            # Collect mismatch rows
            # ------------------------------------------------

            mismatch_mask = (
                fwd_mismatch |
                bwd_mismatch
            )

            if mismatch_mask.any():

                mismatch_chunk = chunk.loc[
                    mismatch_mask,
                    INSPECT_COLUMNS
                ].copy()

                # Add source information.
                mismatch_chunk.insert(
                    0,
                    "source_file",
                    filename,
                )

                # Preserve the original CSV row number.
                mismatch_chunk.insert(
                    1,
                    "csv_row",
                    mismatch_chunk.index + 2,
                )

                # Add numeric comparison columns.
                mismatch_chunk["fwd_difference"] = (
                    fwd_total.loc[mismatch_mask].values
                    -
                    fwd_subflow.loc[mismatch_mask].values
                )

                mismatch_chunk["bwd_difference"] = (
                    bwd_total.loc[mismatch_mask].values
                    -
                    bwd_subflow.loc[mismatch_mask].values
                )

                # Add a simple mismatch classification.
                def classify(row):
                    fwd = (
                        row["fwd_difference"] != 0
                        and pd.notna(row["fwd_difference"])
                    )

                    bwd = (
                        row["bwd_difference"] != 0
                        and pd.notna(row["bwd_difference"])
                    )

                    if fwd and bwd:
                        return "BOTH"

                    if fwd:
                        return "FWD_ONLY"

                    if bwd:
                        return "BWD_ONLY"

                    return "UNKNOWN"

                mismatch_chunk["mismatch_type"] = (
                    mismatch_chunk.apply(
                        classify,
                        axis=1
                    )
                )

                all_mismatches.append(
                    mismatch_chunk
                )

                file_forward_mismatches += int(
                    fwd_mismatch.sum()
                )

                file_backward_mismatches += int(
                    bwd_mismatch.sum()
                )

        total_forward_mismatches += file_forward_mismatches
        total_backward_mismatches += file_backward_mismatches

        print(
            f"    Forward mismatches : "
            f"{file_forward_mismatches:,}"
        )

        print(
            f"    Backward mismatches: "
            f"{file_backward_mismatches:,}"
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("MISMATCH SUMMARY")
    print("=" * 80)

    print(
        f"Forward-byte mismatches : "
        f"{total_forward_mismatches:,}"
    )

    print(
        f"Backward-byte mismatches: "
        f"{total_backward_mismatches:,}"
    )

    if not all_mismatches:
        print()
        print("No mismatches found.")
        return

    result = pd.concat(
        all_mismatches,
        ignore_index=True
    )

    # ========================================================
    # SAVE COMPLETE MISMATCH REPORT
    # ========================================================

    output_path = (
        REPORTS_DIR /
        "subflow_mismatches.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # SUMMARY BY LABEL
    # ========================================================

    label_summary = (
        result
        .groupby(
            ["mismatch_type", "Label"],
            dropna=False
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            "rows",
            ascending=False
        )
    )

    label_output = (
        REPORTS_DIR /
        "subflow_mismatch_labels.csv"
    )

    label_summary.to_csv(
        label_output,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # SUMMARY BY SOURCE FILE
    # ========================================================

    file_summary = (
        result
        .groupby(
            ["source_file", "mismatch_type"],
            dropna=False
        )
        .size()
        .reset_index(
            name="rows"
        )
        .sort_values(
            "rows",
            ascending=False
        )
    )

    file_output = (
        REPORTS_DIR /
        "subflow_mismatch_files.csv"
    )

    file_summary.to_csv(
        file_output,
        index=False,
        encoding="utf-8"
    )

    # ========================================================
    # DIFFERENCE STATISTICS
    # ========================================================

    print()
    print("Difference statistics:")

    print()
    print("Forward difference:")
    print(
        result["fwd_difference"]
        .dropna()
        .describe()
        .to_string()
    )

    print()
    print("Backward difference:")
    print(
        result["bwd_difference"]
        .dropna()
        .describe()
        .to_string()
    )

    # ========================================================
    # LABEL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("MISMATCHES BY LABEL")
    print("=" * 80)

    print(
        label_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # FILE SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print("MISMATCHES BY FILE")
    print("=" * 80)

    print(
        file_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # LARGEST DIFFERENCES
    # ========================================================

    print()
    print("=" * 80)
    print("LARGEST ABSOLUTE DIFFERENCES")
    print("=" * 80)

    result["abs_fwd_difference"] = (
        result["fwd_difference"]
        .abs()
    )

    result["abs_bwd_difference"] = (
        result["bwd_difference"]
        .abs()
    )

    largest_fwd = (
        result
        .sort_values(
            "abs_fwd_difference",
            ascending=False
        )
        .head(10)
    )

    print()
    print("Top 10 forward differences:")

    print(
        largest_fwd[
            [
                "source_file",
                "csv_row",
                "Total Length of Fwd Packets",
                "Subflow Fwd Bytes",
                "fwd_difference",
                "Label",
            ]
        ].to_string(index=False)
    )

    largest_bwd = (
        result
        .sort_values(
            "abs_bwd_difference",
            ascending=False
        )
        .head(10)
    )

    print()
    print("Top 10 backward differences:")

    print(
        largest_bwd[
            [
                "source_file",
                "csv_row",
                "Total Length of Bwd Packets",
                "Subflow Bwd Bytes",
                "bwd_difference",
                "Label",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print(f"Complete report : {output_path}")
    print(f"Label report    : {label_output}")
    print(f"File report     : {file_output}")

    print()
    print("RAW DATA WAS NOT MODIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
