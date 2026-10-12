
from pathlib import Path
import pandas as pd
import numpy as np


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
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
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
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]


# ============================================================
# HELPERS
# ============================================================

def read_csv_chunks(file_path):
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


def add_result(
    results,
    filename,
    row_number,
    label,
    check_name,
    feature,
    value,
    severity,
    reason,
):
    results.append(
        {
            "source_file": filename,
            "csv_row": row_number,
            "Label": label,
            "check": check_name,
            "feature": feature,
            "value": value,
            "severity": severity,
            "reason": reason,
        }
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("FEATURE-SPECIFIC VALIDITY AUDIT")
    print("=" * 80)

    print(f"Raw directory : {RAW_DIR}")
    print(f"Chunk size    : {CHUNK_SIZE}")
    print()

    # --------------------------------------------------------
    # Summary counters
    # --------------------------------------------------------

    summary = {}

    def increment(check_name, severity, count):
        key = (check_name, severity)
        summary[key] = summary.get(key, 0) + int(count)

    suspicious_rows = []

    total_rows = 0

    # ========================================================
    # PROCESS FILES
    # ========================================================

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

        for chunk in read_csv_chunks(file_path):

            chunk.columns = normalize_columns(chunk.columns)

            required = set(FEATURE_COLUMNS + ["Label"])
            missing = required - set(chunk.columns)

            if missing:
                raise KeyError(
                    f"Missing columns in {filename}: "
                    f"{sorted(missing)}"
                )

            total_rows += len(chunk)

            # ------------------------------------------------
            # Convert features to numeric
            # ------------------------------------------------

            numeric = {}

            for feature in FEATURE_COLUMNS:
                numeric[feature] = pd.to_numeric(
                    chunk[feature],
                    errors="coerce",
                )

            # ------------------------------------------------
            # Helper to record vectorized results
            # ------------------------------------------------

            def process_mask(
                mask,
                check_name,
                feature,
                severity,
                reason,
            ):

                count = int(mask.sum())

                if count == 0:
                    return

                increment(
                    check_name,
                    severity,
                    count,
                )

                # Keep examples only.
                # We do not save millions of rows.
                indices = chunk.index[mask]

                # Maximum 10 examples per check per chunk.
                for idx in indices[:10]:

                    add_result(
                        suspicious_rows,
                        filename,
                        int(idx) + 2,
                        chunk.loc[idx, "Label"],
                        check_name,
                        feature,
                        chunk.loc[idx, feature],
                        severity,
                        reason,
                    )

            # =================================================
            # 1. DESTINATION PORT
            # =================================================

            port = numeric["Destination Port"]

            process_mask(
                port < 0,
                "negative_destination_port",
                "Destination Port",
                "DEFINITE_INVALID",
                "Network port numbers cannot be negative.",
            )

            process_mask(
                port > 65535,
                "destination_port_above_65535",
                "Destination Port",
                "DEFINITE_INVALID",
                "TCP/UDP port numbers are limited to 0-65535.",
            )

            # =================================================
            # 2. PACKET COUNTS
            # =================================================

            packet_count_features = [
                "Total Fwd Packets",
                "Total Backward Packets",
                "Subflow Fwd Packets",
                "Subflow Bwd Packets",
                "act_data_pkt_fwd",
            ]

            for feature in packet_count_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Packet counts cannot be negative.",
                )

            # =================================================
            # 3. BYTE COUNTS
            # =================================================

            byte_count_features = [
                "Total Length of Fwd Packets",
                "Total Length of Bwd Packets",
                "Subflow Fwd Bytes",
                "Subflow Bwd Bytes",
            ]

            for feature in byte_count_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Byte counts cannot be negative.",
                )

            # =================================================
            # 4. PACKET LENGTHS
            # =================================================

            packet_length_features = [
                "Fwd Packet Length Max",
                "Fwd Packet Length Min",
                "Fwd Packet Length Mean",
                "Fwd Packet Length Std",
                "Bwd Packet Length Max",
                "Bwd Packet Length Min",
                "Bwd Packet Length Mean",
                "Bwd Packet Length Std",
                "Min Packet Length",
                "Max Packet Length",
                "Packet Length Mean",
                "Packet Length Std",
                "Packet Length Variance",
                "Average Packet Size",
                "Avg Fwd Segment Size",
                "Avg Bwd Segment Size",
            ]

            for feature in packet_length_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Packet lengths/statistics cannot be negative.",
                )

            # =================================================
            # 5. FLAG COUNTS
            # =================================================

            flag_features = [
                "Fwd PSH Flags",
                "Bwd PSH Flags",
                "Fwd URG Flags",
                "Bwd URG Flags",
                "FIN Flag Count",
                "SYN Flag Count",
                "RST Flag Count",
                "PSH Flag Count",
                "ACK Flag Count",
                "URG Flag Count",
                "CWE Flag Count",
                "ECE Flag Count",
            ]

            for feature in flag_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "TCP flag counts cannot be negative.",
                )

            # =================================================
            # 6. TIME / IAT FEATURES
            # =================================================

            time_features = [
                "Flow Duration",
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
                "Active Mean",
                "Active Std",
                "Active Max",
                "Active Min",
                "Idle Mean",
                "Idle Std",
                "Idle Max",
                "Idle Min",
            ]

            for feature in time_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "NEEDS_INVESTIGATION",
                    "Time-related features are normally non-negative; "
                    "negative values may be sentinel values or corruption.",
                )

            # =================================================
            # 7. HEADER LENGTHS
            # =================================================

            header_features = [
                "Fwd Header Length",
                "Bwd Header Length",
                "Fwd Header Length.1",
                "min_seg_size_forward",
            ]

            for feature in header_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Header/segment length cannot be negative.",
                )

            # =================================================
            # 8. INITIAL TCP WINDOW VALUES
            # =================================================

            # IMPORTANT:
            # -1 is deliberately NOT classified as invalid.
            # CICIDS commonly contains -1 here as a sentinel.
            #
            # We only check values below -1.

            for feature in [
                "Init_Win_bytes_forward",
                "Init_Win_bytes_backward",
            ]:

                values = numeric[feature]

                process_mask(
                    values < -1,
                    f"{feature}_below_minus_one",
                    feature,
                    "DEFINITE_INVALID",
                    "Initial TCP window values below -1 are outside "
                    "the observed sentinel/value range.",
                )

                process_mask(
                    values > 65535,
                    f"{feature}_above_65535",
                    feature,
                    "DEFINITE_INVALID",
                    "TCP window values should not exceed 65535 "
                    "in this dataset representation.",
                )

            # =================================================
            # 9. RATE FEATURES
            # =================================================

            rate_features = [
                "Flow Bytes/s",
                "Flow Packets/s",
                "Fwd Packets/s",
                "Bwd Packets/s",
                "Fwd Avg Bulk Rate",
                "Bwd Avg Bulk Rate",
            ]

            for feature in rate_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "NEEDS_INVESTIGATION",
                    "Rates should normally be non-negative.",
                )

                process_mask(
                    np.isposinf(values),
                    f"positive_infinity_{feature}",
                    feature,
                    "NEEDS_HANDLING",
                    "Positive infinity is not a finite numeric value "
                    "and must be handled before ML training.",
                )

                process_mask(
                    np.isneginf(values),
                    f"negative_infinity_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Negative infinity is not a valid finite feature value.",
                )

            # =================================================
            # 10. DOWN/UP RATIO
            # =================================================

            ratio = numeric["Down/Up Ratio"]

            process_mask(
                ratio < 0,
                "negative_down_up_ratio",
                "Down/Up Ratio",
                "DEFINITE_INVALID",
                "A ratio of non-negative packet counts cannot be negative.",
            )

            # =================================================
            # 11. MATHEMATICAL RELATIONSHIPS
            # =================================================

            # Max >= Min for packet lengths.

            fwd_max = numeric["Fwd Packet Length Max"]
            fwd_min = numeric["Fwd Packet Length Min"]

            process_mask(
                fwd_min > fwd_max,
                "fwd_packet_min_above_max",
                "Fwd Packet Length Min",
                "DEFINITE_INVALID",
                "Minimum packet length cannot exceed maximum packet length.",
            )

            bwd_max = numeric["Bwd Packet Length Max"]
            bwd_min = numeric["Bwd Packet Length Min"]

            process_mask(
                bwd_min > bwd_max,
                "bwd_packet_min_above_max",
                "Bwd Packet Length Min",
                "DEFINITE_INVALID",
                "Minimum packet length cannot exceed maximum packet length.",
            )

            # Overall packet minimum/maximum.

            overall_min = numeric["Min Packet Length"]
            overall_max = numeric["Max Packet Length"]

            process_mask(
                overall_min > overall_max,
                "packet_min_above_max",
                "Min Packet Length",
                "DEFINITE_INVALID",
                "Overall minimum packet length cannot exceed maximum.",
            )

            # =================================================
            # 12. STANDARD DEVIATION / VARIANCE
            # =================================================

            std_features = [
                "Fwd Packet Length Std",
                "Bwd Packet Length Std",
                "Packet Length Std",
                "Flow IAT Std",
                "Fwd IAT Std",
                "Bwd IAT Std",
                "Active Std",
                "Idle Std",
            ]

            for feature in std_features:

                values = numeric[feature]

                process_mask(
                    values < 0,
                    f"negative_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Standard deviation cannot be negative.",
                )

            process_mask(
                numeric["Packet Length Variance"] < 0,
                "negative_packet_length_variance",
                "Packet Length Variance",
                "DEFINITE_INVALID",
                "Variance cannot be negative.",
            )

            # =================================================
            # 13. INFINITE VALUES OUTSIDE RATE FEATURES
            # =================================================

            for feature in FEATURE_COLUMNS:

                values = numeric[feature]

                process_mask(
                    np.isneginf(values),
                    f"negative_infinity_{feature}",
                    feature,
                    "DEFINITE_INVALID",
                    "Negative infinity is not a valid finite feature value.",
                )

            # =================================================
            # 14. MISSING VALUES
            # =================================================

            for feature in FEATURE_COLUMNS:

                missing = chunk[feature].isna()

                process_mask(
                    missing,
                    f"missing_{feature}",
                    feature,
                    "NEEDS_HANDLING",
                    "Feature contains a missing value.",
                )

    # ========================================================
    # SAVE EXAMPLES
    # ========================================================

    print()
    print("=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    print(f"Total rows processed: {total_rows:,}")
    print()

    summary_rows = []

    for (check_name, severity), count in sorted(summary.items()):

        summary_rows.append(
            {
                "check": check_name,
                "severity": severity,
                "count": count,
            }
        )

        print(
            f"{severity:20s} "
            f"{check_name:45s} "
            f"{count:,}"
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = (
        REPORTS_DIR /
        "feature_validity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # SAVE EXAMPLES
    # ========================================================

    examples_df = pd.DataFrame(
        suspicious_rows
    )

    examples_path = (
        REPORTS_DIR /
        "feature_validity_examples.csv"
    )

    examples_df.to_csv(
        examples_path,
        index=False,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL COUNTS
    # ========================================================

    definite_invalid = sum(
        count
        for (check, severity), count in summary.items()
        if severity == "DEFINITE_INVALID"
    )

    needs_investigation = sum(
        count
        for (check, severity), count in summary.items()
        if severity == "NEEDS_INVESTIGATION"
    )

    needs_handling = sum(
        count
        for (check, severity), count in summary.items()
        if severity == "NEEDS_HANDLING"
    )

    print()
    print("=" * 80)
    print("FINAL TOTALS")
    print("=" * 80)

    print(
        f"DEFINITE_INVALID     : "
        f"{definite_invalid:,}"
    )

    print(
        f"NEEDS_INVESTIGATION  : "
        f"{needs_investigation:,}"
    )

    print(
        f"NEEDS_HANDLING       : "
        f"{needs_handling:,}"
    )

    print()
    print("Reports:")
    print(summary_path)
    print(examples_path)

    print()
    print("RAW DATA WAS NOT MODIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
