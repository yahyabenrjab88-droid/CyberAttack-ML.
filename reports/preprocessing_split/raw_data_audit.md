# CICIDS2017 Raw Data Audit

## Dataset overview

- Files audited: 8
- Total rows: 2,830,743
- Total columns: 79
- Feature columns: 78
- Target column: `Label`
- Chunk size: 100,000

## Rows per file

| File | Rows |
|---|---:|
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 225,745 |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 286,467 |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 191,033 |
| `Monday-WorkingHours.pcap_ISCX.csv` | 529,918 |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 288,602 |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 170,366 |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 445,909 |
| `Wednesday-workingHours.pcap_ISCX.csv` | 692,703 |

## Label distribution

| Label | Count | Percentage |
|---|---:|---:|
| `BENIGN` | 2,273,097 | 80.3004% |
| `DoS Hulk` | 231,073 | 8.1630% |
| `PortScan` | 158,930 | 5.6144% |
| `DDoS` | 128,027 | 4.5227% |
| `DoS GoldenEye` | 10,293 | 0.3636% |
| `FTP-Patator` | 7,938 | 0.2804% |
| `SSH-Patator` | 5,897 | 0.2083% |
| `DoS slowloris` | 5,796 | 0.2048% |
| `DoS Slowhttptest` | 5,499 | 0.1943% |
| `Bot` | 1,966 | 0.0695% |
| `Web Attack � Brute Force` | 1,507 | 0.0532% |
| `Web Attack � XSS` | 652 | 0.0230% |
| `Infiltration` | 36 | 0.0013% |
| `Web Attack � Sql Injection` | 21 | 0.0007% |
| `Heartbleed` | 11 | 0.0004% |

## Missing values

Total missing cells: **1,358**

| Column | Missing |
|---|---:|
| `Flow Bytes/s` | 1,358 |

## Numeric conversion failures

Total conversion failures: **0**

## Infinity values

- Positive infinity: **4,376**
- Negative infinity: **0**

## Negative values

Total negative numeric values: **2,446,321**

| Column | Negative values |
|---|---:|
| `Init_Win_bytes_backward` | 1,441,552 |
| `Init_Win_bytes_forward` | 1,001,189 |
| `Flow IAT Min` | 2,891 |
| `Flow Duration` | 115 |
| `Flow Packets/s` | 115 |
| `Flow IAT Mean` | 115 |
| `Flow IAT Max` | 115 |
| `Flow Bytes/s` | 85 |
| `Fwd Header Length` | 35 |
| `Fwd Header Length.1` | 35 |
| `min_seg_size_forward` | 35 |
| `Bwd Header Length` | 22 |
| `Fwd IAT Min` | 17 |

## Constant features

- `Bwd PSH Flags`
- `Bwd URG Flags`
- `Fwd Avg Bytes/Bulk`
- `Fwd Avg Packets/Bulk`
- `Fwd Avg Bulk Rate`
- `Bwd Avg Bytes/Bulk`
- `Bwd Avg Packets/Bulk`
- `Bwd Avg Bulk Rate`

## Duplicate analysis

- Exact duplicate complete rows: **308,381**
- Duplicate feature-vector groups: **95,479**
- Duplicate feature-vector rows beyond first occurrence: **309,079**
- Conflicting-label feature-vector groups: **698**
- Rows belonging to conflicting-label feature-vector groups: **7,020**

## Feature columns

Total feature columns: **78**

- `Destination Port`
- `Flow Duration`
- `Total Fwd Packets`
- `Total Backward Packets`
- `Total Length of Fwd Packets`
- `Total Length of Bwd Packets`
- `Fwd Packet Length Max`
- `Fwd Packet Length Min`
- `Fwd Packet Length Mean`
- `Fwd Packet Length Std`
- `Bwd Packet Length Max`
- `Bwd Packet Length Min`
- `Bwd Packet Length Mean`
- `Bwd Packet Length Std`
- `Flow Bytes/s`
- `Flow Packets/s`
- `Flow IAT Mean`
- `Flow IAT Std`
- `Flow IAT Max`
- `Flow IAT Min`
- `Fwd IAT Total`
- `Fwd IAT Mean`
- `Fwd IAT Std`
- `Fwd IAT Max`
- `Fwd IAT Min`
- `Bwd IAT Total`
- `Bwd IAT Mean`
- `Bwd IAT Std`
- `Bwd IAT Max`
- `Bwd IAT Min`
- `Fwd PSH Flags`
- `Bwd PSH Flags`
- `Fwd URG Flags`
- `Bwd URG Flags`
- `Fwd Header Length`
- `Bwd Header Length`
- `Fwd Packets/s`
- `Bwd Packets/s`
- `Min Packet Length`
- `Max Packet Length`
- `Packet Length Mean`
- `Packet Length Std`
- `Packet Length Variance`
- `FIN Flag Count`
- `SYN Flag Count`
- `RST Flag Count`
- `PSH Flag Count`
- `ACK Flag Count`
- `URG Flag Count`
- `CWE Flag Count`
- `ECE Flag Count`
- `Down/Up Ratio`
- `Average Packet Size`
- `Avg Fwd Segment Size`
- `Avg Bwd Segment Size`
- `Fwd Header Length.1`
- `Fwd Avg Bytes/Bulk`
- `Fwd Avg Packets/Bulk`
- `Fwd Avg Bulk Rate`
- `Bwd Avg Bytes/Bulk`
- `Bwd Avg Packets/Bulk`
- `Bwd Avg Bulk Rate`
- `Subflow Fwd Packets`
- `Subflow Fwd Bytes`
- `Subflow Bwd Packets`
- `Subflow Bwd Bytes`
- `Init_Win_bytes_forward`
- `Init_Win_bytes_backward`
- `act_data_pkt_fwd`
- `min_seg_size_forward`
- `Active Mean`
- `Active Std`
- `Active Max`
- `Active Min`
- `Idle Mean`
- `Idle Std`
- `Idle Max`
- `Idle Min`

## Methodology notes

- Raw CSV files were read only; no raw file was modified.
- CSVs were processed in chunks to reduce memory usage.
- Column names were stripped in memory only.
- Numeric conversion was used for auditing only.
- No rows were dropped.
- No labels were changed.
- No scaling or normalization was performed.
- No train/validation/test split was performed.
- Duplicate detection uses pandas 64-bit row hashes.
- Conflicting-label feature vectors are reported for later cleaning decisions; they are not removed by this audit.
