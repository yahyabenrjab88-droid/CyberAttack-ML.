# CICIDS2017 Size-Aware Group-Stratified Split

## Split sizes

| Split | Rows | Percentage |
|---|---:|---:|
| Train | 1,976,499 | 70.000% |
| Validation | 423,537 | 15.000% |
| Test | 423,537 | 15.000% |
| Total | 2,823,573 | 100.000% |

## Class distribution

| Label | Train | Train % | Validation | Val % | Test | Test % |
|---|---:|---:|---:|---:|---:|---:|
| BENIGN | 1,590,547 | 80.473% | 340,832 | 80.473% | 340,832 | 80.473% |
| Bot | 1,376 | 0.070% | 295 | 0.070% | 295 | 0.070% |
| DDoS | 89,615 | 4.534% | 19,203 | 4.534% | 19,203 | 4.534% |
| DoS GoldenEye | 7,205 | 0.365% | 1,544 | 0.365% | 1,544 | 0.365% |
| DoS Hulk | 158,095 | 7.999% | 33,877 | 7.999% | 33,877 | 7.999% |
| DoS Slowhttptest | 3,849 | 0.195% | 825 | 0.195% | 825 | 0.195% |
| DoS slowloris | 4,057 | 0.205% | 869 | 0.205% | 869 | 0.205% |
| FTP-Patator | 5,556 | 0.281% | 1,191 | 0.281% | 1,191 | 0.281% |
| Heartbleed | 7 | 0.000% | 2 | 0.000% | 2 | 0.000% |
| Infiltration | 26 | 0.001% | 5 | 0.001% | 5 | 0.001% |
| PortScan | 110,513 | 5.591% | 23,682 | 5.591% | 23,682 | 5.591% |
| SSH-Patator | 4,127 | 0.209% | 885 | 0.209% | 885 | 0.209% |
| Web Attack - Brute Force | 1,055 | 0.053% | 226 | 0.053% | 226 | 0.053% |
| Web Attack - Sql Injection | 15 | 0.001% | 3 | 0.001% | 3 | 0.001% |
| Web Attack - XSS | 456 | 0.023% | 98 | 0.023% | 98 | 0.023% |

## Method

- Exact identical feature vectors were grouped together.
- Groups were never divided between splits.
- Groups were assigned separately within each class.
- The target row distribution is approximately 70% train / 15% validation / 15% test.
- `source_file` was excluded from grouping and model features.
- NaN values were not imputed during splitting.
- Random state: 42.
