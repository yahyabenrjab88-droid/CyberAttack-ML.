# CICIDS2017 Split Distribution Audit
## Dataset sizes
- Train: 1,976,499 rows
- Validation: 423,537 rows
- Test: 423,537 rows
- Total: 2,823,573 rows

## Class distribution
| Label | Train rows | Train % | Validation rows | Validation % | Test rows | Test % |
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

## Source-file distribution
| Source file | Train % | Validation % | Test % |
|---|---:|---:|---:|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv | 8.059% | 7.874% | 7.815% |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv | 10.096% | 10.178% | 10.087% |
| Friday-WorkingHours-Morning.pcap_ISCX.csv | 6.759% | 6.887% | 6.672% |
| Monday-WorkingHours.pcap_ISCX.csv | 19.488% | 16.670% | 17.495% |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv | 9.199% | 12.737% | 12.330% |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv | 6.032% | 6.102% | 5.970% |
| Tuesday-WorkingHours.pcap_ISCX.csv | 16.045% | 14.996% | 15.397% |
| Wednesday-workingHours.pcap_ISCX.csv | 24.324% | 24.556% | 24.236% |

## Notes

- The primary split is group-aware to prevent identical feature vectors from crossing train/validation/test.
- Class stratification was performed at the group level.
- Therefore, row-level class percentages are not expected to be perfectly identical across splits.
- `source_file` is provenance information and is not a model feature.
- This audit does not modify any dataset.
