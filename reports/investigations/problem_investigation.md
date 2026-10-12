# CICIDS2017 Problem Investigation

This report is READ-ONLY. Raw CSV files were not modified.

## 1. Dataset size

- Total rows: 2,830,743

## 2. Missing / Infinity

- Missing Flow Bytes/s: 1,358
- +Infinity Flow Bytes/s: 1,509
- +Infinity Flow Packets/s: 2,867

## 3. Negative duration

- Negative Flow Duration rows: 115

## 4. Negative IAT values

- Flow IAT Mean: 115
- Flow IAT Max: 115
- Flow IAT Min: 2,891
- Fwd IAT Min: 17

## 5. Negative header values

- Fwd Header Length: 35
- Bwd Header Length: 22
- Fwd Header Length.1: 35
- min_seg_size_forward: 35

## 6. Negative initial-window values

- Init_Win_bytes_forward: 1,001,189
- Init_Win_bytes_backward: 1,441,552

## 7. Conflicting feature-vector groups

- Conflicting groups: 698
- Rows belonging to conflicting groups: 7,020

### Label conflict pairs

- BENIGN <-> PortScan: 564 groups
- BENIGN <-> DoS Hulk: 130 groups
- BENIGN <-> DDoS: 3 groups
- BENIGN <-> DoS slowloris: 1 groups

## 8. Duplicate feature vectors

- Duplicate feature-vector groups: 95,479
- Duplicate rows beyond first occurrence: 309,079

## 9. Exact complete-row duplicates

- Exact duplicate groups: 95,169
- Exact duplicate rows beyond first occurrence: 308,381

## 10. Examples

### Negative duration examples
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', 'Label': 'BENIGN', 'Flow Duration': '-1'}

### Negative IAT examples
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Flow IAT Min', 'value': '-1'}

### Negative header examples
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-1929349973'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885093'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-1006620072'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770470'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885125'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885113'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885081'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885125'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770458'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885137'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770490'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885133'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770490'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770400'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885125'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885221'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770490'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770494'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-83885011'}
{'file': 'Monday-WorkingHours.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Fwd Header Length', 'value': '-167770494'}

### Negative initial-window examples
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
{'file': 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', 'Label': 'BENIGN', 'column': 'Init_Win_bytes_forward', 'value': '-1'}
