# CIC-IDS2017 Comprehensive Dataset Audit Report

> **Stage:** STAGE 1 - Dataset Ingestion and Quality Audit  
> **Dataset Target:** CIC-IDS2017 (`MachineLearningCVE`)  
> **Status:** Complete (Empirical Audit — No Synthetic Data, No Preprocessing, No Training)

---

## 1. Dataset Overview

The Canadian Institute for Cybersecurity Intrusion Detection System 2017 (**CIC-IDS2017**) benchmark dataset contains realistic network traffic flows generated over a 5-day evaluation period. Traffic is categorized into benign behavior and common attack profiles (DoS/DDoS, PortScan, Brute Force, Web Attacks, Infiltration, Botnet).

- **Total CSV Files Discovered:** 8
- **Total Network Flow Records:** 2,830,743
- **Total Features per Record:** 79
- **Target Label Feature:** ` Label`
- **Total Duplicate Records Identified:** 308,381 (10.894%)
- **Total Unique Labels:** 15

## 2. File Inventory & Storage Metrics

| Filename | Size (Bytes) | Size (MB) | Flow Records | Within-File Duplicates | MD5 Checksum |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | 77,123,859 | 73.55 MB | 225,745 | 2,633 | `b2b2764e4c8a4c390506de7ee81c32ee` |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | 76,906,168 | 73.34 MB | 286,467 | 72,353 | `4892380364f9c7ef297d3094d19f26bf` |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | 58,316,725 | 55.62 MB | 191,033 | 6,888 | `134224ec64782709ae1078379f72c4fa` |
| `Monday-WorkingHours.pcap_ISCX.csv` | 176,927,918 | 168.73 MB | 529,918 | 26,935 | `12ca72e319041856f6410e9a14d40581` |
| `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` | 83,102,436 | 79.25 MB | 288,602 | 35,630 | `29ab45bfe378d983552a801d0a90cac8` |
| `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` | 52,023,263 | 49.61 MB | 170,366 | 6,066 | `13e1c70d2b380bf5d90f82e60e7befb1` |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | 135,078,995 | 128.82 MB | 445,909 | 24,065 | `df16dccfd59a4ee126690fd6b71ee0a4` |
| `Wednesday-workingHours.pcap_ISCX.csv` | 225,166,395 | 214.74 MB | 692,703 | 81,909 | `bf0dd7e9d991987df4e13ea58a1b409c` |

*All files were verified readable and remain completely immutable (MD5 verified pre- and post-audit).*

## 3. Class & Label Distribution

| Label | Record Count | Percentage | Class Type |
| :--- | :--- | :--- | :--- |
| **BENIGN** | 2,273,097 | 80.3004% | Normal / Baseline |
| **DoS Hulk** | 231,073 | 8.163% | Attack / Anomaly |
| **PortScan** | 158,930 | 5.6144% | Attack / Anomaly |
| **DDoS** | 128,027 | 4.5227% | Attack / Anomaly |
| **DoS GoldenEye** | 10,293 | 0.3636% | Attack / Anomaly |
| **FTP-Patator** | 7,938 | 0.2804% | Attack / Anomaly |
| **SSH-Patator** | 5,897 | 0.2083% | Attack / Anomaly |
| **DoS slowloris** | 5,796 | 0.2048% | Attack / Anomaly |
| **DoS Slowhttptest** | 5,499 | 0.1943% | Attack / Anomaly |
| **Bot** | 1,966 | 0.0695% | Attack / Anomaly |
| **Web Attack � Brute Force** | 1,507 | 0.0532% | Attack / Anomaly |
| **Web Attack � XSS** | 652 | 0.023% | Attack / Anomaly |
| **Infiltration** | 36 | 0.0013% | Attack / Anomaly |
| **Web Attack � Sql Injection** | 21 | 0.0007% | Attack / Anomaly |
| **Heartbleed** | 11 | 0.0004% | Attack / Anomaly |

### Class Imbalance Findings:
- **Dominant Class:** `BENIGN` represents **80.3004%** of all traffic flows.
- **Severe Minority Classes:** Attacks such as `Heartbleed` (11 flows), `Infiltration` (36 flows), and `Web Attack - Sql Injection` (21 flows) represent extreme edge classes requiring stratified splitting and macro-averaged metrics during evaluation.

## 4. Missing & Infinite Values Analysis

### Columns with Missing (NaN / Null) Values:
| Feature (Raw) | Clean Feature Name | Missing Count | Percentage |
| :--- | :--- | :--- | :--- |
| `Flow Bytes/s` | `Flow Bytes/s` | 1,358 | 0.048% |

### Columns with Infinite (`+inf` / `-inf`) Values:
| Feature (Raw) | Clean Feature Name | +Inf Count | -Inf Count | Total Inf | Percentage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Flow Bytes/s` | `Flow Bytes/s` | 1,509 | 0 | 1,509 | 0.0533% |
| ` Flow Packets/s` | `Flow Packets/s` | 2,867 | 0 | 2,867 | 0.1013% |

> **Root Cause:** Infinite values occur in rate-based calculations (e.g., `Flow Bytes/s` and `Flow Packets/s`) when `Flow Duration` is zero.

## 5. Duplicate Rows Analysis

- **Total Dataset-wide Duplicates:** **308,381** (10.894% of the dataset).
- **Duplication Rationale:** Repeated identical network flows commonly arise from automated heartbeat probes, recurring background services, or network tapping artifacts.
- **Recommendation:** De-duplication or careful partitioned deduplication must be evaluated in Stage 2 to prevent identical flows from leaking across train and test partitions.

## 6. Constant & Zero-Variance Columns

| Feature (Raw) | Clean Name | Constant Value |
| :--- | :--- | :--- |
| ` Bwd PSH Flags` | `Bwd PSH Flags` | `0.0` |
| ` Bwd URG Flags` | `Bwd URG Flags` | `0.0` |
| `Fwd Avg Bytes/Bulk` | `Fwd Avg Bytes/Bulk` | `0.0` |
| ` Fwd Avg Packets/Bulk` | `Fwd Avg Packets/Bulk` | `0.0` |
| ` Fwd Avg Bulk Rate` | `Fwd Avg Bulk Rate` | `0.0` |
| ` Bwd Avg Bytes/Bulk` | `Bwd Avg Bytes/Bulk` | `0.0` |
| ` Bwd Avg Packets/Bulk` | `Bwd Avg Packets/Bulk` | `0.0` |
| `Bwd Avg Bulk Rate` | `Bwd Avg Bulk Rate` | `0.0` |

*These columns provide zero mutual information and should be candidates for safe removal during feature selection.*

## 7. Negative & Suspicious Numerical Values

| Feature (Raw) | Clean Name | Negative Record Count | Min Value Observed | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| ` Flow Duration` | `Flow Duration` | 115 | `-13.0` | 0.0041% |
| `Flow Bytes/s` | `Flow Bytes/s` | 85 | `-261000000.0` | 0.003% |
| ` Flow Packets/s` | `Flow Packets/s` | 115 | `-2000000.0` | 0.0041% |
| ` Flow IAT Mean` | `Flow IAT Mean` | 115 | `-13.0` | 0.0041% |
| ` Flow IAT Max` | `Flow IAT Max` | 115 | `-13.0` | 0.0041% |
| ` Flow IAT Min` | `Flow IAT Min` | 2,891 | `-14.0` | 0.1021% |
| ` Fwd IAT Min` | `Fwd IAT Min` | 17 | `-12.0` | 0.0006% |
| ` Fwd Header Length` | `Fwd Header Length` | 35 | `-32212234632.0` | 0.0012% |
| ` Bwd Header Length` | `Bwd Header Length` | 22 | `-1073741320.0` | 0.0008% |
| ` Fwd Header Length.1` | `Fwd Header Length.1` | 35 | `-32212234632.0` | 0.0012% |
| `Init_Win_bytes_forward` | `Init_Win_bytes_forward` | 1,001,189 | `-1.0` | 35.3684% |
| ` Init_Win_bytes_backward` | `Init_Win_bytes_backward` | 1,441,552 | `-1.0` | 50.9249% |
| ` min_seg_size_forward` | `min_seg_size_forward` | 35 | `-536870661.0` | 0.0012% |

> **Anomaly Insight:** Negative values in fields like `Flow Duration`, `Init_Win_bytes_backward`, and `Fwd Header Length` represent 32-bit integer overflow bugs in the CICFlowMeter extraction tool.

## 8. Column Formatting & Whitespace Artifacts

- **Columns with Leading/Trailing Whitespaces:** 65 out of 79
- **Examples:** `' Destination Port'` (leading space), `' Label'` (leading space), `'Total Length of Fwd Packets'` (no space).
- **Action Required:** Column names must be normalized via `.str.strip()` during ingestion to prevent indexing errors.

## 9. Potential Data Leakage Vectors & Trivial Identifiers

1. **` Destination Port`:** Port numbers (e.g., port 80, 443, 22, 21) can directly leak service identities rather than malicious behavioral patterns. Relying purely on port numbers leads to models that memorize specific service ports instead of generalized anomalies.
2. **Row Duplication:** If identical duplicate rows appear in both training and test sets, models can achieve artificially inflated accuracy scores (data leakage).
3. **Cross-file Temporal Dependency:** Traffic was captured chronologically over 5 days. Random naive cross-validation without temporal or stratified awareness can cause temporal leakage.

## 10. Recommended Stage 2 Preprocessing Pipeline

1. **Column Name Sanitization:** Strip leading and trailing whitespace from all 79 feature names.
2. **Zero-Variance Feature Removal:** Safely drop identified constant columns (e.g., `Bwd URG Flags`, `Fwd URG Flags`, `CWE Flag Count`).
3. **Missing Value Handling:** Impute or remove the small fraction of null records in `Flow Bytes/s` (<0.05%).
4. **Infinite Value Treatment:** Replace `+inf` and `-inf` with either maximum/minimum finite values or robust median thresholds.
5. **Integer Overflow Handling:** Clip or filter negative values resulting from CICFlowMeter counter overflows in durations and header lengths.
6. **Leak-Free Splitting:** Implement stratified splitting maintaining exact class representation, while ensuring no duplicate leak between train, validation, and test splits.