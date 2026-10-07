# CyberSentry Preprocessing & Feature Engineering Methodology

> **Stage:** STAGE 2 - Preprocessing & Feature Engineering  
> **Status:** Complete (Fully Reproducible & Leak-Free)  
> **Random Seed:** `42`

---

## 1. Summary of Data Transformations

- **Raw Input Rows:** 2,830,743
- **Duplicate Rows Removed:** 308,381
- **Distinct Cleaned Rows:** 2,522,362
- **Original Input Features:** 78
- **Features Removed:** 9 (Bwd PSH Flags, Bwd URG Flags, Fwd Avg Bytes/Bulk, Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk, Bwd Avg Packets/Bulk, Bwd Avg Bulk Rate, Fwd Header Length.1)
- **Final Preprocessed Features:** 69
- **Target Classes:** 15 classes (preserves full 15-class taxonomy)

## 2. Train / Validation / Test Split Structure

| Split Partition | Record Count | Proportion | File Path |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 1,765,653 | 70.0% | `data/processed/train/train.parquet` |
| **VALIDATION** | 378,354 | 15.0% | `data/processed/validation/validation.parquet` |
| **TEST** | 378,355 | 15.0% | `data/processed/test/test.parquet` |

### Stratified Class Representation Across Splits:

| Target Class | Total Count | Train Split | Validation Split | Test Split |
| :--- | :--- | :--- | :--- | :--- |
| **BENIGN** | 2,096,484 | 1,467,539 | 314,472 | 314,473 |
| **Bot** | 1,953 | 1,367 | 293 | 293 |
| **DDoS** | 128,016 | 89,611 | 19,202 | 19,203 |
| **DoS GoldenEye** | 10,286 | 7,200 | 1,543 | 1,543 |
| **DoS Hulk** | 172,849 | 120,994 | 25,927 | 25,928 |
| **DoS Slowhttptest** | 5,228 | 3,660 | 784 | 784 |
| **DoS slowloris** | 5,385 | 3,770 | 807 | 808 |
| **FTP-Patator** | 5,933 | 4,153 | 890 | 890 |
| **Heartbleed** | 11 | 8 | 2 | 1 |
| **Infiltration** | 36 | 25 | 6 | 5 |
| **PortScan** | 90,819 | 63,573 | 13,623 | 13,623 |
| **SSH-Patator** | 3,219 | 2,253 | 483 | 483 |
| **Web Attack Brute Force** | 1,470 | 1,029 | 221 | 220 |
| **Web Attack Sql Injection** | 21 | 15 | 3 | 3 |
| **Web Attack XSS** | 652 | 456 | 98 | 98 |

*Every minority class (including Heartbleed with 11 total samples) is guaranteed representation in all three splits without leakage.*

## 3. Engineering Decisions & Treatment Rationales

### 3.1 Zero-Variance Features
The 8 constant features identified in Stage 1 (`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`) possess zero mutual information across all 2.83M records. They were removed.

### 3.2 Duplicate Feature Elimination
`Fwd Header Length.1` is an identical duplicate column of `Fwd Header Length` generated during pcap CSV extraction. It was removed as redundant.

### 3.3 Handling Duplicate Rows
Exactly **308,381** duplicate rows were removed prior to partitioning. This prevents cross-split data leakage where identical network flow observations occur in both training and test partitions.

### 3.4 Handling Negative and Overflow Values
1. **TCP Window Sentinels (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`):**
   - Contains `-1.0` in 35.4% and 50.9% of records.
   - **Decision:** PRESERVED. In socket flow logging, `-1` represents that no initial TCP window handshake was negotiated (e.g., UDP flows or mid-stream flows). Clipping to zero would corrupt this protocol distinction.
2. **Negative Duration and Inter-Arrival Times:**
   - Clock desynchronization in pcap capture causes 115 records with negative duration (down to `-13 \mu s`).
   - **Decision:** Clipped to `0.0`.
3. **Integer Overflow in Header Lengths:**
   - 35 records with negative values down to `-32,212,234,632.0` due to 32-bit signed integer underflow in CICFlowMeter.
   - **Decision:** Replaced with NaN and imputed using train-learned medians.

### 3.5 Handling Infinite and Missing Values
Rate features (`Flow Bytes/s`, `Flow Packets/s`) produce infinite values when `Flow Duration == 0`. All `+inf` and `-inf` were mapped to NaN, and imputed deterministically using the median computed **strictly from the training partition**.

### 3.6 Identifier and Leakage Policy
- Raw IP addresses, Flow IDs, and Timestamps are absent in `MachineLearningCVE`.
- `Destination Port` is preserved as a numerical feature representing network service layer destinations (e.g. 80, 443, 21, 22).

## 4. Final Feature Schema

A total of **69** features remain:
1. `Destination Port`
2. `Flow Duration`
3. `Total Fwd Packets`
4. `Total Backward Packets`
5. `Total Length of Fwd Packets`
6. `Total Length of Bwd Packets`
7. `Fwd Packet Length Max`
8. `Fwd Packet Length Min`
9. `Fwd Packet Length Mean`
10. `Fwd Packet Length Std`
11. `Bwd Packet Length Max`
12. `Bwd Packet Length Min`
13. `Bwd Packet Length Mean`
14. `Bwd Packet Length Std`
15. `Flow Bytes/s`
16. `Flow Packets/s`
17. `Flow IAT Mean`
18. `Flow IAT Std`
19. `Flow IAT Max`
20. `Flow IAT Min`
21. `Fwd IAT Total`
22. `Fwd IAT Mean`
23. `Fwd IAT Std`
24. `Fwd IAT Max`
25. `Fwd IAT Min`
26. `Bwd IAT Total`
27. `Bwd IAT Mean`
28. `Bwd IAT Std`
29. `Bwd IAT Max`
30. `Bwd IAT Min`
31. `Fwd PSH Flags`
32. `Fwd URG Flags`
33. `Fwd Header Length`
34. `Bwd Header Length`
35. `Fwd Packets/s`
36. `Bwd Packets/s`
37. `Min Packet Length`
38. `Max Packet Length`
39. `Packet Length Mean`
40. `Packet Length Std`
41. `Packet Length Variance`
42. `FIN Flag Count`
43. `SYN Flag Count`
44. `RST Flag Count`
45. `PSH Flag Count`
46. `ACK Flag Count`
47. `URG Flag Count`
48. `CWE Flag Count`
49. `ECE Flag Count`
50. `Down/Up Ratio`
51. `Average Packet Size`
52. `Avg Fwd Segment Size`
53. `Avg Bwd Segment Size`
54. `Subflow Fwd Packets`
55. `Subflow Fwd Bytes`
56. `Subflow Bwd Packets`
57. `Subflow Bwd Bytes`
58. `Init_Win_bytes_forward`
59. `Init_Win_bytes_backward`
60. `act_data_pkt_fwd`
61. `min_seg_size_forward`
62. `Active Mean`
63. `Active Std`
64. `Active Max`
65. `Active Min`
66. `Idle Mean`
67. `Idle Std`
68. `Idle Max`
69. `Idle Min`

---

## 5. Evaluation Split Strategies

CyberSentry establishes two complementary partitioning strategies to support both controlled model benchmarking and realistic research-grade out-of-distribution generalization testing.

### A. Random Stratified Split (`data/processed/train/`, `validation/`, `test/`)
- **Partition Ratio:** 70% Train (1,765,653 rows), 15% Validation (378,354 rows), 15% Test (378,355 rows).
- **Core Purpose:** Controlled in-sample baseline comparison. Guarantees that every one of the 15 attack classes (including extreme minority classes such as Heartbleed with 11 samples, and Web Attack Sql Injection with 21 samples) is represented proportionally across training, validation, and test sets.
- **Deduplication:** 308,381 exact duplicate rows removed before splitting to ensure zero cross-split duplicate leakage.
- **Limitation:** In the CIC-IDS2017 testbed, network flows from a given attack or benign period were generated consecutively. Random shuffling across the entire 5-day capture means that related flows from the exact same session or connection burst may appear in both train and test partitions, potentially inflating in-sample evaluation scores.

### B. Temporal / Scenario-Aware Split (`data/processed/temporal/`)
- **Partition Allocation:**
  - **Temporal Train:** Monday, Tuesday, Wednesday captures (1,527,339 rows, 60.55%)
    - *Scenarios:* Baseline Benign (Monday), FTP/SSH Authentication Brute Force (Tuesday), DoS Hulk/GoldenEye/slowloris/Slowhttptest & Heartbleed (Wednesday).
  - **Temporal Validation:** Thursday morning and afternoon captures (398,717 rows, 15.81%)
    - *Scenarios:* Web Attacks (Brute Force, XSS, Sql Injection), Infiltration, and Thursday Benign baseline.
  - **Temporal Test:** Friday morning and afternoon captures (596,306 rows, 23.64%)
    - *Scenarios:* Ares Botnet (Friday morning), PortScan (Friday afternoon), DDoS (Friday afternoon), and Friday Benign baseline.
- **Core Purpose:** Simulates a realistic production SOC deployment where detection systems trained on historical traffic (Mon–Wed) must detect security events on intermediate (Thu) and future days (Fri) without prior exposure to future session dynamics.
- **Cross-Split Leakage Control:** Chronological deduplication ensures exactly **0 duplicate flows** exist between train, validation, and test partitions (filtering background periodic probes that recurred on later days).
- **Class Availability Reality:**
  - In CIC-IDS2017, specific attack tools were executed exclusively on specific calendar days (e.g., Web Attacks on Thursday only, PortScan/DDoS on Friday only).
  - Consequently, **not every class appears in every partition**. The temporal test set does not contain historical training attacks (such as DoS Hulk or SSH-Patator), and the training set does not contain Friday attacks (such as PortScan, DDoS, or Bot).
- **Metric Interpretation:** Evaluation on the temporal test set measures **zero-day anomaly detection and out-of-scenario generalization** rather than closed-world supervised multiclass classification. Metrics must be interpreted per-class and in conjunction with unsupervised anomaly detectors (Isolation Forest).