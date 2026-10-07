# Cleaned dataset (CIC-IDS2017, 6 of 8 files)

`data/processed/` holds the cleaned, split data that the pipeline trains on. It is ready to use: clone the repo
and run `python -m src.ml.train_models` (no need to run `prepare_data` first).

| File | Rows | Notes |
| --- | --- | --- |
| `train.csv.gz.part-00` ... `-04` | 1,454,649 | One gzip file cut into 5 pieces, because GitHub rejects files over 100 MB. `src/ml/common.py` joins them automatically. To join by hand: `cat train.csv.gz.part-* > train.csv.gz` |
| `val.csv.gz` | 309,496 | Validation split |
| `test.csv.gz` | 309,373 | Test split. Use it once, at the end. |
| `../models/meta.json` | | Feature list (69), class list, training medians, block size, seed |
| `../reports/` | | `audit.json`, `split_distribution.csv`, `class_distribution.png` |

Columns: 69 numeric features (float32), `label`, `src_file`, `row_in_file`. `src_file` and `row_in_file` are
metadata, not model features.

## What was done to the raw files

Source: the six `MachineLearningCVE` CSVs for Monday, Tuesday, Wednesday, Thursday morning, Thursday afternoon
and Friday morning (2,318,531 rows). The Friday-afternoon PortScan and DDoS files were not available, so those
two classes are absent.

| Step | Rows affected |
| --- | --- |
| NaN or infinite values (`Flow Bytes/s`, `Flow Packets/s`) removed | 2,462 |
| Impossible negative values (durations, inter-arrival times, header lengths) removed; `Init_Win_bytes_*` = -1 kept | 2,641 |
| Exact duplicate rows removed | 239,618 |
| Identical features with conflicting labels removed | 292 |
| **Rows left** | **2,073,518** |

Also removed: the duplicate column `Fwd Header Length.1`, 8 constant columns, and identifier columns (Flow ID,
IPs, timestamp, source port, where present). Labels are grouped into 6 classes: Benign, DoS (Hulk, GoldenEye,
slowloris, Slowhttptest, Heartbleed), BruteForce (FTP-Patator, SSH-Patator), WebAttack (brute force, XSS, SQL
injection), Botnet, Infiltration.

## Split

Blocked split with seed 42: rows stay in capture order, each file is cut into blocks of 1,000 consecutive rows,
and whole blocks go to train (70%), validation (15%) or test (15%). Classes too small to fill several blocks
(Infiltration) are split chronologically. No identical flow appears in two splits.

| Class | Train | Val | Test |
| --- | --- | --- | --- |
| Benign | 1,308,242 | 279,482 | 279,096 |
| DoS | 136,967 | 27,577 | 28,884 |
| BruteForce | 6,553 | 1,819 | 772 |
| WebAttack | 1,500 | 321 | 322 |
| Botnet | 1,363 | 292 | 293 |
| Infiltration | 24 | 5 | 6 |

## Limits to state in any report

- No PortScan or DDoS class until the two Friday-afternoon files are added and `prepare_data` is re-run.
- The files have no IP addresses or timestamps, so host-level correlation needs another source.
- Infiltration has 35 rows in total; its metrics are not reliable.
- CIC-IDS2017 is a 2017 lab-generated dataset with published labelling issues.

## Source and citation

CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick. Please cite: I. Sharafaldin,
A. H. Lashkari, A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization", ICISSP 2018. Check the CIC website for the current terms of use before sharing publicly.

## Rebuild from the raw files

Put the raw CSVs in `data/raw/` and run `python -m src.ml.prepare_data --also-csv`. The result is identical (fixed seed).
