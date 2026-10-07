# CyberSentry: Review 2 ML half

Data audit, leakage-safe split, baselines + XGBoost, Isolation Forest, detection API.
The agent layer (LangGraph, tools, RAG) is the next step and builds on `POST /detect`.

## Two codebases in one repo

This repo merges two pieces of work on the same project. Both use CIC-IDS2017, but they produce different splits,
so do not mix their numbers in one table.

| | `src/ml`, `src/api`, `tools/` (this README) | `src/cybersentry` (scaffold, see `AGENTS.md`, `docs/`) |
| --- | --- | --- |
| Raw data | `data/raw/*.csv` (6 files used, see `DATA.md`) | `data/raw/cic_ids2017/MachineLearningCVE/*.csv` (all 8) |
| Classes | 6 grouped classes | 15 original labels |
| Split | Blocked by 1,000 capture-order rows | Stratified row split (`preprocess`), split by capture day (`temporal_split`) |
| Output | `data/processed/*.csv.gz` (versioned), `models/`, `reports/` | `data/processed/{train,validation,test,temporal}/*.parquet`, `ml/results/` (not versioned) |
| Status | Training, anomaly detection and API implemented | Audit and preprocessing scripts; agents, backend, frontend are empty placeholders |

Project overview, architecture and roadmap: `docs/overview.md`, `docs/architecture.md`, `docs/development-roadmap.md`.
Engineering rules for contributors and coding agents: `AGENTS.md`, `docs/development_rules.md`.
The numbers in `docs/dataset-audit.md` and `docs/preprocessing.md` come from the scaffold's run on all 8 files
and are rewritten when its scripts run.

Scaffold scripts (need all 8 raw files):

```bash
PYTHONPATH=src python -m cybersentry.data.audit            # -> docs/dataset-audit.md, ml/results/
PYTHONPATH=src python -m cybersentry.data.preprocess       # -> data/processed/{train,validation,test}/, docs/preprocessing.md
PYTHONPATH=src python -m cybersentry.data.temporal_split   # -> data/processed/temporal/
```

Tests: `pip install -r requirements-dev.txt && python -m pytest`. `tests/test_audit.py`, `test_preprocessing.py`
and `test_temporal_split.py` check those script outputs, so they are skipped until the scripts above have been run.

## Setup (MacBook, Apple silicon)

```bash
cd cybersentry
python3 -m venv venv && source venv/bin/activate
brew install libomp                 # XGBoost needs OpenMP on macOS
pip install -r requirements.txt
```

No GPU is needed. XGBoost, Random Forest and Isolation Forest all run on the CPU.

## Data

The cleaned and split dataset is already in this repo: see `DATA.md`. Skip straight to `python -m src.ml.train_models`.
To rebuild it, or to use your own files:

Put the CIC-IDS2017 CSV files (the `MachineLearningCVE` set) in `data/raw/`.
Any CSV with a `Label` column and numeric feature columns works, including files you already cleaned.
Keep rows in their original capture order: the split relies on it (see below).

## Run, in this order

```bash
python -m src.ml.prepare_data        # audit, clean, group labels, blocked split  -> data/processed, reports/
python -m src.ml.train_models        # Logistic Regression, Random Forest, XGBoost -> models/, reports/val_*.csv
python -m src.ml.anomaly             # Isolation Forest + threshold               -> models/iso.joblib
python -m src.ml.evaluate_test --confirm   # run ONCE, when tuning is finished   -> reports/test_*.csv
uvicorn src.api.main:app --reload    # detection API, docs at http://127.0.0.1:8000/docs
```

Demo helpers:

```bash
python -m tools.sample_event --list                  # labels available in the test split
python -m tools.sample_event --label DoS --post      # take a real DoS flow from the test split and POST it to /detect
```

## What each step produces (use these in your Review 2 slides)

| File | Use |
| --- | --- |
| `reports/audit.json`, `reports/class_distribution.png`, `reports/split_distribution.csv` | Dataset audit slide |
| `reports/val_metrics.csv`, `reports/val_per_class.csv` | Model comparison table (validation) |
| `reports/val_confusion_xgb.png`, `reports/xgb_feature_importance.png` | XGBoost results |
| `reports/anomaly_hist.png`, `reports/val_anomaly_per_class.csv` | Isolation Forest slide |
| `reports/test_metrics.csv`, `reports/test_anomaly_and_combined.csv` | Final test numbers (once) |

## Design choices to be able to explain

- **Blocked split, not a split by day.** Each CIC-IDS2017 day has different attacks, so a day split would remove
  whole classes from training. A random row split leaks, because neighbouring flows from one attack run are
  near-copies. Rows stay in capture order, are cut into blocks of 1,000 consecutive rows, and whole blocks go to
  train (70%), validation (15%) or test (15%), stratified by each block's main class. Classes too small to fill
  several blocks (for example Infiltration, about 36 rows) are split chronologically at row level. The script warns
  when a class is missing from a split. If your cleaned file was shuffled, this protection is lost: say so in the report.
- **Identifiers removed:** Flow ID, source and destination IP, timestamp and source port are not model features.
- **Same data for every model.** Training rows are capped (default 200,000 benign, 60,000 per attack class) so the
  three models are compared on identical data and everything fits comfortably in 16 GB. Use `--cap-benign 0 --cap-attack 0`
  for the full set.
- **Validation for decisions, test once.** `train_models` and `anomaly` only look at validation.
- **Isolation Forest** is fitted on benign training flows only. Its threshold is the 99th percentile of benign
  validation scores, which fixes the false-positive budget at about 1%.
- **Combined rule** (`src/ml/common.py: decide`): known attack with probability of at least 0.6, or a low-confidence
  attack that is also anomalous, or a benign-looking flow that is anomalous, becomes "investigate".
- **Latency.** scikit-learn's Isolation Forest has about 20 ms of fixed overhead per call, so `/detect` for a single
  flow costs about that. `/detect/batch` scores 1,000 flows in roughly 25 ms. Report both numbers.

## Smoke test without the real data

```bash
export CS_WORKDIR=/tmp/cs_smoke
python -m tools.make_synthetic_cic      # SYNTHETIC random data: only checks that the scripts run
python -m src.ml.prepare_data && python -m src.ml.train_models && python -m src.ml.anomaly
```

Never quote numbers from synthetic data.

## Next step

LangGraph: `detect -> investigate -> report`, with read-only tools (`get_related_events`, `get_ip_history`,
`search_mitre`) and the PostgreSQL event store. The `/detect` response already contains what the Detection node needs
(`label`, `probability`, `anomaly_score`, `investigate`, `reason`).
