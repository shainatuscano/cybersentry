# Stage 3: ML Detection Results

> **Data used:** the cleaned **blocked split** in `data/processed/` (see `DATA.md`): 6 grouped classes, 6 of the 8
> CIC-IDS2017 files, no PortScan or DDoS. The 15-class random stratified and temporal parquet splits produced by
> `src/cybersentry/data/preprocess.py` / `temporal_split.py` were not available when this stage was run (they need the
> 8 raw CSVs), so every number below refers to the 6-class data. Re-running the scripts on another split
> replaces these numbers; none of them carry over.
>
> All figures were produced by `python -m src.ml.stage3.train_eval`, `src.ml.stage3.anomaly` and
> `src.ml.stage3.temporal` (seed 42) and are stored in `ml/results/stage3/`.
> Machine: 4 CPU cores, 15 GB RAM, no GPU.

## 1. What was tested

| | |
| --- | --- |
| Label space | Benign, Botnet, BruteForce, DoS (Hulk, GoldenEye, slowloris, Slowhttptest, Heartbleed), Infiltration, WebAttack (brute force, XSS, SQL injection) |
| Training rows | 269,440: the same rows for every model. Benign capped at 200,000 and each attack class at 60,000 (seeded sample); smaller classes kept whole. Botnet 1,363, BruteForce 6,553, DoS 60,000, Infiltration 24, WebAttack 1,500. |
| Validation | 309,496 rows (used for model selection and XGBoost early stopping only) |
| Test | 309,373 rows, natural class distribution (Benign 279,096, DoS 28,884, BruteForce 772, WebAttack 322, Botnet 293, Infiltration 6) |
| Class imbalance | `balanced` sample weights, computed once and passed to all three models. No oversampling or undersampling beyond the shared cap. |
| Features | The 69 numeric features in `models/meta.json`, identical for every model |
| Configuration | `ml/configs/stage3.yaml` |

### Why each model

- **Logistic Regression** is the linear baseline. It shows how much of the problem a linear boundary over
  standardised features solves, and it gives a lower bound that the tree models have to beat. The
  `StandardScaler` sits inside the saved pipeline, so it is fitted on training rows only and the same fitted
  scaler transforms validation, test and API inputs.
- **Random Forest** is the standard non-linear baseline for flow-feature intrusion detection. It needs little
  tuning, is robust to unscaled, heavy-tailed features, and gives a strong reference for a boosted model.
  Settings: 150 trees, unlimited depth.
- **XGBoost** is the primary candidate. Gradient boosting usually does well on tabular data and predicts quickly.
  Settings: up to 500 rounds, depth 8, learning rate 0.1, 0.8 row and column subsampling, histogram trees,
  early stopping after 20 rounds without validation improvement. It stopped at round 229.

There was no hyperparameter search. Every model ran once with the configuration above.

### Selection rule (fixed before any result was seen)

The primary detector is the model with the **highest validation macro F1**. Models within 0.005 of the best are
compared on validation macro recall, then on inference cost. The test split plays no part in selection.

## 2. Results on the held-out test split

| Metric | Logistic Regression | Random Forest | XGBoost |
| --- | --- | --- | --- |
| Accuracy | 0.9029 | 0.9981 | **0.9986** |
| Macro precision | 0.3945 | **0.8295** | 0.7750 |
| Macro recall | 0.7663 | 0.7630 | **0.8229** |
| Macro F1 | 0.4297 | 0.7903 | **0.7932** |
| Weighted precision | 0.9827 | 0.9975 | **0.9979** |
| Weighted recall | 0.9029 | 0.9981 | **0.9986** |
| Weighted F1 | 0.9393 | 0.9978 | **0.9982** |
| Attack detection rate (any attack flagged as an attack) | 0.9878 | 0.9886 | **0.9899** |
| Attack false-negative rate | 0.0122 | 0.0114 | **0.0101** |
| Attack false negatives (of 30,277 attack flows) | 369 | 345 | **305** |
| Benign false-positive rate | 0.1046 | 0.0008 | **0.0004** |
| Benign false positives (of 279,096 benign flows) | 29,196 | 226 | **104** |
| Macro ROC-AUC (one class vs rest) | 0.9701 | 0.9452 | **0.9994** |
| Macro PR-AUC (one class vs rest) | 0.5123 | 0.8025 | **0.8519** |
| Attack-vs-benign ROC-AUC | 0.9896 | 0.9998 | **0.9999** |
| Attack-vs-benign PR-AUC | 0.9336 | 0.9984 | **0.9994** |
| Training time (s) | 54.1 | **12.9** | 34.1 |
| Batch inference (ms per 1,000 flows) | **0.33** | 2.35 | 5.06 |
| Single-flow latency, median (ms) | **0.27** | 8.54 | 0.48 |

Inference times were measured on the full test split, with a single thread for the single-flow numbers.
Charts: `model_comparison.png`, `per_class_recall.png`, `confusion_{lr,rf,xgb}_test.png`.

### Per class, test split

| Class (test support) | Recall LR | Recall RF | Recall XGB | Precision LR | Precision RF | Precision XGB | Missed LR / RF / XGB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Benign (279,096) | 0.8954 | 0.9992 | 0.9996 | 0.9985 | 0.9988 | 0.9989 | 29,196 / 226 / 104 |
| Botnet (293) | **0.0000** | **0.0000** | **0.0000** | 0.0000 | 0.0000 | 0.0000 | 293 / 293 / 293 |
| BruteForce (772) | 0.9637 | 1.0000 | 1.0000 | 0.4515 | 0.9822 | 0.9897 | 28 / 0 / 0 |
| DoS (28,884) | 0.9828 | 0.9989 | 0.9998 | 0.8647 | 0.9961 | 0.9981 | 496 / 31 / 7 |
| Infiltration (6) | 0.8333 | 0.6667 | 1.0000 | 0.0010 | 1.0000 | 0.6667 | 1 / 2 / 0 |
| WebAttack (322) | 0.9224 | 0.9130 | 0.9379 | 0.0513 | 1.0000 | 0.9967 | 25 / 28 / 20 |

Per-class FPR, FNR, ROC-AUC and PR-AUC are in `test_per_class.csv`; confusion counts are in `confusion_*_test.csv`.

### Validation split (used for selection)

| Metric | LR | RF | XGB |
| --- | --- | --- | --- |
| Macro F1 | 0.4548 | **0.8959** | 0.8755 |
| Macro recall | 0.8494 | 0.8802 | **0.8817** |
| Benign FPR | 0.1066 | 0.0007 | **0.0003** |
| Attack false negatives | 306 | 228 | **205** |
| Infiltration F1 (**5 rows**) | 0.0019 | **1.0000** | 0.8333 |
| Botnet F1 (292 rows) | 0.0132 | 0.3982 | **0.4376** |

## 3. Which model performed best, and should XGBoost be the primary detector?

**Selection rule outcome: Random Forest** (validation macro F1 0.8959 vs XGBoost 0.8755). This is the model the
detection service currently serves (`ml/models/stage3/detector_meta.json`).

**This outcome rests on very little evidence.** The 0.020 gap in validation macro F1 comes almost entirely from
Infiltration, which has **5 validation rows**. XGBoost found all 5 and raised 2 false alarms (F1 0.83); Random
Forest found all 5 with none (F1 1.00). With Infiltration left out, XGBoost leads on validation (macro F1
0.8840 vs 0.8751; macro recall 0.8580 vs 0.8563). `DATA.md` already warns that Infiltration metrics are not reliable.

**On test, XGBoost is ahead on the metrics this project cares most about:** macro recall (0.823 vs 0.763), missed
attacks (305 vs 345), benign false positives (104 vs 226), per-class recall (higher on DoS, Infiltration and
WebAttack; tied on BruteForce at 1.0 and Botnet at 0), macro ROC-AUC and PR-AUC, and single-flow latency (0.48 ms vs 8.5 ms). Random Forest is
ahead on macro precision (0.830 vs 0.775) and training time.

**Conclusion.** The measured results do not support claiming that XGBoost wins under the pre-declared rule. They
do support XGBoost as a defensible primary detector: it is never meaningfully worse, it is better on recall,
false negatives and latency, and the rule's preference for Random Forest turns on 2 predictions out of 5 rows.
Switching should be an explicit, recorded decision, not a silent change of rule after the fact:

```bash
python -m src.ml.stage3.set_detector xgb --reason "validation lead of RF rests on 5 Infiltration rows; XGB better on recall/FN/latency"
python -m src.ml.stage3.set_detector --auto     # back to the rule's choice
```

If the selection rule is revised for future stages, a reasonable choice is to exclude classes with fewer than
about 30 validation rows from the selection metric, and to declare that before re-running.

**Logistic Regression is not competitive.** With balanced weights it raises 29,196 benign false positives on
test (FPR 10.5%), and its macro precision is 0.39. A linear boundary is not enough for these features.

## 4. Isolation Forest (complementary anomaly detector)

| | |
| --- | --- |
| Trained on | 200,000 **benign** rows of the training split (seeded sample). No attack rows: the model learns normal traffic only. |
| Score | `anomaly_score = -score_samples(x)`; higher means more unusual |
| `contamination` | `"auto"`. It is **not** used for decisions: sklearn only uses it for `predict()`, which this project does not call. |
| Threshold | **0.6204** = 99th percentile of anomaly scores on **benign validation** flows, i.e. a fixed ~1% false-positive budget on normal traffic. Chosen on validation, applied unchanged to test. |
| Flag | `anomaly_flag = anomaly_score > 0.6204` |

| Test split | Value |
| --- | --- |
| Benign flows flagged (FPR) | 1.05% (2,944 of 279,096) |
| Attack flows flagged (recall) | 58.4% |
| Attack-vs-benign ROC-AUC / PR-AUC | 0.905 / 0.644 |
| Flagged per class | DoS 61.2%, Infiltration 33.3% (2 of 6), Botnet 0%, BruteForce 0%, WebAttack 0% |

**What it adds on top of the classifier (test, Random Forest + Isolation Forest, rule "attack prediction OR anomaly flag"):**

| | Random Forest alone | Combined |
| --- | --- | --- |
| Attack detection rate | 0.9886 | 0.9886 |
| Benign FPR | 0.0008 | 0.0114 |

Of the 345 attacks Random Forest missed, Isolation Forest flagged **0**. On this split it adds no detections and
raises the benign false-positive rate from 0.08% to 1.14%. It mostly flags DoS, which the classifier already
catches. Its value shows up in the temporal check below, where it flags 43% of Infiltration flows (15 of 35) the
classifier had never seen. On classes the classifier has learned, the anomaly flag should therefore be read as a
lower-priority signal. The Detection Agent does this: an anomaly flag alone gives SUSPICIOUS / MEDIUM, never
ATTACK_CANDIDATE.

## 5. Temporal / scenario-aware evaluation

Partitions follow the existing file assignment in `src/cybersentry/data/temporal_split.py`, unchanged:
Mon–Wed → train, Thursday → validation, Friday → test. It was applied to the cleaned rows through their
`src_file` column. The Friday-afternoon PortScan/DDoS captures are not in this dataset.

| Class | Train (Mon–Wed) | Validation (Thu) | Test (Fri morning) |
| --- | --- | --- | --- |
| Benign | 1,288,146 | 396,825 | 181,849 |
| DoS | 193,428 | absent | absent |
| BruteForce | 9,144 | absent | absent |
| WebAttack | absent | 2,143 | absent |
| Infiltration | absent | 35 | absent |
| Botnet | absent | absent | 1,948 |
| PortScan, DDoS | not in dataset | not in dataset | not in dataset |

**Every attack class in the temporal validation and test days is absent from the training days.** A closed-world
classifier cannot name a class it never saw, so per-class precision, recall and F1, macro F1 and a multiclass
confusion matrix would be meaningless here and are not reported. What is valid:

- the benign false-positive rate on unseen days (Benign is in the training label space), and
- for each unseen attack class, the share of its flows flagged as *some* attack (open-world detection).

The selected model type (Random Forest, same configuration, same cap and weights) was retrained on Mon–Wed:
269,144 rows of Benign, BruteForce and DoS. Isolation Forest was retrained on Mon–Wed benign rows, and its
threshold was set from Thursday benign rows.

| Day / class | Random split: classifier | Temporal: classifier | Temporal: Isolation Forest | Temporal: combined |
| --- | --- | --- | --- | --- |
| Thu Benign (FPR) | 0.0008 | 0.0005 | 0.0100 | 0.0105 |
| Thu WebAttack (2,143) | 0.9410 | **0.0551** | 0.0070 | 0.0621 |
| Thu Infiltration (35) | 0.6667 | **0.0000** | 0.4286 | 0.4286 |
| Fri Benign (FPR) | 0.0008 | 0.0005 | 0.0130 | 0.0135 |
| Fri Botnet (1,948) | 0.0000 | **0.0000** | 0.0056 | 0.0056 |

(Random-split column: share of test flows predicted as any attack class, from the Stage 3 models.)

**Interpretation.** On the random blocked split, every attack family appears in training, so the scores measure
recognition of *known* attack types. The temporal check measures something else: attack families that appear
only on a later day. There the classifier misses almost all of them (5.5% of WebAttack, 0% of Botnet and
Infiltration), and calls 2,060 of 2,178 Thursday attack flows Benign. Isolation Forest recovers 43% of
Infiltration but almost none of WebAttack or Botnet, whose flow statistics look like normal traffic. Benign
false-positive rates stay low on new days, so the model does not raise alarms on normal traffic just because the
day changed.

The high random-split scores should therefore be read as **known-attack recognition**, not as evidence that
CyberSentry detects new attack families. Results: `ml/results/stage3/temporal/`.

## 6. Limitations

- **Not the 15-class benchmark.** These results use the 6-class blocked split. When the 15-class random stratified
  and temporal parquet splits are available, rerun the three scripts against them. The code reads class names
  from the data, but the loaders currently read `data/processed/*.csv.gz` (`src/ml/common.load_table`).
- **No PortScan or DDoS** in the data.
- **Tiny classes:** Infiltration has 24 training, 5 validation and 6 test rows, so its numbers, and any macro
  average that includes it, can swing by 0.1 or more on a single flow.
- **Botnet is not detected on test by any model.** Validation Botnet recall is about 0.31 for every model. Test
  blocks come from different parts of the capture than training blocks, and none of the 293 test Botnet flows
  are recognised. This is a real failure, not a metric artefact.
- **Training cap.** Benign is capped at 200,000 and DoS at 60,000 rows, so training covers part of the data. The
  cap and the balanced weights shift predicted probabilities away from the true class frequencies. Probabilities
  should be read as model confidence, not calibrated likelihoods.
- **One run, one seed, no tuning:** there are no confidence intervals. Differences under about 0.01 in macro
  scores should not be treated as meaningful.
- **2017 lab data** with known labelling issues; there are no IPs or timestamps, so no host-level context.
