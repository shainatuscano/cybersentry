# Detection Architecture: ML Detection Service and Detection Agent

> Stage covered: from the cleaned dataset up to the first AI agent (Detection Agent).
> Measured model results: `docs/stage3-ml-results.md`.

## 1. End-to-end flow

```
Network flow (69 CIC-IDS2017 features)
    │
    ▼
Input validation ── missing / unknown / non-numeric / non-finite feature → HTTP 422
    │
    ▼
ML detection (src/cybersentry/detection/service.py)
    ├── Supervised classifier (selected model, currently Random Forest)
    │       → prediction, confidence, class probabilities
    └── Isolation Forest (trained on benign flows)
            → anomaly_score, anomaly_flag (score > threshold)
    │
    ▼
Structured DetectionResult (src/cybersentry/detection/schemas.py)
    │
    ▼
Detection Agent: LangGraph workflow (src/cybersentry/agents/detection_agent/)
    receive → validate → evaluate_signals → decide
    │
    ▼
AgentDecision: NORMAL / SUSPICIOUS / ATTACK_CANDIDATE (or INVALID_INPUT)
    │
    ▼
[next stage] Investigation Agent receives decisions with investigation_required = true
```

## 2. ML detection versus the Detection Agent

| | ML detection service | Detection Agent |
| --- | --- | --- |
| Question it answers | What does this flow look like, numerically? | Should this event enter the investigation workflow, and how urgently? |
| Input | 69 numeric flow features | One structured `DetectionResult` |
| Method | Trained models (classifier + Isolation Forest) | Fixed, auditable rule table run as a LangGraph workflow |
| Output | `prediction`, `confidence`, `class_probabilities`, `anomaly_score`, `anomaly_flag`, `is_suspicious` | `status`, `priority`, `investigation_required`, `next_step`, `reason`, `rules_fired`, `evidence` |
| Can it see raw traffic? | Yes, the features | No, only the detection result |

The models do the mathematics. The agent never re-classifies a flow, changes a score or adds a fact; it
interprets the signals the models produced and routes the event.

### Why two detectors

- **The supervised classifier** answers: *which known attack class does this flow resemble?* It is accurate on
  attack families it was trained on (test macro recall 0.76–0.82 for the tree models, benign FPR under 0.1%).
- **Isolation Forest** answers: *how unusual is this flow compared with learned normal traffic?* It needs no
  attack labels, so it can react to patterns the classifier has no class for. In the temporal check it flagged
  43% of Infiltration flows that the classifier, never having seen Infiltration, missed entirely.
- They are complementary, not interchangeable. On known classes the anomaly flag added no detections and cost
  about 1 percentage point of benign false positives (see the results document). That is why an anomaly flag on its own
  produces SUSPICIOUS, never ATTACK_CANDIDATE.

## 3. Why an LLM is not the network classifier

- **Accuracy:** a trained tree model on these 69 numeric features reaches attack-vs-benign ROC-AUC ≈ 0.9999. An LLM
  reading a list of numbers has no advantage and no calibrated output.
- **Determinism and audit:** the same flow must always give the same verdict, with a probability that can be
  measured, thresholded and evaluated on a test split. LLM outputs vary with sampling and prompts.
- **Cost and latency:** classification takes 0.3–9 ms per flow on a CPU. An 8B-parameter LLM call is orders of
  magnitude slower and would not keep up with flow rates in a data centre.
- **Hallucination risk:** `AGENTS.md` forbids inventing evidence. A classifier cannot invent an IP address or a CVE;
  a language model can.

**LLM use in this stage: none.** Every Detection Agent step compares numbers against thresholds, so a
deterministic rule table is exact, testable and explainable without a model. No Ollama or Qwen configuration exists
in the project yet. An LLM becomes useful in later stages (investigation reasoning over retrieved evidence, report
writing), always fed with verified structured data.

## 4. Detection service and API

**Models** (written by the Stage 3 scripts, loaded once per process, never retrained by requests):
`ml/models/stage3/{rf,xgb,lr}.joblib`, `xgb_model.json` (native XGBoost), `iso.joblib`, `label_encoder.json`,
`detector_meta.json` (feature order, classes, selected classifier and why, anomaly threshold).
Override the folder with `CS_STAGE3_MODELS`.

**Suspicious rule (service):** `is_suspicious = (prediction != Benign) OR anomaly_flag`.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/health` | 200 with classifier, why it was selected, classes, threshold; 503 with the reason if models are missing |
| `GET /api/v1/detection/features` | The exact 69 feature names a request must contain |
| `POST /api/v1/detection/predict` | One flow → `DetectionResult` |
| `POST /api/v1/detection/analyze` | One flow → `DetectionResult` + Detection Agent `AgentDecision` |

Request: `{"event_id": "optional-caller-id", "features": {"Destination Port": 80, "Flow Duration": 1200, ...}}`

- All 69 features are required. Missing, unknown (for example `Source IP`), non-numeric or non-finite values
  return 422 and list the offending names. Missing values are never filled in silently.
- `event_id` is optional and echoed back unchanged; if absent it is `null`. The API never generates an ID.
- Inference is deterministic (fixed models, single-threaded scoring); tests check that repeated calls give
  identical responses.

Example response (`/predict`, the first DoS flow of the test split, Wednesday capture row 40737; real output):

```json
{
  "event_id": "test-Wednesday-workingHours-40737",
  "prediction": "DoS",
  "benign_label": "Benign",
  "confidence": 1.0,
  "class_probabilities": {"Benign": 0.0, "Botnet": 0.0, "BruteForce": 0.0, "DoS": 1.0, "Infiltration": 0.0, "WebAttack": 0.0},
  "anomaly_score": 0.65934,
  "anomaly_threshold": 0.620363,
  "anomaly_flag": true,
  "is_suspicious": true,
  "classifier": "rf",
  "anomaly_detector": "isolation_forest"
}
```

The earlier endpoints `/health`, `/detect` and `/detect/batch` (models in `models/`) still exist unchanged.

## 5. Detection Agent

**Responsibilities:**

1. Receive one structured detection result.
2. Validate it: strict schema (types, ranges, no unknown fields) plus consistency checks.
   `anomaly_flag` must match score vs threshold, `is_suspicious` must follow from prediction and flag, and the
   probabilities must sum to 1 and agree with `confidence`.
3. Interpret the signals: attack or benign prediction, confidence band, anomaly flag.
4. Decide NORMAL / SUSPICIOUS / ATTACK_CANDIDATE and whether investigation is required.
5. Return a structured `AgentDecision` whose `evidence` lists only fields copied from the input, with their source.

**What it does not do:** it has no tools, makes no network calls, takes no remediation action (no blocking,
firewall changes or deletion), invents no indicators or threat intelligence, and never states that an attack
*occurred*. An ATTACK_CANDIDATE is a prediction for the Investigation Agent to confirm or rule out. Unknown input
fields (for example a `source_ip` added by a caller) are rejected, so unverifiable data cannot pass through
as evidence.

**Decision table** (thresholds: high confidence 0.90, minimum confidence 0.60; configurable through `Thresholds`):

| Classifier says | Confidence | Anomaly flag | Status | Priority |
| --- | --- | --- | --- | --- |
| attack class | ≥ 0.90 | any | ATTACK_CANDIDATE | HIGH |
| attack class | 0.60–0.90 | yes | ATTACK_CANDIDATE | MEDIUM |
| attack class | 0.60–0.90 | no | SUSPICIOUS | MEDIUM |
| attack class | < 0.60 | yes | SUSPICIOUS | MEDIUM |
| attack class | < 0.60 | no | SUSPICIOUS | LOW |
| Benign | any | yes | SUSPICIOUS | MEDIUM |
| Benign | < 0.60 | no | SUSPICIOUS | LOW |
| Benign | ≥ 0.60 | no | NORMAL | NONE |
| invalid input | | | INVALID_INPUT | NONE |

Everything except NORMAL and INVALID_INPUT has `investigation_required = true` and
`next_step = HAND_OFF_TO_INVESTIGATION`. The 0.60 threshold matches the earlier pipeline's decision rule
(`src/ml/common.py`). The thresholds are design choices, not tuned on test data.

Example decision (`/analyze` on the same flow; real output, evidence list shortened):

```json
{
  "event_id": "test-Wednesday-workingHours-40737",
  "status": "ATTACK_CANDIDATE",
  "attack_type": "DoS",
  "confidence": 1.0,
  "anomaly_score": 0.65934,
  "anomaly_flag": true,
  "investigation_required": true,
  "priority": "HIGH",
  "next_step": "HAND_OFF_TO_INVESTIGATION",
  "reason": "rf predicted DoS with confidence 1.00 (>= 0.90); the anomaly detector flagged the flow (score 0.6593 > threshold 0.6204). Further investigation is required to confirm or rule out an attack.",
  "rules_fired": ["attack_prediction_high_confidence", "anomaly_corroborates"],
  "evidence": [{"field": "prediction", "value": "DoS", "source": "rf"}, "..."],
  "validation_errors": [],
  "agent": "detection_agent",
  "agent_version": "1.0.0"
}
```

## 6. LangGraph workflow

```
START → receive → validate ──valid──→ evaluate_signals → decide → END
                          └─invalid─→ reject ─────────────────────→ END
```

| Node | Reads | Writes |
| --- | --- | --- |
| `receive` | `raw_input` | normalised `raw_input`, empty working fields |
| `validate` | `raw_input` | `detection` (typed `DetectionResult`) or `validation_errors` |
| `evaluate_signals` | `detection` | `signals` (attack?, confidence band, anomaly flag, thresholds) |
| `decide` | `detection`, `signals` | `decision` (`AgentDecision`) |
| `reject` | `validation_errors` | `decision` with status INVALID_INPUT |

The state is a `TypedDict` (`DetectionAgentState`), and every value in it is a Pydantic model. The graph
contains only these five nodes; there are no placeholder Investigation, Threat-Intel or Report nodes.
Entry point: `run_detection_agent(result) -> AgentDecision`.

## 7. Code layout

| Path | Contents |
| --- | --- |
| `ml/configs/stage3.yaml` | Seeds, caps, weights, model parameters, selection rule, anomaly settings |
| `src/ml/stage3/` | `train_eval.py`, `anomaly.py`, `temporal.py`, `set_detector.py`, `metrics.py`, `models.py` |
| `src/cybersentry/detection/` | `schemas.py` (`DetectionResult`), `service.py` (`DetectionService`) |
| `src/cybersentry/agents/detection_agent/` | `schemas.py` (state, decision), `policy.py` (rules), `graph.py` (LangGraph) |
| `src/api/v1.py` | `/api/v1` routes, mounted in `src/api/main.py` |
| `ml/models/stage3/`, `ml/results/stage3/` | Trained models and results (generated, not in git) |

Training code sits in `src/ml/stage3/` because it reuses the existing data loaders in `src/ml/`.
`ai-engine/` holds no code: its name is not an importable Python package, and the Stage 0 test requires it to
stay a skeleton. The agent therefore lives in the existing `src/cybersentry/agents/` package.

## 8. Current limitations

- Models are trained on the 6-class blocked split (no PortScan or DDoS); see `docs/stage3-ml-results.md`.
- Botnet is not detected on the test split; new attack families on later days are mostly missed (temporal check).
- The selected classifier (Random Forest) costs about 8.5 ms per single flow; XGBoost costs about 0.5 ms. Switch with
  `python -m src.ml.stage3.set_detector xgb --reason "..."` if that matters for the demo.
- The agent sees one flow at a time. Without IPs or timestamps in the data it cannot correlate flows, and it does
  not try to.
- Decision thresholds (0.90 / 0.60) are not calibrated against analyst workload.

## 9. Next stage (not implemented)

- Investigation Agent: takes AgentDecisions with `investigation_required = true` and gathers evidence through
  read-only tools (related events, history), within the same LangGraph.
- An event store, so tools have real telemetry to read.
- Threat-intelligence grounding (MITRE ATT&CK) and the Report Agent.
- Optional local LLM (for example via Ollama) for investigation reasoning over retrieved, verified evidence only.
