# CyberSentry Development Roadmap (Stage 0 through Stage 10)

## Overview & Progression Policy

Development in CyberSentry follows an incremental, evidence-verified discipline. Each stage must be implemented, tested, and verified before proceeding to subsequent stages. Future stages are explicitly **unimplemented** during earlier phases.

The primary near-term milestone is **Review 2** (achieved across Stages 0 through 7).

---

## Stage Breakdown

### Stage 0: Project Foundation (Current Active Stage)
- **Status:** Baseline Established
- **Objectives:**
  - Establish repository directory hierarchy.
  - Define project specifications and guidelines in [AGENTS.md](file:///c:/Users/kaush/OneDrive/Documents/CyberSentry/AGENTS.md).
  - Configure Git rules and comprehensive [.gitignore](file:///c:/Users/kaush/OneDrive/Documents/CyberSentry/.gitignore) protecting datasets, models, environments, and secrets.
  - Establish baseline environment configuration ([pyproject.toml](file:///c:/Users/kaush/OneDrive/Documents/CyberSentry/pyproject.toml), [requirements.txt](file:///c:/Users/kaush/OneDrive/Documents/CyberSentry/requirements.txt), [configs/default_config.yaml](file:///c:/Users/kaush/OneDrive/Documents/CyberSentry/configs/default_config.yaml)).
  - Document system architecture and multi-stage roadmap.
  - Implement automated foundation verification test suite.
- **Constraints:** No ML training, no agents, no RAG, no premature services.

---

### Stage 1: Dataset Ingestion & Quality Audit (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Place and inspect the CIC-IDS2017 dataset at `data/raw/cic_ids2017/`.
  - Perform exhaustive audit: row/column counts, data types, missing values, infinite float values, and duplicate records.
  - Inspect class label balance (Benign vs. Attack classes).
  - Identify and record potential data leakage vectors (e.g., source IP, flow timestamps).
  - Produce comprehensive, non-fabricated dataset audit documentation.

---

### Stage 2: Reproducible Data Preprocessing & Feature Engineering (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Clean missing and infinite float values using documented, justified strategies.
  - Construct leak-free feature scaling and encoding pipelines.
  - Implement reproducible, stratified train/validation/test dataset splits with fixed random seeds.
  - Persist processed partitions in `data/processed/` and generate deterministic test samples in `data/samples/`.

---

### Stage 3: Supervised Machine Learning Model Comparison (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Train and evaluate baseline candidate models on identical data splits:
    1. Logistic Regression
    2. Random Forest
    3. XGBoost
  - Compute standardized metrics: Precision, Recall, F1-Score (Macro and Weighted), ROC-AUC, PR-AUC, FPR, FNR, and Confusion Matrix.
  - Select best-performing model based solely on empirical evidence without preordained bias.
  - Store model artifacts in `ml/models/` and metrics in `ml/results/`.

---

### Stage 4: Complementary Anomaly Detection (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Train an unsupervised Isolation Forest model on benign flow baselines.
  - Calibrate anomaly score thresholds for outlier detection.
  - Validate that Isolation Forest acts as a complementary detector for zero-day/unknown attack patterns.

---

### Stage 5: Detection Service & API Architecture (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Determine and implement the backend/API architecture (Standalone Python ML service or Spring Boot application coordinating with a Python ML/AI service).
  - Implement inference endpoints for batch and simulated flow event streams.
  - Package model outputs and anomaly scores into structured incident objects.

---

### Stage 6: Initial Agentic Investigation Workflow (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Implement the core Review 2 agent workflow using LangGraph:
    - **Detection Agent:** Ingests alerts and extracts IoCs.
    - **Investigation Agent:** Executes read-only queries against flow telemetry to correlate behavior.
    - **Report Agent:** Synthesizes findings into a structured incident report.
  - Enforce read-only tools and zero hallucination of security evidence.

---

### Stage 7: Review 2 Vertical Slice Integration (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Execute end-to-end vertical slice from raw CIC-IDS2017 telemetry to final investigation report.
  - Validate inter-stage data integrity and latency.
  - Produce complete Review 2 demonstration artifacts and verification reports.

---

### Stage 8: Threat Intelligence Grounding & RAG PoC (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Introduce vector-based retrieval for MITRE ATT&CK techniques, CVEs, and NIST guidance.
  - Ground agentic explanations in authoritative threat intelligence frameworks.
  - Maintain strict separation between observed telemetry evidence and retrieved external knowledge.

---

### Stage 9: Security Operations Dashboard & Frontend (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Develop an analyst-facing React/TypeScript dashboard.
  - Display alert triage feeds, agent investigation timelines, and incident reports.
  - Provide human-in-the-loop analyst feedback controls.

---

### Stage 10: Production Hardening & Future Infrastructure (Planned)
- **Status:** Unimplemented
- **Objectives:**
  - Evaluate integration with enterprise event streaming (Apache Kafka).
  - Ingest live network sensor telemetry (Zeek, Suricata).
  - Implement multi-hop correlation graph queries via Neo4j.
  - Conduct full security auditing and pipeline resilience testing.
