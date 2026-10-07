# CyberSentry

> **An AI-Powered Cybersecurity Investigation Pipeline**  
> *Academic / Research Prototype for Telemetry-to-Report Incident Analysis*

---

## 1. What CyberSentry Is

**CyberSentry** is an academic/research-oriented cybersecurity system designed to assist Security Operations Center (SOC) analysts in detecting, investigating, correlating, and explaining suspicious network activity.

Rather than relying purely on static rule-based alerting or ungrounded generative AI predictions, CyberSentry pairs rigorous **machine learning detection** (supervised classification and unsupervised anomaly detection) with an **agentic investigation workflow** (orchestrated multi-agent reasoning).

---

## 2. The Problem CyberSentry Addresses

Modern enterprise and data-center environments face acute cybersecurity challenges:
1. **Alert Fatigue & False Positives:** High-throughput network sensors generate massive volumes of alerts, overwhelming human analysts with noise and benign anomalies.
2. **Context Gap & Slow Triage:** Traditional Intrusion Detection Systems (IDS) flag packet or flow anomalies with cryptic signatures but provide zero contextual explanation, correlation with threat frameworks, or root-cause hypotheses.
3. **Manual, Repetitive Investigations:** Analysts must manually pivot across firewall logs, threat intelligence sources, and flow records to determine whether an anomaly is a genuine breach or a false alarm.
4. **Ungrounded AI Risks:** Off-the-shelf Large Language Models (LLMs) hallucinate indicators of compromise (IoCs) and invent evidence when disconnected from deterministic telemetry.

CyberSentry bridges this divide by using **machine learning** for fast, quantitative pattern recognition and **agentic AI** for explainable, evidence-grounded incident investigation.

---

## 3. Academic & Research Prototype Scope

* **Research Context:** CyberSentry is an academic prototype inspired by high-assurance enterprise and government data-center defense scenarios.
* **Integrity Notice:** The system does **not** claim access to real government or confidential production network traffic.
* **Benchmark Focus:** The primary research dataset is **CIC-IDS2017**, kept strictly outside version control.

---

## 4. High-Level System Architecture

CyberSentry operates across a layered, modular architecture:

```
Network / Security Data (CIC-IDS2017)
            ↓
Data Layer (Auditing, Cleaning, Leakage Prevention)
            ↓
ML Detection Layer (Supervised Classification: Logistic Regression, RF, XGBoost)
            ↓
Anomaly Detection Layer (Unsupervised Isolation Forest)
            ↓
Detection API & Incident Creation (Architecture deferred: Spring Boot / Python ML Service)
            ↓
Agentic Investigation Layer (Detection Agent → Investigation Agent → Report Agent)
            ↓
Threat Intelligence & Grounding (RAG / MITRE ATT&CK / CVE)
            ↓
Deterministic Verification & Risk Assessment
            ↓
Structured Incident Investigation Report
            ↓
Security Operations Dashboard (Deferred Frontend)
```

---

## 5. Review 2 Objective: Vertical Slice

The immediate milestone is **Review 2**, demonstrating a verified, end-to-end **vertical slice** rather than disconnected technologies:

```
CIC-IDS2017 Benchmark Dataset
            ↓
Dataset Audit & Quality Verification
            ↓
Reproducible Preprocessing & Leak-Free Splits
            ↓
Supervised ML Model Comparison (Logistic Regression, Random Forest, XGBoost)
            ↓
Complementary Anomaly Detection (Isolation Forest)
            ↓
Detection API Service
            ↓
Initial Agentic Investigation Workflow:
    • Detection Agent: Identifies IoCs and anomalous flows
    • Investigation Agent: Queries telemetry evidence & correlates behavior
    • Report Agent: Generates structured, analyst-ready incident reports
            ↓
Investigation Report Output & Verification
```

---

## 6. Current Development Stage

* **Current Active Stage:** `STAGE 0 - PROJECT FOUNDATION`
* **Status:** In Progress / Baseline Established.
* **Stage 0 Scope:** Project structure, documentation, Git configuration, and verification tests.
* **Explicit Constraints:** No ML model training, no agent implementations, no RAG, no premature infrastructure (Kafka, Zeek, Suricata, Neo4j), and no frontend development in Stage 0.

---

## 7. Project Structure

```text
CyberSentry/
├── AGENTS.md                    # Project-level specification & behavioral guidelines
├── README.md                    # System overview, problem statement, architecture & roadmap
├── .gitignore                   # Excludes raw/processed data, models, results, caches
├── pyproject.toml               # Packaging metadata & dependency definitions
├── requirements.txt             # Core runtime dependencies
├── requirements-dev.txt         # Development & test tooling
│
├── configs/
│   └── default_config.yaml      # Baseline system & pipeline configurations
├── data/
│   ├── raw/
│   │   └── cic_ids2017/         # Unversioned raw CIC-IDS2017 dataset location
│   ├── processed/               # Cleaned & preprocessed feature sets
│   └── samples/                 # Small sample extracts for deterministic testing
├── ml/
│   ├── src/                     # ML training, feature pipelines, and evaluation modules
│   ├── models/                  # Serialized model artifacts (untracked)
│   ├── notebooks/               # Exploratory data analysis notebooks
│   ├── results/                 # Evaluation metric reports and confusion matrices
│   └── configs/                 # ML-specific experiment configurations
├── ai-engine/
│   ├── agents/                  # LangGraph investigation agents
│   ├── tools/                  # Read-only evidence retrieval tools
│   ├── rag/                    # Threat intelligence grounding modules
│   └── graph/                  # Agent state graphs and orchestration
├── backend/                     # Backend API service (implementation deferred)
├── frontend/                    # Analyst dashboard (implementation deferred)
├── docs/
│   ├── architecture.md          # Multi-layer architectural specification
│   ├── development_rules.md     # Engineering standards & verification protocol
│   └── development-roadmap.md   # Detailed roadmap (Stage 0 through Stage 10)
├── src/cybersentry/             # Cybersentry shared Python packages
└── tests/                       # Automated test suites
```

---

## 8. Verification & Running Tests

Stage 0 foundation verification:

```powershell
$env:PYTHONPATH="src"; python -m unittest discover -s tests -p "test_*.py" -v
```
