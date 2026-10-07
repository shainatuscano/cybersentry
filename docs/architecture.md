# CyberSentry Architecture

## 1. System Mission & Prototype Scope

**CyberSentry** is a research-oriented AI-powered cybersecurity investigation system designed to assist Security Operations Center (SOC) analysts in detecting, investigating, correlating, and explaining suspicious network activity.

> **Academic Prototype Notice:** CyberSentry is an academic research prototype inspired by enterprise and government data-center security monitoring scenarios. It does **not** claim access to real government or confidential production network traffic. All empirical evaluations use standardized benchmark datasets (primarily CIC-IDS2017).

---

## 2. High-Level Architecture Overview

The CyberSentry architecture is structured into decoupled, modular layers:

```
                  [Raw Network Flow Telemetry (CIC-IDS2017)]
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA LAYER                                                              │
│    • Ingestion & schema validation                                         │
│    • Data auditing (missing, infinite floats, duplicate detection)          │
│    • Reproducible preprocessing & leak-free train/val/test splitting       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. ML DETECTION & ANOMALY DETECTION LAYER                                  │
│    • Supervised Classifiers: Logistic Regression, Random Forest, XGBoost   │
│    • Unsupervised Anomaly Detection: Isolation Forest (Zero-Day/Outliers)  │
│    • Unified quantitative evaluation (F1, Precision, Recall, ROC-AUC, FPR) │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. DETECTION API & BACKEND INTEGRATION LAYER (Architecture Deferred)       │
│    • Ingestion of inference events and alert thresholds                    │
│    • Incident creation & IoC packaging                                     │
│    • Architectural Options: Python ML Service or Spring Boot + Python      │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 4. AGENTIC INVESTIGATION LAYER (LangGraph)                                 │
│    • Detection Agent: Triage alert, isolate indicators & attack hypothesis │
│    • Investigation Agent: Read-only telemetry queries & flow correlation   │
│    • Report Agent: Structured synthesis of findings & risk assessment      │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 5. RAG & THREAT INTELLIGENCE LAYER                                         │
│    • Grounding in MITRE ATT&CK techniques, CVEs, and NIST frameworks       │
│    • Strictly separated from raw event retrieval (No LLM hallucinations)   │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 6. VERIFICATION & INCIDENT REPORT OUTPUT                                   │
│    • Deterministic risk scoring and schema validation                      │
│    • Human-readable and machine-actionable incident report                 │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ 7. SECURITY OPERATIONS DASHBOARD (Future / Post-Review 2)                  │
│    • React/TypeScript analyst interface (Deferred)                         │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer Specifications

### 3.1 Data Layer
- **Source:** CIC-IDS2017 benchmark dataset, kept uncommitted at `data/raw/cic_ids2017/`.
- **Integrity & Auditing:** Systematic verification of feature formats, label distributions, missing values, infinite floats, and duplicate records.
- **Leakage Prevention:** Strict chronological or stratified splitting before any normalization or transformation to guarantee test set independence.
- **Artifacts:** Preprocessed partitions stored under `data/processed/`, with small reproducible slices in `data/samples/`.

### 3.2 ML Detection Layer
- **Objective:** High-throughput, low-latency supervised classification of network traffic flows.
- **Candidate Comparison:** Fair, head-to-head comparison across:
  1. Logistic Regression (linear baseline)
  2. Random Forest (bagging baseline)
  3. XGBoost (gradient boosting candidate)
- **Selection Mandate:** No model is assumed superior *a priori*; selection is strictly driven by quantitative metrics on the held-out test split.
- **Required Metrics:** Precision, Recall, F1-Score (Macro and Weighted), ROC-AUC, PR-AUC, False Positive Rate (FPR), False Negative Rate (FNR), and Confusion Matrices.

### 3.3 Anomaly Detection Layer
- **Model:** Isolation Forest.
- **Role:** Serves as a **complementary** unsupervised detector for zero-day, outlier, or evasive activity that does not match known attack signatures.
- **Separation:** Isolation Forest does not replace supervised classification; rather, its anomaly scores augment incident triage.

### 3.4 Detection API Layer (Architecture Deferred)
- **Purpose:** Bridges the detection models with downstream consumers, receiving telemetry batches, running inference pipelines, and creating structured incident objects.
- **Architectural Flexibility:**
  - *Option A:* Standalone Python ML/AI microservice (e.g., FastAPI).
  - *Option B:* Enterprise Spring Boot backend coordinating with a specialized Python ML/AI service.
  - *Decision Status:* **Deferred**. The backend/API architecture will be decided in a later stage after the core ML and data pipelines are implemented and evaluated.

### 3.5 Agentic Investigation Layer
- **Framework:** Orchestrated multi-agent workflow (LangGraph).
- **Core Agents for Review 2:**
  - **Detection Agent:** Ingests ML detections, evaluates anomaly confidence, and establishes initial incident hypotheses.
  - **Investigation Agent:** Executes read-only queries against flow telemetry to correlate host behavior, ports, temporal patterns, and attack progression.
  - **Report Agent:** Synthesizes verified findings, timeline, and risk level into structured reports.
- **Core Safety Rule:** Agents have strictly **read-only** tool access. No automated destructive remediation (e.g., automated firewall blocks) is permitted. Agents are never allowed to invent security evidence.

### 3.6 RAG & Threat Intelligence Layer
- **Scope:** Grounds the investigation in recognized cybersecurity knowledge:
  - MITRE ATT&CK tactics, techniques, and procedures (TTPs).
  - Common Vulnerabilities and Exposures (CVE) definitions.
  - Incident response playbooks and defense guidelines.
- **Boundary:** RAG is **never** used for tasks better solved by deterministic database queries or ML classifiers. The system strictly distinguishes between retrieved external knowledge and observed empirical evidence.

### 3.7 Future Infrastructure (Deferred)
To maintain incremental development discipline, the following enterprise components are deferred until the core vertical slice is proven:
- **Streaming Pipeline:** Apache Kafka for high-throughput message streaming.
- **Network Sensors:** Zeek and Suricata for live packet capture and network protocol analysis.
- **Graph Database:** Neo4j for large-scale multi-hop entity graph correlation.
- **Vector Database:** Dedicated vector store (e.g., Qdrant) for scalable RAG retrieval.

### 3.8 Frontend / Dashboard (Deferred)
- **Role:** An interactive Security Operations Center (SOC) dashboard.
- **Technology:** React, TypeScript, and modern data visualization libraries.
- **Status:** Deferred to later stages. The Review 2 milestone focuses on terminal/API verification and structured report generation.
