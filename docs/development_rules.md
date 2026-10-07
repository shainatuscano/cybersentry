# CyberSentry Development Rules & Engineering Standards

## 1. Incremental Stage Discipline

1. **Strict Stage Adherence:** Only implement the requested stage. Do not jump ahead to future stages before current stage verification is complete.
2. **Minimal and Justified Dependencies:** Do not add dependencies or services (e.g., Kafka, Zeek, Suricata, Neo4j, vector stores) merely because they are popular. Every dependency must have a direct, justified role in the active stage.
3. **No Premature Architecture:** Do not build multi-container microservice meshes or complex web UIs until the core Python data, detection, and agentic pipelines are established and verified.

---

## 2. Data Engineering & Integrity Rules

1. **Dataset Location:** The primary dataset is **CIC-IDS2017**, located outside version control at `data/raw/cic_ids2017/`.
2. **No Unapproved Downloads:** Never download or synthesize dataset files automatically without explicit user authorization.
3. **Mandatory Dataset Audit:** Before any training, every feature, label distribution, missing value, infinite float, duplicate, and potential data leakage vector must be systematically audited and recorded.
4. **No Silent Drops:** Never silently drop columns, rows, or impute values without logging the technical rationale in audit documentation.
5. **Reproducible Splits:** Always use deterministic random seeds and leak-free splitting strategies (Train / Validation / Test).
6. **No Fabricated Statistics:** Never fabricate dataset distributions, metrics, or sample records.

---

## 3. Machine Learning Evaluation Principles

1. **Head-to-Head Fair Comparison:**
   - The initial baseline comparison requires:
     1. Logistic Regression
     2. Random Forest
     3. XGBoost
   - All candidate models must be trained and evaluated on the exact same data splits with identical preprocessing pipelines.
2. **No Preordained Winners:**
   - Do not claim XGBoost is superior prior to obtaining empirical experimental results.
   - Model selection must be strictly supported by measured quantitative data.
3. **Required Metrics:**
   - Precision, Recall, F1-Score (Macro and Weighted)
   - ROC-AUC and PR-AUC
   - False Positive Rate (FPR) and False Negative Rate (FNR)
   - Confusion Matrix breakdown
4. **Complementary Anomaly Detection:**
   - Isolation Forest is intended as a complementary anomaly detector for zero-day/outlier patterns, not an outright replacement for supervised classifiers.

---

## 4. Agentic AI & RAG Rules

1. **Separation of Concerns:**
   - Detection Agent: Ingests model outputs and flags indicators of compromise (IoCs).
   - Investigation Agent: Correlates telemetry events and validates observed facts.
   - Report Agent: Formats findings into actionable, structured incident reports.
2. **Truthfulness and Grounding:**
   - Agents must never invent security evidence.
   - RAG is reserved for domain grounding (MITRE ATT&CK, CVE records, analyst playbooks). RAG must not be used for tasks better handled by database queries or ML classifiers.
3. **Safety & Containment:**
   - Tool execution must remain strictly read-only.
   - No automated destructive remediation (e.g., automated firewall drop commands without analyst oversight).

---

## 5. Verification & Evidence Standard

Every completed stage must produce concrete, verifiable proof before being declared complete:
- Successful test execution output (e.g., `pytest` passing).
- Generated file or directory artifacts.
- Exact commands used for reproduction.
- Explicit disclosure of any assumptions made or incomplete elements.
