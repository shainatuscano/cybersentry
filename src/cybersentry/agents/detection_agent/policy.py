"""Deterministic decision policy of the Detection Agent.

No LLM is involved: every step is a comparison of numbers the ML layer already produced, so a fixed
rule table is exact, reproducible and auditable. The table:

| Classifier says          | Confidence     | Anomaly flag | Status            | Priority |
|--------------------------|----------------|--------------|-------------------|----------|
| attack class             | >= high        | any          | ATTACK_CANDIDATE  | HIGH     |
| attack class             | [min, high)    | yes          | ATTACK_CANDIDATE  | MEDIUM   |
| attack class             | [min, high)    | no           | SUSPICIOUS        | MEDIUM   |
| attack class             | < min          | yes          | SUSPICIOUS        | MEDIUM   |
| attack class             | < min          | no           | SUSPICIOUS        | LOW      |
| benign                   | any            | yes          | SUSPICIOUS        | MEDIUM   |
| benign                   | < min          | no           | SUSPICIOUS        | LOW      |
| benign                   | >= min         | no           | NORMAL            | NONE     |

Anything other than NORMAL requires investigation. The agent never states that an attack occurred: an
ATTACK_CANDIDATE is a prediction to be investigated, not a confirmed incident.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...detection.schemas import DetectionResult
from .schemas import AgentDecision, Evidence, NextStep, Priority, Signals, Status


@dataclass(frozen=True)
class Thresholds:
    high_confidence: float = 0.90
    min_confidence: float = 0.60

    def __post_init__(self):
        if not 0.0 <= self.min_confidence <= self.high_confidence <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= min_confidence <= high_confidence <= 1")


def consistency_errors(d: DetectionResult) -> list[str]:
    """Checks beyond the schema: the suspicious flag must follow from the other signals."""
    expected = d.predicted_attack or d.anomaly_flag
    if d.is_suspicious != expected:
        return [f"is_suspicious={d.is_suspicious} contradicts prediction={d.prediction!r} and "
                f"anomaly_flag={d.anomaly_flag} (expected {expected})"]
    return []


def evaluate_signals(d: DetectionResult, t: Thresholds) -> Signals:
    band = ("high" if d.confidence >= t.high_confidence
            else "medium" if d.confidence >= t.min_confidence else "low")
    return Signals(predicted_attack=d.predicted_attack, confidence_band=band, anomaly_flag=d.anomaly_flag,
                   high_confidence_threshold=t.high_confidence, min_confidence_threshold=t.min_confidence)


def _evidence(d: DetectionResult) -> list[Evidence]:
    clf, iso = d.classifier or "supervised_classifier", d.anomaly_detector or "anomaly_detector"
    return [
        Evidence(field="prediction", value=d.prediction, source=clf),
        Evidence(field="confidence", value=d.confidence, source=clf),
        Evidence(field="anomaly_score", value=d.anomaly_score, source=iso),
        Evidence(field="anomaly_threshold", value=d.anomaly_threshold, source=iso),
        Evidence(field="anomaly_flag", value=d.anomaly_flag, source=iso),
        Evidence(field="is_suspicious", value=d.is_suspicious, source="detection_service"),
    ]


def decide(d: DetectionResult, s: Signals) -> AgentDecision:
    clf = d.classifier or "the supervised classifier"
    conf_txt = f"confidence {d.confidence:.2f}"
    anom_txt = (f"the anomaly detector flagged the flow (score {d.anomaly_score:.4f} > threshold "
                f"{d.anomaly_threshold:.4f})" if d.anomaly_flag else
                f"the anomaly detector did not flag the flow (score {d.anomaly_score:.4f} <= threshold "
                f"{d.anomaly_threshold:.4f})")
    rules: list[str] = []

    if s.predicted_attack:
        if s.confidence_band == "high":
            status, prio = Status.ATTACK_CANDIDATE, Priority.HIGH
            rules.append("attack_prediction_high_confidence")
            lead = f"{clf} predicted {d.prediction} with {conf_txt} (>= {s.high_confidence_threshold:.2f})"
        elif s.confidence_band == "medium":
            lead = (f"{clf} predicted {d.prediction} with {conf_txt} (between {s.min_confidence_threshold:.2f} "
                    f"and {s.high_confidence_threshold:.2f})")
            if s.anomaly_flag:
                status, prio = Status.ATTACK_CANDIDATE, Priority.MEDIUM
                rules.append("attack_prediction_medium_confidence_corroborated_by_anomaly")
            else:
                status, prio = Status.SUSPICIOUS, Priority.MEDIUM
                rules.append("attack_prediction_medium_confidence")
        else:
            lead = f"{clf} predicted {d.prediction} with low {conf_txt} (< {s.min_confidence_threshold:.2f})"
            status = Status.SUSPICIOUS
            prio = Priority.MEDIUM if s.anomaly_flag else Priority.LOW
            rules.append("attack_prediction_low_confidence" + ("_with_anomaly" if s.anomaly_flag else ""))
        if s.anomaly_flag and "anomaly" not in rules[-1]:
            rules.append("anomaly_corroborates")
    else:
        lead = f"{clf} predicted {d.prediction} (normal traffic) with {conf_txt}"
        if s.anomaly_flag:
            status, prio = Status.SUSPICIOUS, Priority.MEDIUM
            rules.append("anomaly_only_unknown_pattern")
        elif s.confidence_band == "low":
            status, prio = Status.SUSPICIOUS, Priority.LOW
            rules.append("benign_prediction_low_confidence")
        else:
            status, prio = Status.NORMAL, Priority.NONE
            rules.append("benign_prediction_no_anomaly")

    investigate = status is not Status.NORMAL
    tail = ("Further investigation is required to confirm or rule out an attack." if investigate
            else "No further investigation is required.")
    return AgentDecision(
        event_id=d.event_id, status=status,
        attack_type=d.prediction if s.predicted_attack else None,
        confidence=d.confidence, anomaly_score=d.anomaly_score, anomaly_flag=d.anomaly_flag,
        investigation_required=investigate, priority=prio,
        next_step=NextStep.HAND_OFF_TO_INVESTIGATION if investigate else NextStep.LOG_ONLY,
        reason=f"{lead}; {anom_txt}. {tail}",
        rules_fired=rules, evidence=_evidence(d))


def reject(raw: dict, errors: list[str]) -> AgentDecision:
    event_id = raw.get("event_id") if isinstance(raw, dict) and isinstance(raw.get("event_id"), str) else None
    return AgentDecision(
        event_id=event_id, status=Status.INVALID_INPUT, attack_type=None, confidence=None,
        anomaly_score=None, anomaly_flag=None, investigation_required=False, priority=Priority.NONE,
        next_step=NextStep.REJECT_INPUT,
        reason="The detection result failed validation, so no detection decision was made.",
        rules_fired=["input_validation_failed"], evidence=[], validation_errors=errors)
