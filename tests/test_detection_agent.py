"""Detection Agent: decision cases, malformed input, determinism, and no unsupported evidence."""
import json

import pytest

from src.cybersentry.agents.detection_agent import (NextStep, Priority, Status, Thresholds, build_graph,
                                                    run_detection_agent)
from src.cybersentry.detection.service import DetectionService

THR = 0.62


def result(prediction="Benign", confidence=0.99, anomaly_score=0.50, **kw):
    flag = anomaly_score > THR
    base = {"event_id": "evt-42", "prediction": prediction, "benign_label": "Benign", "confidence": confidence,
            "anomaly_score": anomaly_score, "anomaly_threshold": THR, "anomaly_flag": flag,
            "is_suspicious": prediction != "Benign" or flag, "classifier": "rf",
            "anomaly_detector": "isolation_forest"}
    return {**base, **kw}


def test_normal_event():
    d = run_detection_agent(result())
    assert d.status is Status.NORMAL and d.priority is Priority.NONE
    assert d.investigation_required is False and d.next_step is NextStep.LOG_ONLY
    assert d.attack_type is None
    assert d.rules_fired == ["benign_prediction_no_anomaly"]


def test_high_confidence_attack():
    d = run_detection_agent(result("DoS", 0.97, 0.80))
    assert d.status is Status.ATTACK_CANDIDATE and d.priority is Priority.HIGH
    assert d.attack_type == "DoS" and d.investigation_required
    assert d.next_step is NextStep.HAND_OFF_TO_INVESTIGATION
    assert d.rules_fired == ["attack_prediction_high_confidence", "anomaly_corroborates"]


def test_high_confidence_attack_without_anomaly_is_still_a_candidate():
    d = run_detection_agent(result("BruteForce", 0.95, 0.40))
    assert d.status is Status.ATTACK_CANDIDATE and d.rules_fired == ["attack_prediction_high_confidence"]


def test_anomaly_only_event():
    d = run_detection_agent(result("Benign", 0.99, 0.75))
    assert d.status is Status.SUSPICIOUS and d.priority is Priority.MEDIUM
    assert d.attack_type is None and d.investigation_required
    assert d.rules_fired == ["anomaly_only_unknown_pattern"]


@pytest.mark.parametrize("confidence,score,status,priority,rule", [
    (0.75, 0.80, Status.ATTACK_CANDIDATE, Priority.MEDIUM,
     "attack_prediction_medium_confidence_corroborated_by_anomaly"),
    (0.75, 0.40, Status.SUSPICIOUS, Priority.MEDIUM, "attack_prediction_medium_confidence"),
    (0.45, 0.80, Status.SUSPICIOUS, Priority.MEDIUM, "attack_prediction_low_confidence_with_anomaly"),
    (0.45, 0.40, Status.SUSPICIOUS, Priority.LOW, "attack_prediction_low_confidence"),
])
def test_lower_confidence_attack_predictions(confidence, score, status, priority, rule):
    d = run_detection_agent(result("WebAttack", confidence, score))
    assert (d.status, d.priority, d.rules_fired) == (status, priority, [rule])
    assert d.investigation_required


def test_low_confidence_benign_prediction_is_suspicious():
    d = run_detection_agent(result("Benign", 0.55, 0.40))
    assert d.status is Status.SUSPICIOUS and d.priority is Priority.LOW
    assert d.rules_fired == ["benign_prediction_low_confidence"]


def test_thresholds_are_configurable():
    strict = Thresholds(high_confidence=0.99, min_confidence=0.9)
    assert run_detection_agent(result("DoS", 0.97, 0.40), strict).status is Status.SUSPICIOUS
    with pytest.raises(ValueError):
        Thresholds(high_confidence=0.5, min_confidence=0.9)


@pytest.mark.parametrize("raw,fragment", [
    ({k: v for k, v in result().items() if k != "confidence"}, "confidence"),
    (result(confidence=1.4), "confidence"),
    (result(confidence="0.97"), "confidence"),                       # strings are not numbers
    (result(anomaly_flag="yes"), "anomaly_flag"),
    (result(anomaly_score=0.9, anomaly_flag=False, is_suspicious=False), "anomaly_flag"),  # contradicts score
    (result("DoS", 0.97, 0.40, is_suspicious=False), "is_suspicious"),                     # contradicts prediction
    (result(class_probabilities={"Benign": 0.2, "DoS": 0.2}), "class_probabilities"),
    (result(source_ip="10.0.0.5"), "source_ip"),                     # unverifiable extra evidence
    (result(prediction=""), "prediction"),
])
def test_malformed_detection_result_is_rejected(raw, fragment):
    d = run_detection_agent(raw)
    assert d.status is Status.INVALID_INPUT and d.next_step is NextStep.REJECT_INPUT
    assert d.investigation_required is False and d.evidence == []
    assert any(fragment in e for e in d.validation_errors), d.validation_errors


@pytest.mark.parametrize("raw", [None, "DoS", [1, 2, 3]])
def test_non_object_input_is_rejected(raw):
    d = run_detection_agent(raw)
    assert d.status is Status.INVALID_INPUT and d.event_id is None


def test_decisions_are_deterministic():
    inputs = [result(), result("DoS", 0.97, 0.80), result("Benign", 0.99, 0.75), result(confidence=2.0)]
    first = [run_detection_agent(r).model_dump() for r in inputs]
    again = [build_graph().invoke({"raw_input": r})["decision"].model_dump() for r in inputs]
    assert first == again


@pytest.mark.parametrize("raw", [result(), result("DoS", 0.97, 0.80), result("WebAttack", 0.45, 0.40),
                                 result("Benign", 0.99, 0.75)])
def test_no_unsupported_evidence(raw):
    d = run_detection_agent(raw)
    # every evidence item is a field of the input, with exactly the input's value
    for ev in d.evidence:
        assert ev.field in raw and ev.value == raw[ev.field]
        assert ev.source in {raw["classifier"], raw["anomaly_detector"], "detection_service"}
    # numbers echoed in the decision come from the input
    assert (d.confidence, d.anomaly_score, d.anomaly_flag) == (raw["confidence"], raw["anomaly_score"],
                                                                raw["anomaly_flag"])
    assert d.attack_type in (None, raw["prediction"])
    # no claims of a confirmed attack, no actions, no invented indicators
    text = json.dumps(d.model_dump(mode="json")).lower()
    for word in ("confirmed attack", "compromised", "block", "firewall", "quarantine", "delete",
                 "ip address", "cve-", "mitre", "t1"):
        assert word not in text, word
    assert d.next_step in (NextStep.LOG_ONLY, NextStep.HAND_OFF_TO_INVESTIGATION)


def test_agent_accepts_output_of_the_detection_service(stage3_artifacts, dos_flow, benign_flow):
    svc = DetectionService(stage3_artifacts)
    attack = run_detection_agent(svc.detect(dos_flow, event_id="e1"))
    assert attack.status in (Status.ATTACK_CANDIDATE, Status.SUSPICIOUS) and attack.attack_type == "DoS"
    normal = svc.detect(benign_flow, event_id="e2")
    assert run_detection_agent(normal).status in (Status.NORMAL, Status.SUSPICIOUS)


def test_graph_has_only_detection_nodes():
    nodes = set(build_graph().get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"receive", "validate", "evaluate_signals", "decide", "reject"}
