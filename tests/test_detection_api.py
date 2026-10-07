"""Detection API: health, valid prediction, invalid requests, response schema, determinism."""
import pytest
from fastapi.testclient import TestClient

from src.api import v1
from src.api.main import app
from src.cybersentry.detection.schemas import DetectionResult
from src.cybersentry.detection.service import DetectionService, FeatureValidationError
from tests.conftest import TINY_CLASSES, TINY_FEATURES


@pytest.fixture
def client(stage3_artifacts, monkeypatch):
    monkeypatch.setenv("CS_STAGE3_MODELS", str(stage3_artifacts))
    v1._load_service.cache_clear()
    yield TestClient(app)
    v1._load_service.cache_clear()


def test_health_reports_loaded_models(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["models_loaded"] is True
    assert body["classifier"] == "rf" and body["classes"] == TINY_CLASSES
    assert body["n_features"] == len(TINY_FEATURES)


def test_health_and_predict_report_missing_models(tmp_path, monkeypatch, benign_flow):
    monkeypatch.setenv("CS_STAGE3_MODELS", str(tmp_path))
    v1._load_service.cache_clear()
    c = TestClient(app)
    r = c.get("/api/v1/health")
    assert r.status_code == 503 and r.json()["models_loaded"] is False
    r = c.post("/api/v1/detection/predict", json={"features": benign_flow})
    assert r.status_code == 503 and "train_eval" in r.json()["detail"]
    v1._load_service.cache_clear()


def test_valid_prediction_matches_schema(client, dos_flow):
    r = client.post("/api/v1/detection/predict", json={"event_id": "evt-1", "features": dos_flow})
    assert r.status_code == 200
    result = DetectionResult.model_validate(r.json())  # strict schema re-validation
    assert result.event_id == "evt-1"
    assert result.prediction == "DoS" and result.is_suspicious
    assert set(result.class_probabilities) == set(TINY_CLASSES)
    assert result.classifier == "rf" and result.anomaly_detector == "isolation_forest"


def test_event_id_is_never_invented(client, benign_flow):
    r = client.post("/api/v1/detection/predict", json={"features": benign_flow})
    assert r.status_code == 200 and r.json()["event_id"] is None


def test_inference_is_deterministic(client, far_out_flow):
    a = client.post("/api/v1/detection/predict", json={"features": far_out_flow}).json()
    b = client.post("/api/v1/detection/predict", json={"features": far_out_flow}).json()
    assert a == b


def test_missing_feature_is_rejected_with_names(client, benign_flow):
    del benign_flow["f2"]
    r = client.post("/api/v1/detection/predict", json={"features": benign_flow})
    assert r.status_code == 422
    assert r.json()["detail"]["missing_features"] == ["f2"]


def test_unknown_feature_is_rejected(client, benign_flow):
    r = client.post("/api/v1/detection/predict", json={"features": {**benign_flow, "Source IP": 1.0}})
    assert r.status_code == 422 and r.json()["detail"]["unknown_features"] == ["Source IP"]


@pytest.mark.parametrize("payload", [
    {"features": {"f0": "fast"}},                     # non-numeric value
    {"features": {}},                                 # empty
    {"event_id": "bad id with spaces", "features": {"f0": 1.0}},
    {"features": {"f0": 1.0}, "action": "block_ip"},  # unexpected top-level field
    {"event_id": "x"},                                # no features
])
def test_malformed_requests_are_rejected(client, payload):
    assert client.post("/api/v1/detection/predict", json=payload).status_code == 422


def test_non_finite_values_are_rejected_by_the_service(stage3_artifacts, benign_flow):
    svc = DetectionService(stage3_artifacts)
    with pytest.raises(FeatureValidationError) as e:
        svc.detect({**benign_flow, "f1": float("nan"), "f2": float("inf")})
    assert e.value.invalid == ["f1", "f2"]


def test_analyze_returns_detection_and_agent_decision(client, dos_flow):
    r = client.post("/api/v1/detection/analyze", json={"event_id": "evt-2", "features": dos_flow})
    assert r.status_code == 200
    body = r.json()
    assert body["detection"]["prediction"] == "DoS"
    dec = body["agent_decision"]
    assert dec["event_id"] == "evt-2"
    assert dec["status"] in {"SUSPICIOUS", "ATTACK_CANDIDATE"} and dec["investigation_required"] is True


def test_features_endpoint_lists_required_names(client):
    assert client.get("/api/v1/detection/features").json() == TINY_FEATURES
