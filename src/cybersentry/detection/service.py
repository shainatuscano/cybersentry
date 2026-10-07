"""Unified detection service: supervised classifier + Isolation Forest -> DetectionResult.

Models are loaded once from ml/models/stage3/ (written by src.ml.stage3.train_eval and src.ml.stage3.anomaly)
and never retrained here. Inference is deterministic: the same features give the same result.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from .schemas import DetectionResult

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "ml" / "models" / "stage3"


def model_dir_from_env() -> Path:
    return Path(os.environ.get("CS_STAGE3_MODELS", DEFAULT_MODEL_DIR))


def suspicious_rule(predicted_attack: bool | np.ndarray, anomaly_flag: bool | np.ndarray):
    """A flow is suspicious when the classifier predicts any attack class or the anomaly detector flags it."""
    return predicted_attack | anomaly_flag


class FeatureValidationError(ValueError):
    def __init__(self, missing=(), unknown=(), invalid=()):
        self.missing, self.unknown, self.invalid = sorted(missing), sorted(unknown), sorted(invalid)
        parts = []
        if self.missing:
            parts.append(f"missing features: {self.missing}")
        if self.unknown:
            parts.append(f"unknown features: {self.unknown}")
        if self.invalid:
            parts.append(f"non-numeric or non-finite values: {self.invalid}")
        super().__init__("; ".join(parts))

    def to_dict(self) -> dict:
        return {"missing_features": self.missing, "unknown_features": self.unknown,
                "invalid_values": self.invalid}


class ModelsNotAvailable(RuntimeError):
    pass


class DetectionService:
    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else model_dir_from_env()
        meta_path = self.model_dir / "detector_meta.json"
        if not meta_path.exists():
            raise ModelsNotAvailable(f"{meta_path} not found. Run: python -m src.ml.stage3.train_eval "
                                     "and python -m src.ml.stage3.anomaly")
        self.meta = json.loads(meta_path.read_text())
        for key in ("features", "classes", "benign_class", "selected_model", "anomaly_threshold"):
            if key not in self.meta:
                raise ModelsNotAvailable(f"detector_meta.json has no {key!r}; re-run the Stage 3 scripts")
        self.features: list[str] = self.meta["features"]
        self.classes: list[str] = self.meta["classes"]
        self.benign_label: str = self.meta["benign_class"]
        self.classifier_name: str = self.meta["selected_model"]
        self.threshold: float = float(self.meta["anomaly_threshold"])
        self.clf = joblib.load(self.model_dir / f"{self.classifier_name}.joblib")
        self.iso = joblib.load(self.model_dir / "iso.joblib")
        for m in (self.clf, self.iso):  # one flow per call: thread pools only add latency
            est = m.steps[-1][1] if hasattr(m, "steps") else m
            if "n_jobs" in est.get_params():
                est.set_params(n_jobs=1)
        self._index = {name: i for i, name in enumerate(self.features)}

    # ------------------------------------------------------------ input
    def vectorize(self, features: Mapping[str, Any]) -> np.ndarray:
        """Strict: every model feature must be present, numeric and finite; no unknown names."""
        clean = {str(k).strip(): v for k, v in features.items()}
        missing = set(self.features) - clean.keys()
        unknown = clean.keys() - set(self.features)
        invalid = []
        x = np.zeros(len(self.features), dtype=np.float32)
        for name, i in self._index.items():
            if name not in clean:
                continue
            v = clean[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                invalid.append(name)
                continue
            x[i] = v
        if missing or unknown or invalid:
            raise FeatureValidationError(missing, unknown, invalid)
        return x.reshape(1, -1)

    # ------------------------------------------------------------ scoring
    def score(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch scoring: (class probabilities, anomaly scores). Higher anomaly score = more unusual."""
        return self.clf.predict_proba(x), -self.iso.score_samples(x)

    def detect(self, features: Mapping[str, Any], event_id: str | None = None) -> DetectionResult:
        x = self.vectorize(features)
        proba, anomaly = self.score(x)
        proba, anomaly = proba[0], float(anomaly[0])
        idx = int(np.argmax(proba))
        prediction = self.classes[idx]
        probs = {c: round(float(p), 6) for c, p in zip(self.classes, proba)}
        # The flag is computed from the same rounded values the result reports, so it always matches them.
        score, threshold = round(anomaly, 6), round(self.threshold, 6)
        flag = score > threshold
        return DetectionResult(
            event_id=event_id,
            prediction=prediction,
            benign_label=self.benign_label,
            confidence=probs[prediction],
            class_probabilities=probs,
            anomaly_score=score,
            anomaly_threshold=threshold,
            anomaly_flag=flag,
            is_suspicious=bool(suspicious_rule(prediction != self.benign_label, flag)),
            classifier=self.classifier_name,
            anomaly_detector="isolation_forest",
        )
