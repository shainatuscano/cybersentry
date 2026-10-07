"""Detection logic, kept free of web-framework code so it can be tested on its own."""
from __future__ import annotations

import time
from functools import lru_cache

import joblib
import numpy as np

from src.ml.common import MODELS, decide, load_meta


class Detector:
    def __init__(self) -> None:
        self.meta = load_meta()
        self.classes: list[str] = self.meta["classes"]
        self.features: list[str] = self.meta["features"]
        self.medians: dict[str, float] = self.meta["medians"]
        if "anomaly_threshold" not in self.meta:
            raise RuntimeError("anomaly_threshold missing. Run: python -m src.ml.anomaly")
        self.threshold: float = self.meta["anomaly_threshold"]
        self.clf = joblib.load(MODELS / "xgb.joblib")
        self.iso = joblib.load(MODELS / "iso.joblib")
        self.iso.n_jobs = 1  # one flow at a time: threads only add overhead

    def vectorize(self, flow: dict) -> tuple[np.ndarray, int]:
        """Turn one flow (feature name -> value) into a model row; missing values use training medians."""
        clean = {str(k).strip(): v for k, v in flow.items()}
        x = np.empty(len(self.features), dtype=np.float32)
        missing = 0
        for i, name in enumerate(self.features):
            try:
                v = float(clean.get(name))
            except (TypeError, ValueError):
                v = float("nan")
            if not np.isfinite(v):
                v = self.medians[name]
                missing += 1
            x[i] = v
        return x.reshape(1, -1), missing

    def detect(self, flow: dict) -> dict:
        t0 = time.perf_counter()
        x, missing = self.vectorize(flow)
        probs = self.clf.predict_proba(x)[0]
        order = np.argsort(probs)[::-1]
        label, prob = self.classes[int(order[0])], float(probs[order[0]])
        anomaly = float(-self.iso.score_samples(x)[0])
        investigate, reason = decide(label, prob, anomaly, self.threshold)
        return {
            "label": label,
            "probability": round(prob, 4),
            "top3": [{"label": self.classes[int(i)], "probability": round(float(probs[i]), 4)}
                     for i in order[:3]],
            "anomaly_score": round(anomaly, 4),
            "anomaly_threshold": round(self.threshold, 4),
            "anomalous": anomaly > self.threshold,
            "investigate": investigate,
            "reason": reason,
            "missing_features": missing,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()
