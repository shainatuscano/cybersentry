"""Structured detection result: the contract between the ML detection layer and the Detection Agent."""
from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class DetectionResult(BaseModel):
    """Output of the ML detection service for one network flow.

    Every value here is produced by the models; nothing is inferred or added afterwards.
    """
    # strict: numbers must be numbers and flags must be booleans (no "0.97" or "yes" coercion).
    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str | None = Field(None, max_length=128,
                                 description="Caller-supplied event identifier, echoed back. Never generated.")
    prediction: str = Field(..., min_length=1, description="Class predicted by the supervised classifier")
    benign_label: str = Field("Benign", min_length=1, description="Name of the normal-traffic class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier probability of `prediction`")
    class_probabilities: dict[str, float] | None = Field(
        None, description="Classifier probability for every class")
    anomaly_score: float = Field(..., allow_inf_nan=False,
                                 description="Isolation Forest score; higher = more unusual")
    anomaly_threshold: float = Field(..., allow_inf_nan=False,
                                     description="Score above which the flow is flagged anomalous")
    anomaly_flag: StrictBool = Field(..., description="anomaly_score > anomaly_threshold")
    is_suspicious: StrictBool = Field(..., description="Predicted attack class OR anomaly_flag")
    classifier: str | None = Field(None, description="Supervised model that produced `prediction`")
    anomaly_detector: str | None = Field(None, description="Anomaly model that produced `anomaly_score`")

    @model_validator(mode="after")
    def _consistent(self) -> "DetectionResult":
        if self.anomaly_flag != (self.anomaly_score > self.anomaly_threshold):
            raise ValueError("anomaly_flag does not match anomaly_score > anomaly_threshold")
        if self.class_probabilities is not None:
            probs = self.class_probabilities
            if self.prediction not in probs:
                raise ValueError("prediction is missing from class_probabilities")
            if any(not (0.0 <= p <= 1.0) or math.isnan(p) for p in probs.values()):
                raise ValueError("class_probabilities must be within [0, 1]")
            if abs(sum(probs.values()) - 1.0) > 1e-3:
                raise ValueError("class_probabilities must sum to 1")
            if abs(probs[self.prediction] - self.confidence) > 1e-3:
                raise ValueError("confidence must equal the probability of prediction")
        return self

    @property
    def predicted_attack(self) -> bool:
        return self.prediction != self.benign_label
