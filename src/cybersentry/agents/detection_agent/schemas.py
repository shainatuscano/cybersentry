"""Typed state and output of the Detection Agent."""
from __future__ import annotations

from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ...detection.schemas import DetectionResult

AGENT_NAME = "detection_agent"
AGENT_VERSION = "1.0.0"


class Status(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ATTACK_CANDIDATE = "ATTACK_CANDIDATE"
    INVALID_INPUT = "INVALID_INPUT"


class Priority(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class NextStep(str, Enum):
    LOG_ONLY = "LOG_ONLY"
    HAND_OFF_TO_INVESTIGATION = "HAND_OFF_TO_INVESTIGATION"
    REJECT_INPUT = "REJECT_INPUT"


class Evidence(BaseModel):
    """One signal the decision used, copied verbatim from the detection result."""
    model_config = ConfigDict(extra="forbid")
    field: str = Field(..., description="Field of the detection result")
    value: Any
    source: str = Field(..., description="Component that produced the value")


class Signals(BaseModel):
    """Deterministic reading of the detection result (no new facts, only comparisons)."""
    model_config = ConfigDict(extra="forbid")
    predicted_attack: bool
    confidence_band: str  # "high" | "medium" | "low"
    anomaly_flag: bool
    high_confidence_threshold: float
    min_confidence_threshold: float


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str | None
    status: Status
    attack_type: str | None = Field(None, description="Predicted attack class; None for benign predictions")
    confidence: float | None
    anomaly_score: float | None
    anomaly_flag: bool | None
    investigation_required: bool
    priority: Priority
    next_step: NextStep
    reason: str
    rules_fired: list[str]
    evidence: list[Evidence]
    validation_errors: list[str] = Field(default_factory=list)
    agent: str = AGENT_NAME
    agent_version: str = AGENT_VERSION


class DetectionAgentState(TypedDict, total=False):
    """LangGraph state. Each node reads what earlier nodes wrote and adds its own key."""
    raw_input: dict[str, Any]
    detection: DetectionResult | None
    validation_errors: list[str]
    signals: Signals | None
    decision: AgentDecision | None
