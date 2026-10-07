"""CyberSentry detection API, version 1.

    GET  /api/v1/health                 model status
    POST /api/v1/detection/predict      one flow -> structured ML detection result
    POST /api/v1/detection/analyze      one flow -> detection result + Detection Agent decision

Models are loaded once per process from ml/models/stage3/ (or CS_STAGE3_MODELS) and never retrained here.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.cybersentry.agents.detection_agent import AgentDecision, run_detection_agent
from src.cybersentry.detection.schemas import DetectionResult
from src.cybersentry.detection.service import DetectionService, FeatureValidationError, ModelsNotAvailable

router = APIRouter(prefix="/api/v1")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "event_id": "demo-0001", "features": {"Destination Port": 80, "Flow Duration": 1200, "...": "..."}}})
    event_id: str | None = Field(None, max_length=128, pattern=r"^[A-Za-z0-9._:\-]+$",
                                 description="Optional caller identifier, echoed back unchanged")
    features: dict[str, float] = Field(..., min_length=1,
                                       description="Every model feature (CIC-IDS2017 column name) -> number")


class AnalyzeResponse(BaseModel):
    detection: DetectionResult
    agent_decision: AgentDecision


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    classifier: str | None = None
    selection_reason: str | None = None
    anomaly_detector: str | None = None
    anomaly_threshold: float | None = None
    classes: list[str] = []
    n_features: int = 0
    detail: str | None = None


@lru_cache(maxsize=1)
def _load_service() -> DetectionService:
    return DetectionService()


def get_service() -> DetectionService:
    try:
        return _load_service()
    except ModelsNotAvailable as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _detect(req: PredictRequest, svc: DetectionService) -> DetectionResult:
    try:
        return svc.detect(req.features, event_id=req.event_id)
    except FeatureValidationError as e:
        raise HTTPException(status_code=422, detail={"message": str(e), **e.to_dict()}) from e


@router.get("/health", response_model=HealthResponse)
def health():
    try:
        svc = _load_service()
    except ModelsNotAvailable as e:
        return JSONResponse(status_code=503, content=HealthResponse(
            status="unavailable", models_loaded=False, detail=str(e)).model_dump())
    return HealthResponse(status="ok", models_loaded=True, classifier=svc.classifier_name,
                          selection_reason=svc.meta.get("selection_reason"),
                          anomaly_detector="isolation_forest", anomaly_threshold=svc.threshold,
                          classes=svc.classes, n_features=len(svc.features))


@router.get("/detection/features", response_model=list[str])
def features(svc: DetectionService = Depends(get_service)):
    """The exact feature names a predict request must contain."""
    return svc.features


@router.post("/detection/predict", response_model=DetectionResult)
def predict(req: PredictRequest, svc: DetectionService = Depends(get_service)):
    return _detect(req, svc)


@router.post("/detection/analyze", response_model=AnalyzeResponse)
def analyze(req: PredictRequest, svc: DetectionService = Depends(get_service)):
    detection = _detect(req, svc)
    return AnalyzeResponse(detection=detection, agent_decision=run_detection_agent(detection))
