"""CyberSentry detection API.

Run:  uvicorn src.api.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from fastapi import Body, FastAPI, HTTPException

from .service import get_detector

app = FastAPI(title="CyberSentry Detection API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    d = get_detector()
    return {"status": "ok", "n_features": len(d.features), "classes": d.classes}


@app.post("/detect")
def detect(flow: dict = Body(..., description="One network flow: feature name -> value")) -> dict:
    """Score one flow. Feature names are the CIC-IDS2017 column names; missing ones use training medians."""
    if not flow:
        raise HTTPException(status_code=422, detail="Empty flow")
    return get_detector().detect(flow)


@app.post("/detect/batch")
def detect_batch(flows: list[dict] = Body(...)) -> list[dict]:
    if len(flows) > 1000:
        raise HTTPException(status_code=413, detail="At most 1000 flows per request")
    d = get_detector()
    return [d.detect(f) for f in flows]
