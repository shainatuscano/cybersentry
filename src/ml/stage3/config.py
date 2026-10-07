"""Paths and configuration for Stage 3."""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

from src.ml.common import WORK

CONFIG_PATH = Path(__file__).resolve().parents[3] / "ml" / "configs" / "stage3.yaml"
# CS_STAGE3_MODELS / CS_STAGE3_RESULTS let tests and smoke runs write somewhere else.
MODEL_DIR = Path(os.environ.get("CS_STAGE3_MODELS", WORK / "ml" / "models" / "stage3"))
RESULT_DIR = Path(os.environ.get("CS_STAGE3_RESULTS", WORK / "ml" / "results" / "stage3"))
DETECTOR_META = "detector_meta.json"


def load_config(path: Path | None = None) -> dict:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float))


def read_detector_meta(model_dir: Path | None = None) -> dict:
    p = (model_dir or MODEL_DIR) / DETECTOR_META
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run: python -m src.ml.stage3.train_eval")
    return json.loads(p.read_text())


def update_detector_meta(model_dir: Path | None = None, **kw) -> dict:
    d = model_dir or MODEL_DIR
    p = d / DETECTOR_META
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(kw)
    d.mkdir(parents=True, exist_ok=True)
    write_json(meta, p)
    return meta
