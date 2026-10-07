"""Shared paths, label grouping, decision rule and small helpers for CyberSentry."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[2]
# CS_WORKDIR lets you point the whole pipeline at another folder (used for smoke tests).
WORK = Path(os.environ.get("CS_WORKDIR", CODE_ROOT))
RAW = WORK / "data" / "raw"
PROC = WORK / "data" / "processed"
MODELS = WORK / "models"
REPORTS = WORK / "reports"

SEED = 42
META_COLS = ["src_file", "row_in_file"]
# Identifier-like columns removed from the model features so the model cannot memorise hosts or times.
ID_COLS = {
    "flow id", "source ip", "src ip", "destination ip", "dst ip",
    "timestamp", "source port", "src port", "simillarhttp",
}


def ensure_dirs() -> None:
    for p in (RAW, PROC, MODELS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- labels
def group_label(raw: str) -> str:
    """Map the original CIC-IDS2017 labels to a small set of attack classes."""
    s = re.sub(r"[^A-Za-z0-9 \-]", " ", str(raw)).strip()  # removes the odd dash byte in 'Web Attack - ...'
    low = s.lower()
    if low.startswith("benign"):
        return "Benign"
    if low.startswith("web attack") or "sql injection" in low or low == "xss":
        return "WebAttack"
    if low in ("ftp-patator", "ssh-patator"):
        return "BruteForce"
    if low.startswith("ddos"):
        return "DDoS"
    if low.startswith("dos") or low == "heartbleed":
        return "DoS"
    if low.startswith("portscan"):
        return "PortScan"
    if low == "bot":
        return "Botnet"
    if low.startswith("infil"):
        return "Infiltration"
    return s


# ---------------------------------------------------------------- tables
def save_table(df: pd.DataFrame, name: str) -> Path:
    """Save as parquet (needs pyarrow); fall back to pickle if pyarrow is missing."""
    try:
        import pyarrow  # noqa: F401
        path = PROC / f"{name}.parquet"
        df.to_parquet(path, index=False)
    except ImportError:
        path = PROC / f"{name}.pkl"
        df.to_pickle(path)
    return path


def load_table(name: str) -> pd.DataFrame:
    pq, pk, cs = PROC / f"{name}.parquet", PROC / f"{name}.pkl", PROC / f"{name}.csv.gz"
    if pq.exists():
        return pd.read_parquet(pq)
    if pk.exists():
        return pd.read_pickle(pk)
    # GitHub blocks files over 100 MB, so the train file may be stored as train.csv.gz.part-00, -01, ...
    parts = sorted(PROC.glob(f"{name}.csv.gz.part-*"))
    if cs.exists() or parts:
        if cs.exists():
            src = cs
        else:
            import io
            src = io.BytesIO(b"".join(p.read_bytes() for p in parts))
        df = pd.read_csv(src, compression="gzip", low_memory=False)
        num = [c for c in df.columns if c not in ("label", "src_file", "row_in_file")]
        df[num] = df[num].astype("float32")
        return df
    raise SystemExit(f"{name} table not found in {PROC}. Run: python -m src.ml.prepare_data")


# ---------------------------------------------------------------- metadata
def load_meta() -> dict:
    p = MODELS / "meta.json"
    if not p.exists():
        raise SystemExit(f"{p} not found. Run: python -m src.ml.prepare_data")
    return json.loads(p.read_text())


def save_meta(meta: dict) -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))


def update_meta(**kw) -> dict:
    meta = load_meta()
    meta.update(kw)
    save_meta(meta)
    return meta


# ---------------------------------------------------------------- decision rule
PROB_THRESHOLD = 0.60


def decide(label: str, prob: float, anomaly: float, thr: float,
           prob_thr: float = PROB_THRESHOLD) -> tuple[bool, str]:
    """Combine XGBoost and Isolation Forest into an investigate / log decision (section 6.5 of the guide)."""
    anomalous = anomaly > thr
    if label != "Benign":
        if prob >= prob_thr:
            return True, "known_attack"
        if anomalous:
            return True, "low_confidence_and_anomalous"
        return False, "low_confidence_attack_not_anomalous"
    if anomalous:
        return True, "unknown_anomaly"
    return False, "benign"


def decide_many(pred_idx: np.ndarray, prob: np.ndarray, anomaly: np.ndarray, thr: float,
                benign_idx: int = 0, prob_thr: float = PROB_THRESHOLD) -> np.ndarray:
    """Vectorised version of decide(); returns a boolean 'investigate' array."""
    attack = pred_idx != benign_idx
    anomalous = anomaly > thr
    return (attack & ((prob >= prob_thr) | anomalous)) | (~attack & anomalous)
