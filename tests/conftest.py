"""Shared fixtures: small, real Stage 3 artifacts built from synthetic data, so API and agent tests run
without the dataset. Numbers from these models mean nothing; they only exercise the code paths."""
import json

import joblib
import numpy as np
import pytest

TINY_FEATURES = [f"f{i}" for i in range(5)]
TINY_CLASSES = ["Benign", "DoS", "WebAttack"]


def tiny_data(n_per_class: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for c in range(len(TINY_CLASSES)):
        x = rng.normal(0, 1, size=(n_per_class, len(TINY_FEATURES)))
        if c:
            x[:, c - 1] += 6.0  # each attack class is shifted on its own feature
        xs.append(x)
        ys.append(np.full(n_per_class, c))
    return np.vstack(xs).astype(np.float32), np.concatenate(ys)


@pytest.fixture(scope="session")
def stage3_artifacts(tmp_path_factory):
    from src.ml.stage3.anomaly import anomaly_scores, fit_isolation_forest, threshold_from_benign
    from src.ml.stage3.models import build_model, fit_model

    d = tmp_path_factory.mktemp("stage3_models")
    x, y = tiny_data()
    rf = build_model("rf", {"n_estimators": 20}, seed=42, n_jobs=1)
    fit_model("rf", rf, x, y, None)
    iso = fit_isolation_forest(x[y == 0], n_estimators=50, seed=42)
    xv, yv = tiny_data(seed=1)
    thr = threshold_from_benign(anomaly_scores(iso, xv[yv == 0]), 0.99)
    joblib.dump(rf, d / "rf.joblib")
    joblib.dump(iso, d / "iso.joblib")
    (d / "detector_meta.json").write_text(json.dumps({
        "features": TINY_FEATURES, "classes": TINY_CLASSES, "benign_idx": 0, "benign_class": "Benign",
        "selected_model": "rf", "anomaly_threshold": thr}))
    return d


@pytest.fixture
def benign_flow():
    return {f: 0.0 for f in TINY_FEATURES}


@pytest.fixture
def dos_flow():
    return {**{f: 0.0 for f in TINY_FEATURES}, "f0": 6.0}


@pytest.fixture
def far_out_flow():
    """Benign-looking on f0/f1 but extreme elsewhere: should be flagged by the anomaly detector."""
    return {**{f: 0.0 for f in TINY_FEATURES}, "f3": 40.0, "f4": -40.0}
