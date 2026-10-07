"""Model definitions and the one fitting protocol shared by all three supervised models."""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

MODEL_NAMES = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


def build_model(name: str, params: dict, seed: int, n_jobs: int = -1):
    if name == "lr":
        # The scaler lives inside the pipeline, so it is fitted on training rows only and the
        # same fitted scaler transforms validation, test and API inputs.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=params.get("max_iter", 1000), C=params.get("C", 1.0),
                                       random_state=seed)),
        ])
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 150), max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1), n_jobs=n_jobs, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            objective="multi:softprob", n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 8), learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8), colsample_bytree=params.get("colsample_bytree", 0.8),
            tree_method="hist", eval_metric="mlogloss",
            early_stopping_rounds=params.get("early_stopping_rounds"), n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"Unknown model {name!r}")


def sample_weights(y: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "balanced":
        return compute_sample_weight("balanced", y)
    if mode in (None, "none"):
        return None
    raise ValueError(f"Unknown sample_weight mode {mode!r}")


def fit_model(name: str, model, x: np.ndarray, y: np.ndarray, weight: np.ndarray | None,
              x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> float:
    """Fit with the same sample weights for every model type; returns training seconds."""
    t0 = time.perf_counter()
    if name == "lr":
        model.fit(x, y, clf__sample_weight=weight)
    elif name == "xgb":
        has_es = model.get_params().get("early_stopping_rounds")
        if has_es and x_val is None:
            model.set_params(early_stopping_rounds=None)
        kw = {"eval_set": [(x_val, y_val)], "verbose": False} if x_val is not None and has_es else {}
        model.fit(x, y, sample_weight=weight, **kw)
    else:
        model.fit(x, y, sample_weight=weight)
    return time.perf_counter() - t0


def set_single_thread(model) -> None:
    """For single-flow latency: thread pools only add overhead for one row."""
    est = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if "n_jobs" in est.get_params():
        est.set_params(n_jobs=1)
