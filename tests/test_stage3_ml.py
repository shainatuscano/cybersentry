"""Stage 3 ML: training, loading, prediction shape, labels, metrics, confusion matrix, reproducibility."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.ml.stage3.anomaly import anomaly_scores, fit_isolation_forest, threshold_from_benign
from src.ml.stage3.metrics import evaluate
from src.ml.stage3.models import MODEL_NAMES, build_model, fit_model, sample_weights
from src.ml.stage3.train_eval import select_model
from tests.conftest import TINY_CLASSES, tiny_data

ROOT = Path(__file__).resolve().parents[1]
SMALL = {"lr": {"max_iter": 300}, "rf": {"n_estimators": 20}, "xgb": {"n_estimators": 30, "max_depth": 3}}


def _fit(name, seed=42, weight_mode="balanced"):
    x, y = tiny_data()
    model = build_model(name, SMALL[name], seed=seed, n_jobs=1)
    fit_model(name, model, x, y, sample_weights(y, weight_mode))
    return model


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_model_trains_and_predicts_valid_shape_and_labels(name):
    model = _fit(name)
    xt, yt = tiny_data(seed=3)
    proba = model.predict_proba(xt)
    assert proba.shape == (len(xt), len(TINY_CLASSES))
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    pred = proba.argmax(axis=1)
    assert set(np.unique(pred)) <= set(range(len(TINY_CLASSES)))
    assert (pred == yt).mean() > 0.9  # synthetic classes are well separated


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_training_is_reproducible(name):
    xt, _ = tiny_data(seed=3)
    np.testing.assert_array_equal(_fit(name).predict_proba(xt), _fit(name).predict_proba(xt))


@pytest.mark.parametrize("name", list(MODEL_NAMES))
def test_saved_model_reloads_with_identical_predictions(name, tmp_path):
    model = _fit(name)
    joblib.dump(model, tmp_path / "m.joblib")
    xt, _ = tiny_data(seed=3)
    np.testing.assert_array_equal(model.predict_proba(xt), joblib.load(tmp_path / "m.joblib").predict_proba(xt))


def test_xgboost_native_model_reloads_without_sklearn_wrapper(tmp_path):
    import xgboost as xgb
    model = _fit("xgb")
    model.get_booster().save_model(str(tmp_path / "xgb.json"))
    booster = xgb.Booster()
    booster.load_model(str(tmp_path / "xgb.json"))
    xt, _ = tiny_data(seed=3)
    np.testing.assert_allclose(booster.predict(xgb.DMatrix(xt)), model.predict_proba(xt), atol=1e-6)


def test_lr_scaler_is_fitted_on_training_data_only():
    model = _fit("lr")
    x, _ = tiny_data()
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, x.mean(axis=0), rtol=1e-5, atol=1e-5)


def test_balanced_weights_are_identical_for_every_model_and_equalise_classes():
    _, y = tiny_data()
    y = np.concatenate([y, np.zeros(400, dtype=int)])  # make Benign the majority
    w = sample_weights(y, "balanced")
    totals = [w[y == c].sum() for c in range(3)]
    np.testing.assert_allclose(totals, totals[0])
    assert sample_weights(y, "none") is None
    with pytest.raises(ValueError):
        sample_weights(y, "oversample")


def test_evaluate_known_case():
    y_true = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 2, 2])
    summary, per, cm = evaluate(y_true, y_pred, None, TINY_CLASSES, benign_idx=0)
    assert cm.tolist() == [[3, 1, 0], [1, 1, 0], [0, 0, 2]]
    assert summary["accuracy"] == pytest.approx(6 / 8)
    assert summary["benign_fpr"] == pytest.approx(1 / 4)          # one benign flow called an attack
    assert summary["attack_detection_rate"] == pytest.approx(3 / 4)
    assert summary["attack_false_negatives"] == 1
    row = per.set_index("class").loc["DoS"]
    assert row["recall"] == pytest.approx(0.5) and row["fnr"] == pytest.approx(0.5)
    assert row["fpr"] == pytest.approx(1 / 6)


def test_metric_ranges_and_confusion_matrix_with_probabilities():
    model = _fit("xgb")
    xt, yt = tiny_data(seed=4)
    proba = model.predict_proba(xt)
    summary, per, cm = evaluate(yt, proba.argmax(axis=1), proba, TINY_CLASSES, 0)
    assert cm.shape == (3, 3) and cm.sum() == len(yt)
    for k, v in summary.items():
        if k in ("n_rows", "attack_false_negatives", "benign_false_positives"):
            assert v >= 0
        else:
            assert 0.0 <= v <= 1.0, k
    np.testing.assert_allclose(per["recall"] + per["fnr"], 1.0)
    assert per[["roc_auc", "pr_auc"]].notna().all().all()


def test_absent_class_does_not_count_in_macro_average():
    y = np.array([0, 0, 1, 1])
    summary, per, _ = evaluate(y, y, None, TINY_CLASSES, 0)
    assert summary["macro_f1"] == 1.0
    assert per.set_index("class").loc["WebAttack", "support"] == 0


def test_selection_rule_uses_validation_metric_then_recall_then_cost():
    val = pd.DataFrame({"model": ["lr", "rf", "xgb"], "macro_f1": [0.5, 0.90, 0.80], "macro_recall": [.9, .8, .9]})
    assert select_model(val, {"lr": 1, "rf": 2, "xgb": 3}, "macro_f1", 0.005)[0] == "rf"
    tie = val.assign(macro_f1=[0.5, 0.900, 0.898])
    assert select_model(tie, {"lr": 1, "rf": 2, "xgb": 3}, "macro_f1", 0.005)[0] == "xgb"


def test_isolation_forest_threshold_gives_configured_false_positive_budget():
    x, y = tiny_data(n_per_class=2000)
    iso = fit_isolation_forest(x[y == 0][:1000], n_estimators=50, seed=42)
    s_val = anomaly_scores(iso, x[y == 0][1000:])
    thr = threshold_from_benign(s_val, 0.99)
    assert (s_val > thr).mean() == pytest.approx(0.01, abs=0.002)
    assert anomaly_scores(iso, x[y == 1]).mean() > s_val.mean()  # shifted traffic is more unusual


def test_temporal_partition_covers_every_capture_file():
    from src.ml.stage3.temporal import day_partition
    files = pd.Series(["Monday-WorkingHours_pcap_ISCX.csv", "Tuesday-WorkingHours_pcap_ISCX.csv",
                       "Wednesday-workingHours_pcap_ISCX.csv",
                       "Thursday-WorkingHours-Morning-WebAttacks_pcap_ISCX.csv",
                       "Thursday-WorkingHours-Afternoon-Infilteration_pcap_ISCX.csv",
                       "Friday-WorkingHours-Morning_pcap_ISCX.csv"])
    assert day_partition(files).tolist() == ["train"] * 3 + ["validation"] * 2 + ["test"]


# ------------------------------------------------------------------ trained artifacts (if present)
MODEL_DIR = ROOT / "ml" / "models" / "stage3"
RESULT_DIR = ROOT / "ml" / "results" / "stage3"
needs_artifacts = pytest.mark.skipif(
    not (MODEL_DIR / "detector_meta.json").exists() or not (MODEL_DIR / "iso.joblib").exists(),
    reason="Stage 3 models not trained: run python -m src.ml.stage3.train_eval and src.ml.stage3.anomaly")


@needs_artifacts
def test_trained_detector_loads_and_scores_real_test_flows():
    from src.cybersentry.detection.service import DetectionService
    from src.ml.common import load_table
    svc = DetectionService(MODEL_DIR)
    test = load_table("test")
    sample = pd.concat([test[test["label"] == c].head(3) for c in test["label"].unique()])
    for _, row in sample.iterrows():
        r = svc.detect({f: float(row[f]) for f in svc.features})
        assert r.prediction in svc.classes
        assert 0.0 <= r.confidence <= 1.0
        assert r.anomaly_flag == (r.anomaly_score > r.anomaly_threshold)


@needs_artifacts
def test_reported_test_metrics_are_in_range():
    summary = json.loads((RESULT_DIR / "stage3_summary.json").read_text())
    assert {r["model"] for r in summary["test"]} == {"lr", "rf", "xgb"}
    for row in summary["test"]:
        for k in ("accuracy", "macro_f1", "macro_recall", "weighted_f1", "benign_fpr"):
            assert 0.0 <= row[k] <= 1.0
    assert summary["selection"]["selected_model"] in {"lr", "rf", "xgb"}
