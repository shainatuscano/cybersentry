"""Stage 3: train Logistic Regression, Random Forest and XGBoost under one protocol, pick the primary
detector on VALIDATION, then evaluate all three on the same held-out TEST split.

Usage:
    python -m src.ml.stage3.train_eval
    python -m src.ml.stage3.train_eval --models lr rf xgb

Protocol (identical for every model): same capped training rows, same 6-class label space, same
sample weights (config `data.sample_weight`), same seed, same validation and test rows.
Outputs: ml/models/stage3/ (models, label encoder, detector_meta.json) and ml/results/stage3/.
"""
from __future__ import annotations

import argparse
import platform
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.ml.common import load_meta
from src.ml.train_models import cap_rows, to_xy

from .config import MODEL_DIR, RESULT_DIR, ensure_dirs, load_config, update_detector_meta, write_json
from .metrics import evaluate, plot_confusion, plot_grouped_bars
from .models import MODEL_NAMES, build_model, fit_model, sample_weights, set_single_thread

SUMMARY_COLS = ["model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1",
                "attack_detection_rate", "attack_fnr", "benign_fpr",
                "attack_false_negatives", "benign_false_positives",
                "macro_roc_auc_ovr", "macro_pr_auc_ovr", "attack_roc_auc", "attack_pr_auc"]


def predict(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(x)
    return proba.argmax(axis=1), proba


def time_inference(model, x: np.ndarray, repeats: int) -> dict:
    t0 = time.perf_counter()
    model.predict_proba(x)
    batch_s = time.perf_counter() - t0
    set_single_thread(model)
    lat = []
    for i in range(min(repeats, len(x))):
        t = time.perf_counter()
        model.predict_proba(x[i:i + 1])
        lat.append((time.perf_counter() - t) * 1000)
    return {"batch_rows": int(len(x)), "batch_seconds": round(batch_s, 3),
            "ms_per_1000_flows_batch": round(batch_s / len(x) * 1e6, 3),
            "single_flow_ms_median": round(float(np.median(lat)), 3),
            "single_flow_ms_p95": round(float(np.percentile(lat, 95)), 3)}


def select_model(val: pd.DataFrame, cost: dict, metric: str, margin: float) -> tuple[str, str]:
    ranked = val.sort_values(metric, ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    close = ranked[ranked[metric] >= best[metric] - margin]
    if len(close) == 1:
        why = (f"highest validation {metric} ({best[metric]:.4f}); next best "
               f"{ranked.iloc[1]['model']} at {ranked.iloc[1][metric]:.4f}" if len(ranked) > 1 else "only model")
        return best["model"], why
    close = close.assign(cost=close["model"].map(cost)).sort_values(["macro_recall", "cost"],
                                                                     ascending=[False, True])
    pick = close.iloc[0]
    return pick["model"], (f"validation {metric} within {margin} of the best for {list(close['model'])}; "
                           f"tie broken by validation macro recall ({pick['macro_recall']:.4f}) then cost")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb"], choices=list(MODEL_NAMES))
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config()
    seed, dcfg = cfg["seed"], cfg["data"]
    ensure_dirs()

    meta = load_meta()
    classes, benign = meta["classes"], meta["benign_idx"]
    print("Loading splits...")
    xtr, ytr = to_xy("train", meta)
    xva, yva = to_xy("val", meta)
    xte, yte = to_xy("test", meta)
    rows = cap_rows(ytr, dcfg["cap_benign"], dcfg["cap_attack"], benign, seed)
    xtr, ytr = xtr[rows], ytr[rows]
    weight = sample_weights(ytr, dcfg["sample_weight"])
    counts = {c: int(n) for c, n in zip(classes, np.bincount(ytr, minlength=len(classes)))}
    print(f"Training rows: {len(ytr):,} {counts}\nValidation rows: {len(yva):,} | Test rows: {len(yte):,}")

    val_rows, test_rows, per_val, per_test, timing = [], [], [], [], {}
    for name in args.models:
        print(f"\n=== {MODEL_NAMES[name]} ===")
        model = build_model(name, cfg["models"][name], seed)
        train_s = fit_model(name, model, xtr, ytr, weight, xva, yva)
        extra = {}
        if name == "xgb" and getattr(model, "best_iteration", None) is not None:
            extra["xgb_best_iteration"] = int(model.best_iteration)
        if name == "lr":
            extra["lr_n_iter"] = int(model.named_steps["clf"].n_iter_.max())
        print(f"  trained in {train_s:.1f}s {extra}")

        for split, (x, y), out_rows, out_per in (("val", (xva, yva), val_rows, per_val),
                                                  ("test", (xte, yte), test_rows, per_test)):
            pred, proba = predict(model, x)
            summ, per, cm = evaluate(y, pred, proba, classes, benign)
            out_rows.append({"model": name, **summ})
            out_per.append(per.assign(model=name))
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(RESULT_DIR / f"confusion_{name}_{split}.csv")
            if split == "test":
                plot_confusion(cm, classes, RESULT_DIR / f"confusion_{name}_test.png",
                               f"{MODEL_NAMES[name]}: test confusion matrix")
            print(f"  {split}: macro F1 {summ['macro_f1']:.4f} | macro recall {summ['macro_recall']:.4f} | "
                  f"accuracy {summ['accuracy']:.4f} | benign FPR {summ['benign_fpr']:.5f}")

        timing[name] = {"model": name, "train_seconds": round(train_s, 2), **extra,
                        **time_inference(model, xte, cfg["timing"]["single_flow_repeats"])}
        if name in ("rf", "xgb"):
            model.set_params(n_jobs=-1)
        joblib.dump(model, MODEL_DIR / f"{name}.joblib", compress=3)
        if name == "xgb":
            model.get_booster().save_model(str(MODEL_DIR / "xgb_model.json"))

    val_df = pd.DataFrame(val_rows)[SUMMARY_COLS]
    test_df = pd.DataFrame(test_rows)[SUMMARY_COLS]
    timing_df = pd.DataFrame(timing.values())
    val_df.to_csv(RESULT_DIR / "val_metrics.csv", index=False)
    test_df.to_csv(RESULT_DIR / "test_metrics.csv", index=False)
    timing_df.to_csv(RESULT_DIR / "timing.csv", index=False)
    cols = ["model", "class", "support", "precision", "recall", "f1", "fpr", "fnr",
            "false_negatives", "false_positives", "roc_auc", "pr_auc"]
    pd.concat(per_val)[cols].to_csv(RESULT_DIR / "val_per_class.csv", index=False)
    per_test_df = pd.concat(per_test)[cols]
    per_test_df.to_csv(RESULT_DIR / "test_per_class.csv", index=False)

    sel = cfg["selection"]
    cost = {r["model"]: r["ms_per_1000_flows_batch"] for r in timing.values()}
    selected, why = select_model(val_df, cost, sel["metric"], sel["tie_margin"])
    print(f"\nSelected primary detector: {selected} ({why})")

    # Comparison table and plots
    comp = test_df.merge(timing_df[["model", "train_seconds", "ms_per_1000_flows_batch",
                                    "single_flow_ms_median"]], on="model")
    comp.insert(1, "name", comp["model"].map(MODEL_NAMES))
    comp.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    long = comp.melt(id_vars=["name"], value_vars=["macro_f1", "macro_recall", "macro_precision", "weighted_f1"],
                     var_name="metric", value_name="value")
    plot_grouped_bars(long, "metric", "name", "value", RESULT_DIR / "model_comparison.png",
                      "Test split: macro and weighted scores", "Score")
    per_plot = per_test_df.assign(name=per_test_df["model"].map(MODEL_NAMES))
    plot_grouped_bars(per_plot, "class", "name", "recall", RESULT_DIR / "per_class_recall.png",
                      "Test split: recall per class", "Recall")

    summary = {
        "data": {"source": "data/processed blocked split (DATA.md), 6 grouped classes",
                 "train_rows_used": int(len(ytr)), "train_class_counts": counts,
                 "val_rows": int(len(yva)), "test_rows": int(len(yte)),
                 "cap_benign": dcfg["cap_benign"], "cap_attack": dcfg["cap_attack"],
                 "sample_weight": dcfg["sample_weight"]},
        "selection": {"rule": sel, "selected_model": selected, "reason": why},
        "validation": val_df.to_dict(orient="records"),
        "test": test_df.to_dict(orient="records"),
        "timing": list(timing.values()),
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "xgboost": __import__("xgboost").__version__, "machine": platform.machine(),
                        "cpus": __import__("os").cpu_count()},
    }
    write_json(summary, RESULT_DIR / "stage3_summary.json")
    write_json({"classes": classes, "encoding": "label index = position in classes"},
               MODEL_DIR / "label_encoder.json")
    update_detector_meta(
        features=meta["features"], classes=classes, benign_idx=benign, benign_class=classes[benign],
        selected_model=selected, selection_reason=why, trained_models=args.models, seed=seed,
        data_source="data/processed blocked split (DATA.md)")
    print("\nTest comparison:\n", comp.drop(columns=["model"]).round(4).to_string(index=False))
    print(f"\nModels: {MODEL_DIR}\nResults: {RESULT_DIR}")


if __name__ == "__main__":
    main()
