"""Final evaluation on the TEST split. Run this once, after you have finished tuning on validation.

Usage:
    python -m src.ml.evaluate_test --confirm
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from .anomaly import scores
from .common import MODELS, REPORTS, decide_many, load_meta
from .metrics import plot_confusion, summarize
from .train_models import to_xy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true",
                    help="required: acknowledges that the test set is used once, at the very end")
    args = ap.parse_args()
    if not args.confirm:
        print("The test split should be used once, after tuning is finished.\n"
              "Re-run with --confirm when you are ready.")
        return

    meta = load_meta()
    classes, benign = meta["classes"], meta["benign_idx"]
    xte, yte = to_xy("test", meta)
    print(f"Test rows: {len(yte):,}")

    rows, per_class = [], []
    xgb_pred = xgb_prob = None
    for name in meta.get("trained_models", ["lr", "rf", "xgb"]):
        model = joblib.load(MODELS / f"{name}.joblib")
        pred = model.predict(xte)
        s, per = summarize(yte, pred, classes, benign)
        s["model"] = name
        per.insert(0, "model", name)
        rows.append(s)
        per_class.append(per)
        if name == "xgb":
            xgb_pred = pred
            xgb_prob = model.predict_proba(xte).max(axis=1)
            plot_confusion(yte, pred, classes, REPORTS / "test_confusion_xgb.png", "XGBoost, test")

    table = pd.DataFrame(rows)[["model", "accuracy", "macro_f1", "weighted_f1",
                                "attack_detection_rate", "false_positive_rate"]]
    table.to_csv(REPORTS / "test_metrics.csv", index=False)
    pd.concat(per_class).to_csv(REPORTS / "test_per_class.csv", index=False)
    print("\nTest comparison:\n", table.round(4).to_string(index=False))

    iso_path = MODELS / "iso.joblib"
    if iso_path.exists() and "anomaly_threshold" in meta:
        iso = joblib.load(iso_path)
        thr = meta["anomaly_threshold"]
        s_te = scores(iso, xte)
        attack = yte != benign
        res = {
            "iforest_attack_flag_rate": float((s_te[attack] > thr).mean()),
            "iforest_benign_flag_rate": float((s_te[~attack] > thr).mean()),
        }
        if xgb_pred is not None:
            inv = decide_many(xgb_pred, xgb_prob, s_te, thr, benign)
            res["combined_attack_detection_rate"] = float(inv[attack].mean())
            res["combined_false_positive_rate"] = float(inv[~attack].mean())
        pd.DataFrame([res]).to_csv(REPORTS / "test_anomaly_and_combined.csv", index=False)
        print("\nIsolation Forest and combined XGBoost + anomaly rule on test:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
    print(f"\nSaved results to {REPORTS}")


if __name__ == "__main__":
    main()
