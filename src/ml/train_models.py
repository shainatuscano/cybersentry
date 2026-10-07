"""Step 2: train Logistic Regression, Random Forest and XGBoost on the same data and compare on VALIDATION.

Usage:
    python -m src.ml.train_models
    python -m src.ml.train_models --models lr rf xgb --cap-benign 200000 --cap-attack 60000

All models see exactly the same (capped) training rows. Evaluation here uses the validation split only;
the test split is touched once, later, by evaluate_test.py.
"""
from __future__ import annotations

import argparse
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from .common import MODELS, REPORTS, SEED, ensure_dirs, load_meta, load_table, update_meta  # noqa: E402
from .metrics import plot_confusion, summarize  # noqa: E402


def to_xy(name: str, meta: dict):
    df = load_table(name)
    idx = {c: i for i, c in enumerate(meta["classes"])}
    x = df[meta["features"]].to_numpy(dtype=np.float32)
    y = df["label"].map(idx).to_numpy()
    return x, y


def cap_rows(y: np.ndarray, cap_benign: int, cap_attack: int, benign_idx: int, seed: int) -> np.ndarray:
    """Sub-sample every class to at most its cap, keeping small classes whole."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        rows = np.where(y == c)[0]
        cap = cap_benign if c == benign_idx else cap_attack
        if cap and len(rows) > cap:
            rows = rng.choice(rows, cap, replace=False)
        keep.append(rows)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return keep


def build_models(selected, n_classes, xgb_estimators):
    models = {}
    if "lr" in selected:
        models["lr"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED))
    if "rf" in selected:
        models["rf"] = RandomForestClassifier(
            n_estimators=150, n_jobs=-1, class_weight="balanced_subsample", random_state=SEED)
    if "xgb" in selected:
        from xgboost import XGBClassifier  # imported here so lr/rf work without xgboost
        models["xgb"] = XGBClassifier(
            objective="multi:softprob", n_estimators=xgb_estimators, max_depth=8,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            n_jobs=-1, eval_metric="mlogloss", early_stopping_rounds=20, random_state=SEED)
    return models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lr", "rf", "xgb"], choices=["lr", "rf", "xgb"])
    ap.add_argument("--cap-benign", type=int, default=200_000, help="max benign training rows (0 = all)")
    ap.add_argument("--cap-attack", type=int, default=60_000, help="max training rows per attack class (0 = all)")
    ap.add_argument("--xgb-estimators", type=int, default=500)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()

    meta = load_meta()
    classes, benign = meta["classes"], meta["benign_idx"]
    xtr, ytr = to_xy("train", meta)
    xva, yva = to_xy("val", meta)
    rows = cap_rows(ytr, args.cap_benign, args.cap_attack, benign, SEED)
    xtr, ytr = xtr[rows], ytr[rows]
    print(f"Training rows after capping: {len(ytr):,} | validation rows: {len(yva):,}")
    print("Train class counts:", {c: int(n) for c, n in zip(classes, np.bincount(ytr, minlength=len(classes)))})

    summaries, per_class = [], []
    for name, model in build_models(args.models, len(classes), args.xgb_estimators).items():
        print(f"\n=== {name} ===")
        t0 = time.perf_counter()
        if name == "xgb":
            w = compute_sample_weight("balanced", ytr)
            model.fit(xtr, ytr, sample_weight=w, eval_set=[(xva, yva)], verbose=False)
        else:
            model.fit(xtr, ytr)
        train_s = time.perf_counter() - t0

        n_time = min(20_000, len(yva))
        t0 = time.perf_counter()
        model.predict(xva[:n_time])
        ms_per_1000 = (time.perf_counter() - t0) / n_time * 1000 * 1000
        pred = model.predict(xva)

        s, per = summarize(yva, pred, classes, benign)
        s = {k: float(v) for k, v in s.items()}
        s.update(model=name, train_seconds=round(train_s, 1), ms_per_1000_flows=round(ms_per_1000, 1))
        per.insert(0, "model", name)
        summaries.append(s)
        per_class.append(per)
        joblib.dump(model, MODELS / f"{name}.joblib", compress=3)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})

        if name == "xgb":
            plot_confusion(yva, pred, classes, REPORTS / "val_confusion_xgb.png", "XGBoost, validation")
            imp = pd.Series(model.feature_importances_, index=meta["features"]).nlargest(20)[::-1]
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.barh(imp.index, imp.values)
            ax.set_title("XGBoost: top 20 features by importance")
            fig.tight_layout()
            fig.savefig(REPORTS / "xgb_feature_importance.png", dpi=150)
            plt.close(fig)

    table = pd.DataFrame(summaries)[["model", "accuracy", "macro_f1", "weighted_f1",
                                     "attack_detection_rate", "false_positive_rate",
                                     "train_seconds", "ms_per_1000_flows"]]
    table.to_csv(REPORTS / "val_metrics.csv", index=False)
    pd.concat(per_class).to_csv(REPORTS / "val_per_class.csv", index=False)
    update_meta(trained_models=args.models, cap_benign=args.cap_benign, cap_attack=args.cap_attack)
    print("\nValidation comparison:\n", table.round(4).to_string(index=False))
    print(f"\nSaved models to {MODELS} and tables/plots to {REPORTS}")


if __name__ == "__main__":
    main()
