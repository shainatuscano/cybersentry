"""Step 3: Isolation Forest anomaly detector, trained on benign traffic only.

Usage:
    python -m src.ml.anomaly
    python -m src.ml.anomaly --max-benign 200000 --quantile 0.99

The threshold is the chosen quantile of anomaly scores on *benign validation* rows, which fixes the
false-positive budget (default about 1%). Only validation data is used here.
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from .common import MODELS, REPORTS, SEED, ensure_dirs, load_meta, load_table, update_meta  # noqa: E402


def scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x)  # higher = more unusual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-benign", type=int, default=200_000)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--quantile", type=float, default=0.99)
    args = ap.parse_args()
    ensure_dirs()

    meta = load_meta()
    feats = meta["features"]
    train, val = load_table("train"), load_table("val")

    benign_train = train[train["label"] == "Benign"]
    if args.max_benign and len(benign_train) > args.max_benign:
        benign_train = benign_train.sample(args.max_benign, random_state=SEED)
    print(f"Fitting Isolation Forest on {len(benign_train):,} benign flows...")
    iso = IsolationForest(n_estimators=args.n_estimators, random_state=SEED, n_jobs=-1)
    iso.fit(benign_train[feats].to_numpy(dtype=np.float32))

    s_val = scores(iso, val[feats].to_numpy(dtype=np.float32))
    is_benign = (val["label"] == "Benign").to_numpy()
    thr = float(np.quantile(s_val[is_benign], args.quantile))

    out = {
        "threshold": thr,
        "quantile": args.quantile,
        "val_benign_flagged_rate": float((s_val[is_benign] > thr).mean()),
        "val_attack_flagged_rate": float((s_val[~is_benign] > thr).mean()),
    }
    if is_benign.any() and (~is_benign).any():
        out["val_auroc_attack_vs_benign"] = float(roc_auc_score(~is_benign, s_val))
    print({k: round(v, 4) for k, v in out.items()})

    per = (pd.DataFrame({"class": val["label"].to_numpy(), "flagged": s_val > thr})
           .groupby("class")["flagged"].agg(["mean", "count"])
           .rename(columns={"mean": "flagged_rate", "count": "support"}).reset_index())
    per.to_csv(REPORTS / "val_anomaly_per_class.csv", index=False)
    print("\nShare of validation flows flagged as anomalous, per class:\n", per.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(s_val.min(), s_val.max(), 80)
    ax.hist(s_val[is_benign], bins=bins, alpha=0.6, density=True, label="Benign")
    ax.hist(s_val[~is_benign], bins=bins, alpha=0.6, density=True, label="Attack")
    ax.axvline(thr, color="k", linestyle="--", label=f"threshold ({args.quantile:.0%} benign)")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Density")
    ax.set_title("Isolation Forest scores, validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "anomaly_hist.png", dpi=150)
    plt.close(fig)

    joblib.dump(iso, MODELS / "iso.joblib", compress=3)
    update_meta(anomaly_threshold=thr, anomaly_quantile=args.quantile)
    print(f"\nSaved {MODELS / 'iso.joblib'}; threshold stored in meta.json")


if __name__ == "__main__":
    main()
