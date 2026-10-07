"""Stage 3: Isolation Forest as a complementary anomaly detector.

Usage:
    python -m src.ml.stage3.anomaly          # run after train_eval

Training data: BENIGN flows of the training split only (capped, seeded). The model learns what normal
traffic looks like; attacks are never shown to it, so it can flag patterns the classifier has no class for.
Score: anomaly_score = -score_samples(x); higher means more unusual.
Threshold: the configured quantile (default 0.99) of anomaly scores on BENIGN VALIDATION flows, i.e. a
false-positive budget of about 1% on normal traffic. The `contamination` parameter is not used for the
decision: sklearn only uses it to place the offset of predict(), which this project does not call.
The test split is used for evaluation only.
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from src.cybersentry.detection.service import suspicious_rule  # noqa: E402
from src.ml.common import load_meta  # noqa: E402
from src.ml.train_models import to_xy  # noqa: E402

from .config import (MODEL_DIR, RESULT_DIR, ensure_dirs, load_config, read_detector_meta,  # noqa: E402
                     update_detector_meta, write_json)
from .metrics import INK, INK_2, SERIES, _style  # noqa: E402


def anomaly_scores(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x)


def fit_isolation_forest(x_benign: np.ndarray, n_estimators: int, seed: int) -> IsolationForest:
    iso = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed, n_jobs=-1)
    return iso.fit(x_benign)


def threshold_from_benign(scores_benign: np.ndarray, quantile: float) -> float:
    return float(np.quantile(scores_benign, quantile))


def flag_report(scores: np.ndarray, y: np.ndarray, classes: list[str], benign: int,
                thr: float) -> tuple[dict, pd.DataFrame]:
    attack = y != benign
    flag = scores > thr
    out = {
        "benign_flag_rate_fpr": float(flag[~attack].mean()) if (~attack).any() else None,
        "attack_flag_rate_recall": float(flag[attack].mean()) if attack.any() else None,
    }
    if attack.any() and (~attack).any():
        out["attack_vs_benign_roc_auc"] = float(roc_auc_score(attack, scores))
        out["attack_vs_benign_pr_auc"] = float(average_precision_score(attack, scores))
        out["attack_prevalence"] = float(attack.mean())
    per = (pd.DataFrame({"class": np.asarray(classes)[y], "flagged": flag})
           .groupby("class")["flagged"].agg(["mean", "sum", "count"])
           .rename(columns={"mean": "flagged_rate", "sum": "flagged", "count": "support"}).reset_index())
    return out, per


def plot_hist(scores, is_benign, thr, quantile, path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(np.quantile(scores, 0.001), np.quantile(scores, 0.999), 80)
    ax.hist(scores[is_benign], bins=bins, density=True, color=SERIES[0], alpha=0.7, label="Benign")
    ax.hist(scores[~is_benign], bins=bins, density=True, color=SERIES[1], alpha=0.7, label="Attack (all classes)")
    ax.axvline(thr, color=INK, linestyle="--", linewidth=1.2)
    ax.text(thr, ax.get_ylim()[1] * 0.95, f" threshold {thr:.3f}\n ({quantile:.0%} of benign val)",
            fontsize=8, color=INK_2, va="top")
    ax.set_xlabel("Anomaly score (higher = more unusual)", color=INK_2)
    ax.set_ylabel("Density", color=INK_2)
    ax.set_title(title, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    acfg, seed = cfg["anomaly"], cfg["seed"]
    ensure_dirs()
    meta = load_meta()
    classes, benign = meta["classes"], meta["benign_idx"]
    det_meta = read_detector_meta()

    xtr, ytr = to_xy("train", meta)
    xva, yva = to_xy("val", meta)
    xte, yte = to_xy("test", meta)
    ben = np.where(ytr == benign)[0]
    rng = np.random.default_rng(seed)
    if acfg["max_benign_train"] and len(ben) > acfg["max_benign_train"]:
        ben = rng.choice(ben, acfg["max_benign_train"], replace=False)
    print(f"Fitting Isolation Forest on {len(ben):,} benign training flows...")
    iso = fit_isolation_forest(xtr[ben], acfg["n_estimators"], seed)
    del xtr

    s_val = anomaly_scores(iso, xva)
    q = acfg["threshold_quantile"]
    thr = threshold_from_benign(s_val[yva == benign], q)
    val_sum, val_per = flag_report(s_val, yva, classes, benign, thr)
    s_te = anomaly_scores(iso, xte)
    te_sum, te_per = flag_report(s_te, yte, classes, benign, thr)
    print(f"Threshold {thr:.4f} ({q:.0%} quantile of benign validation scores)")
    print("Validation:", {k: round(v, 4) for k, v in val_sum.items()})
    print("Test:      ", {k: round(v, 4) for k, v in te_sum.items()})

    # Complementarity on test: what the anomaly flag adds on top of the selected classifier.
    clf_name = det_meta["selected_model"]
    clf = joblib.load(MODEL_DIR / f"{clf_name}.joblib")
    pred = clf.predict_proba(xte).argmax(axis=1)
    attack, pred_attack, flag = yte != benign, pred != benign, s_te > thr
    susp = suspicious_rule(pred_attack, flag)
    combined = {
        "classifier": clf_name,
        "classifier_attack_detection_rate": float(pred_attack[attack].mean()),
        "classifier_benign_fpr": float(pred_attack[~attack].mean()),
        "combined_attack_detection_rate": float(susp[attack].mean()),
        "combined_benign_fpr": float(susp[~attack].mean()),
        "attacks_missed_by_classifier": int((attack & ~pred_attack).sum()),
        "of_which_flagged_by_isolation_forest": int((attack & ~pred_attack & flag).sum()),
        "benign_flows_flagged_only_by_isolation_forest": int((~attack & ~pred_attack & flag).sum()),
    }
    print("Combined on test:", combined)

    val_per.to_csv(RESULT_DIR / "anomaly_val_per_class.csv", index=False)
    te_per.to_csv(RESULT_DIR / "anomaly_test_per_class.csv", index=False)
    write_json({"training": {"data": "benign rows of the training split", "rows": int(len(ben)),
                             "n_estimators": acfg["n_estimators"], "seed": seed, "contamination": "auto (unused)"},
                "threshold": {"value": thr, "quantile_of_benign_validation": q},
                "validation": val_sum, "test": te_sum, "combined_test": combined},
               RESULT_DIR / "anomaly_summary.json")
    plot_hist(s_te, yte == benign, thr, q, RESULT_DIR / "anomaly_score_hist_test.png",
              "Isolation Forest scores on the test split")
    joblib.dump(iso, MODEL_DIR / "iso.joblib", compress=3)
    update_detector_meta(anomaly_threshold=thr, anomaly_quantile=q, anomaly_model="iso.joblib")
    print(f"Saved {MODEL_DIR / 'iso.joblib'}; threshold stored in detector_meta.json")


if __name__ == "__main__":
    main()
