"""Stage 3: temporal / scenario-aware generalisation check.

Usage:
    python -m src.ml.stage3.temporal         # run after train_eval and anomaly

Partitions follow the existing assignment in src/cybersentry/data/temporal_split.py (unchanged):
Mon-Wed -> train, Thursday -> validation, Friday -> test. They are applied to the cleaned rows in
data/processed/ through their `src_file` column. The Friday-afternoon PortScan/DDoS captures are not in
this dataset, so temporal test = Friday morning only.

Why this is not a normal closed-world benchmark: the attack classes of Thursday and Friday never occur on
Mon-Wed, so a classifier trained on Mon-Wed cannot name them. What is valid and reported here:
- Benign false-positive rate on later days (closed-world: Benign is in the training label space).
- For each attack class, the share of its flows flagged as *some* attack (open-world detection), by the
  classifier, by Isolation Forest, and by the combined suspicious rule.
- The same per-class detection rates on the random split, for comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cybersentry.detection.service import suspicious_rule
from src.ml.common import load_meta, load_table
from src.ml.train_models import cap_rows, to_xy

from .anomaly import anomaly_scores, fit_isolation_forest, threshold_from_benign
from .config import MODEL_DIR, RESULT_DIR, load_config, read_detector_meta, write_json
from .metrics import plot_grouped_bars
from .models import MODEL_NAMES, build_model, fit_model, sample_weights

# temporal_split.py imports `cybersentry...` (src layout), so make src/ importable for it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cybersentry.data.temporal_split import TEMPORAL_ASSIGNMENTS  # noqa: E402


def day_partition(src_file: pd.Series) -> pd.Series:
    """Map src_file to the temporal partition. The cleaned data writes '_pcap_ISCX' for '.pcap_ISCX'."""
    lookup = {f.replace(".pcap_ISCX", "_pcap_ISCX"): part
              for part, files in TEMPORAL_ASSIGNMENTS.items() for f in files}
    return src_file.map(lookup)


def detection_by_class(y_names: np.ndarray, pred_attack, flag, susp) -> pd.DataFrame:
    df = pd.DataFrame({"class": y_names, "classifier": pred_attack, "isolation_forest": flag, "combined": susp})
    out = df.groupby("class")[["classifier", "isolation_forest", "combined"]].mean()
    out["support"] = df.groupby("class").size()
    return out.reset_index()


def main() -> None:
    cfg = load_config()
    seed, dcfg = cfg["seed"], cfg["data"]
    out_dir = RESULT_DIR / "temporal"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    det_meta = read_detector_meta()
    classes, benign_name = meta["classes"], meta["classes"][meta["benign_idx"]]
    feats, name = meta["features"], det_meta["selected_model"]

    df = pd.concat([load_table(n) for n in ("train", "val", "test")], ignore_index=True)
    df["partition"] = day_partition(df["src_file"])
    if df["partition"].isna().any():
        raise SystemExit(f"Unmapped source files: {sorted(df.loc[df['partition'].isna(), 'src_file'].unique())}")

    presence = pd.crosstab(df["label"], df["partition"]).reindex(columns=["train", "validation", "test"],
                                                                fill_value=0)
    presence.to_csv(out_dir / "class_presence.csv")
    print("Rows per class and temporal partition:\n", presence.to_string())
    files = df.groupby("partition")["src_file"].unique().apply(sorted).to_dict()

    tr = df[df["partition"] == "train"]
    train_classes = [c for c in classes if (tr["label"] == c).any()]
    if benign_name not in train_classes:
        raise SystemExit("Temporal train has no benign rows")
    idx = {c: i for i, c in enumerate(train_classes)}
    benign = idx[benign_name]
    xtr = tr[feats].to_numpy(dtype=np.float32)
    ytr = tr["label"].map(idx).to_numpy()
    rows = cap_rows(ytr, dcfg["cap_benign"], dcfg["cap_attack"], benign, seed)

    # Same model type and hyperparameters as the selected detector. The temporal validation day has no
    # attack class from the training days, so XGBoost early stopping cannot use it: the number of
    # boosting rounds found on the random split is reused instead.
    params = dict(cfg["models"][name])
    timing = pd.read_csv(RESULT_DIR / "timing.csv").set_index("model")
    if name == "xgb":
        params["early_stopping_rounds"] = None
        if "xgb_best_iteration" in timing.columns and pd.notna(timing.loc["xgb", "xgb_best_iteration"]):
            params["n_estimators"] = int(timing.loc["xgb", "xgb_best_iteration"]) + 1
    model = build_model(name, params, seed)
    print(f"\nTraining {MODEL_NAMES[name]} on temporal train ({len(rows):,} rows, classes {train_classes})...")
    train_s = fit_model(name, model, xtr[rows], ytr[rows], sample_weights(ytr[rows], dcfg["sample_weight"]))

    ben_rows = np.where(ytr == benign)[0]
    acfg = cfg["anomaly"]
    if acfg["max_benign_train"] and len(ben_rows) > acfg["max_benign_train"]:
        ben_rows = np.random.default_rng(seed).choice(ben_rows, acfg["max_benign_train"], replace=False)
    iso = fit_isolation_forest(xtr[ben_rows], acfg["n_estimators"], seed)
    del xtr

    va = df[df["partition"] == "validation"]
    x_va = va[feats].to_numpy(dtype=np.float32)
    s_va = anomaly_scores(iso, x_va)
    thr = threshold_from_benign(s_va[(va["label"] == benign_name).to_numpy()], acfg["threshold_quantile"])

    results, summary = [], {}
    for part in ("validation", "test"):
        sub = df[df["partition"] == part]
        x = x_va if part == "validation" else sub[feats].to_numpy(dtype=np.float32)
        y_names = sub["label"].to_numpy()
        pred = model.predict_proba(x).argmax(axis=1)
        pred_attack = pred != benign
        flag = (s_va if part == "validation" else anomaly_scores(iso, x)) > thr
        susp = suspicious_rule(pred_attack, flag)
        tab = detection_by_class(y_names, pred_attack, flag, susp).assign(partition=part)
        tab["in_training_label_space"] = tab["class"].isin(train_classes)
        results.append(tab)
        is_ben = y_names == benign_name
        summary[part] = {
            "rows": int(len(sub)),
            "benign_fpr_classifier": float(pred_attack[is_ben].mean()),
            "benign_fpr_isolation_forest": float(flag[is_ben].mean()),
            "benign_fpr_combined": float(susp[is_ben].mean()),
            "attack_detection_rate_classifier": float(pred_attack[~is_ben].mean()) if (~is_ben).any() else None,
            "attack_detection_rate_isolation_forest": float(flag[~is_ben].mean()) if (~is_ben).any() else None,
            "attack_detection_rate_combined": float(susp[~is_ben].mean()) if (~is_ben).any() else None,
            "unseen_attack_predicted_as": (pd.Series(np.asarray(train_classes)[pred[~is_ben]])
                                           .value_counts().to_dict() if (~is_ben).any() else {}),
        }
    temporal_tab = pd.concat(results)[["partition", "class", "support", "in_training_label_space",
                                       "classifier", "isolation_forest", "combined"]]

    # Same per-class detection rates on the random split test, from the Stage 3 models.
    xte, yte = to_xy("test", meta)
    clf = joblib.load(MODEL_DIR / f"{name}.joblib")
    iso_r = joblib.load(MODEL_DIR / "iso.joblib")
    pa = clf.predict_proba(xte).argmax(axis=1) != meta["benign_idx"]
    fl = anomaly_scores(iso_r, xte) > det_meta["anomaly_threshold"]
    random_tab = detection_by_class(np.asarray(classes)[yte], pa, fl, suspicious_rule(pa, fl))

    temporal_tab.to_csv(out_dir / "temporal_detection_by_class.csv", index=False)
    random_tab.to_csv(out_dir / "random_split_detection_by_class.csv", index=False)
    day = {"validation": "Thu", "test": "Fri"}
    cmp = (temporal_tab[["partition", "class", "combined", "classifier"]]
           .rename(columns={"classifier": "temporal_classifier", "combined": "temporal_combined"})
           .merge(random_tab[["class", "classifier", "combined"]].rename(
               columns={"classifier": "random_classifier", "combined": "random_combined"}), on="class"))
    cmp["group"] = cmp["class"] + " (" + cmp["partition"].map(day) + ")"
    long = cmp.melt(id_vars="group", value_vars=["random_classifier", "temporal_classifier", "temporal_combined"],
                    var_name="setting", value_name="rate")
    long["setting"] = long["setting"].map({"random_classifier": "Random split, classifier",
                                           "temporal_classifier": "Temporal, classifier",
                                           "temporal_combined": "Temporal, classifier + IF"})
    plot_grouped_bars(long, "group", "setting", "rate", out_dir / "detection_random_vs_temporal.png",
                      "Share of flows flagged as an attack (Benign: false-positive rate)", "Share flagged")
    write_json({"partition_files": files, "classifier": name, "classifier_params": params,
                "temporal_train_classes": train_classes, "train_rows_used": int(len(rows)),
                "train_seconds": round(train_s, 1),
                "anomaly_threshold": thr, "anomaly_threshold_source": "benign rows of temporal validation",
                "summary": summary}, out_dir / "temporal_summary.json")
    print("\nTemporal detection by class:\n", temporal_tab.round(4).to_string(index=False))
    print("\nRandom split detection by class:\n", random_tab.round(4).to_string(index=False))
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()
