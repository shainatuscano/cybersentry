"""Step 1: audit, clean, group labels and split CIC-IDS2017 without leakage.

Usage:
    python -m src.ml.prepare_data                 # reads data/raw/*.csv
    python -m src.ml.prepare_data --block-size 1000

Why a *blocked* split and not a split by day: each CIC-IDS2017 day contains different attacks, so a
day split would remove whole classes from training. A random row split is also wrong, because
neighbouring flows from the same attack run are near-copies. Instead, rows stay in capture order
inside each file, are cut into blocks of consecutive rows, and whole blocks go to train, validation
or test. Classes too small to fill several blocks are split chronologically instead.
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import (ID_COLS, META_COLS, PROC, RAW, REPORTS, SEED, ensure_dirs,  # noqa: E402
                     group_label, save_meta, save_table)


def read_raw(raw_dir):
    """Read every CSV, one file at a time, converting features to float32 as we go (keeps memory low).

    Returns (DataFrame, stats) where stats counts the rows with NaN / infinite values found in the raw files.
    """
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {raw_dir}. Put the CIC-IDS2017 CSV files there.")
    frames = []
    stats = {"rows_with_inf": 0, "rows_with_nan_or_inf": 0}
    for f in files:
        # latin-1 because some labels contain a non-UTF-8 dash byte
        df = pd.read_csv(f, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        df = df[[c for c in df.columns if not c.endswith(".1")]]  # duplicated header column in CIC files
        label_col = next(c for c in df.columns if c.lower() == "label")
        feat = [c for c in df.columns if c != label_col and c.lower() not in ID_COLS]
        num = df[feat].apply(pd.to_numeric, errors="coerce")
        stats["rows_with_inf"] += int(np.isinf(num.to_numpy(dtype="float64")).any(axis=1).sum())
        num = num.replace([np.inf, -np.inf], np.nan)
        stats["rows_with_nan_or_inf"] += int(num.isna().any(axis=1).sum())
        out = num.astype("float32")
        out["label"] = df[label_col].astype(str).str.strip().to_numpy()
        out["src_file"] = f.name
        out["row_in_file"] = np.arange(len(out))
        frames.append(out)
        print(f"  read {f.name}: {len(out):,} rows")
        del df, num
    return pd.concat(frames, ignore_index=True), stats


def blocked_split(df: pd.DataFrame, block_size: int, ratios=(0.70, 0.15, 0.15), seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    key = df["src_file"].astype(str) + ":" + (df["row_in_file"] // block_size).astype(str)
    majority = df.groupby(key)["label"].agg(lambda s: s.value_counts().index[0])
    split_of_block: dict[str, str] = {}
    for _, keys in majority.groupby(majority).groups.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            for k in keys:
                split_of_block[k] = "train"
            continue
        n_val, n_test = max(1, round(n * ratios[1])), max(1, round(n * ratios[2]))
        for i, k in enumerate(keys):
            split_of_block[k] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    split = key.map(split_of_block)

    # Classes too small to fill several blocks: chronological split at row level.
    counts = df["label"].value_counts()
    for lab in counts[counts < 3 * block_size].index:
        order = df[df["label"] == lab].sort_values(["src_file", "row_in_file"]).index
        n = len(order)
        a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        split.loc[order[:a]] = "train"
        split.loc[order[a:b]] = "val"
        split.loc[order[b:]] = "test"
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=str(RAW))
    ap.add_argument("--block-size", type=int, default=1000, help="consecutive rows per split block")
    ap.add_argument("--also-csv", action="store_true",
                    help="also write train/val/test as gzip-compressed CSV (portable, readable anywhere)")
    args = ap.parse_args()
    from pathlib import Path
    ensure_dirs()

    print("Reading raw files...")
    df, raw_stats = read_raw(Path(args.raw_dir))

    feat = [c for c in df.columns if c not in ["label"] + META_COLS]
    audit: dict = {
        "files": sorted(df["src_file"].unique().tolist()),
        "rows_raw": int(len(df)),
        "n_feature_columns_raw": len(feat),
        "raw_label_counts": df["label"].value_counts().to_dict(),
        **raw_stats,
    }

    # --- clean (values are already numeric float32; infinities were turned into NaN while reading)
    df = df.dropna(subset=feat)
    df["label"] = df["label"].map(group_label)

    # Impossible negative values (negative durations, inter-arrival times, header lengths) are corrupt flows.
    # Init_Win_bytes_* legitimately use -1 for "no TCP window", so they are excluded from this rule.
    checked = [c for c in feat if not c.startswith("Init_Win_bytes")]
    bad = (df[checked] < 0).any(axis=1)
    audit["rows_with_invalid_negative_values"] = int(bad.sum())
    audit["invalid_negative_rows_by_label"] = df.loc[bad, "label"].value_counts().to_dict()
    df = df[~bad]

    before = len(df)
    df = df.drop_duplicates(subset=feat + ["label"])
    audit["duplicate_rows_removed"] = int(before - len(df))

    # Identical features with different labels cannot be learned or evaluated fairly: drop all of them.
    conflict = df.duplicated(subset=feat, keep=False)
    audit["conflicting_label_rows_removed"] = int(conflict.sum())
    audit["conflicting_label_rows_by_label"] = df.loc[conflict, "label"].value_counts().to_dict()
    df = df[~conflict].reset_index(drop=True)
    audit["rows_after_cleaning"] = int(len(df))

    # --- split
    df["split"] = blocked_split(df, args.block_size)
    table = pd.crosstab(df["label"], df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    table["total"] = table.sum(axis=1)
    table = table.sort_values("total", ascending=False)
    print("\nClass counts per split:\n", table.to_string())
    for cls, row in table.iterrows():
        for sp in ("train", "val", "test"):
            if row[sp] == 0:
                print(f"WARNING: class '{cls}' has no rows in {sp}")

    train = df[df["split"] == "train"]
    nun = train[feat].nunique()
    const = nun[nun <= 1].index.tolist()
    feat = [c for c in feat if c not in const]
    audit["constant_columns_dropped"] = const
    audit["n_features_final"] = len(feat)

    classes = ["Benign"] + sorted(c for c in df["label"].unique() if c != "Benign")
    keep = feat + ["label"] + META_COLS
    for sp in ("train", "val", "test"):
        part = df.loc[df["split"] == sp, keep].reset_index(drop=True)
        path = save_table(part, sp)
        print(f"saved {path}")
        if args.also_csv:
            csv_path = PROC / f"{sp}.csv.gz"
            part.to_csv(csv_path, index=False, float_format="%.9g", compression="gzip")
            print(f"saved {csv_path}")

    medians = train[feat].median().to_dict()
    save_meta({
        "classes": classes, "benign_idx": 0, "features": feat,
        "medians": {k: float(v) for k, v in medians.items()},
        "block_size": args.block_size, "seed": SEED,
    })

    # --- audit outputs
    audit["class_counts_after_cleaning"] = df["label"].value_counts().to_dict()
    (REPORTS / "audit.json").write_text(json.dumps(audit, indent=2, default=str))
    table.to_csv(REPORTS / "split_distribution.csv")
    counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index, counts.values)
    ax.set_yscale("log")
    ax.set_ylabel("Flows (log scale)")
    ax.set_title("Class distribution after cleaning")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "class_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\nRows: {audit['rows_raw']:,} raw -> {audit['rows_after_cleaning']:,} after cleaning "
          f"({audit['rows_with_nan_or_inf']:,} NaN/inf, "
          f"{audit['rows_with_invalid_negative_values']:,} invalid negative values, "
          f"{audit['duplicate_rows_removed']:,} duplicates, "
          f"{audit['conflicting_label_rows_removed']:,} conflicting-label rows removed).")
    print(f"Features: {audit['n_features_final']} (dropped constant columns: {const})")
    print(f"Audit written to {REPORTS}")


if __name__ == "__main__":
    main()
