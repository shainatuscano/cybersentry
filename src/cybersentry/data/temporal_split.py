"""
Stage 2.1: Temporal and Scenario-Aware Data Split for CyberSentry.
Partitions the CIC-IDS2017 dataset strictly by calendar day / capture session:
- Train: Monday, Tuesday, Wednesday (Baseline Benign, Brute Force, DoS, Heartbleed)
- Validation: Thursday (Web Attacks, Infiltration, Benign)
- Test: Friday (Botnet, PortScan, DDoS, Benign)

Reuses exact Stage 2 cleaning logic; learns imputations from temporal train only;
guarantees zero duplicate leakage across partitions.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from cybersentry.data.preprocess import (
    clean_dataframe_schema,
    remove_unwanted_columns,
    clean_numerical_values,
    EXPECTED_CLASSES,
)


TEMPORAL_ASSIGNMENTS = {
    "train": [
        "Monday-WorkingHours.pcap_ISCX.csv",
        "Tuesday-WorkingHours.pcap_ISCX.csv",
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "validation": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "test": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
}


def compute_row_hash(row_tuple) -> bytes:
    """Compute deterministic MD5 hash for a row to check exact duplicate leakage."""
    return hashlib.md5(str(row_tuple).encode("utf-8")).digest()


def run_temporal_split_pipeline(
    raw_dir: str = "data/raw/cic_ids2017/MachineLearningCVE",
    output_dir: str = "data/processed/temporal",
    results_dir: str = "ml/results",
) -> Dict[str, Any]:
    """Execute complete Stage 2.1 temporal / scenario-aware split pipeline."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    res_path = Path(results_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)

    print("Executing Stage 2.1 Temporal / Scenario-Aware Split Pipeline...")

    # Load and clean schemas per partition
    raw_counts = {}
    partition_dfs = {}

    for split_name, filenames in TEMPORAL_ASSIGNMENTS.items():
        print(f"\nLoading {split_name} partition ({len(filenames)} files)...")
        dfs = []
        raw_c = 0
        for fname in filenames:
            fpath = raw_path / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Source file not found: {fpath}")
            print(f"  Reading {fname}...")
            df_file = pd.read_csv(fpath, low_memory=False)
            raw_c += len(df_file)
            df_file = clean_dataframe_schema(df_file)
            df_file, _ = remove_unwanted_columns(df_file)
            dfs.append(df_file)

        raw_counts[split_name] = raw_c
        combined = pd.concat(dfs, ignore_index=True)
        # Internal deduplication
        dedup_internal = combined.drop_duplicates()
        partition_dfs[split_name] = dedup_internal
        print(f"  {split_name}: {raw_c:,} raw rows -> {len(dedup_internal):,} internal unique rows")

    # Chronological deduplication across partitions to eliminate cross-split duplicate leakage
    # 1. Train keeps all internal unique rows
    df_train = partition_dfs["train"]
    train_hashes = set()
    for row in df_train.itertuples(index=False):
        train_hashes.add(compute_row_hash(row))

    # 2. Validation removes any rows already seen in Train
    df_val = partition_dfs["validation"]
    val_keep_mask = []
    val_hashes = set()
    val_train_duplicates = 0

    for row in df_val.itertuples(index=False):
        h = compute_row_hash(row)
        if h in train_hashes:
            val_keep_mask.append(False)
            val_train_duplicates += 1
        else:
            val_keep_mask.append(True)
            val_hashes.add(h)

    df_val = df_val[val_keep_mask].reset_index(drop=True)
    print(f"\nFiltered {val_train_duplicates:,} rows from validation that appeared in train.")

    # 3. Test removes any rows already seen in Train or Validation
    df_test = partition_dfs["test"]
    test_keep_mask = []
    test_hashes = set()
    test_train_duplicates = 0
    test_val_duplicates = 0

    for row in df_test.itertuples(index=False):
        h = compute_row_hash(row)
        if h in train_hashes:
            test_keep_mask.append(False)
            test_train_duplicates += 1
        elif h in val_hashes:
            test_keep_mask.append(False)
            test_val_duplicates += 1
        else:
            test_keep_mask.append(True)
            test_hashes.add(h)

    df_test = df_test[test_keep_mask].reset_index(drop=True)
    print(f"Filtered {test_train_duplicates:,} rows from test that appeared in train.")
    print(f"Filtered {test_val_duplicates:,} rows from test that appeared in val.")

    # Numerical cleaning & imputation learned strictly from TEMPORAL TRAIN
    print("\nApplying Stage 2 numerical cleaning (learning medians strictly from temporal train)...")
    df_train, train_medians, _ = clean_numerical_values(df_train, train_medians=None)
    df_val, _, _ = clean_numerical_values(df_val, train_medians=train_medians)
    df_test, _, _ = clean_numerical_values(df_test, train_medians=train_medians)

    # Verify no nulls or infs remain
    for name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        feat_cols = [c for c in df_split.columns if c != "Label"]
        null_count = int(df_split[feat_cols].isna().sum().sum())
        inf_count = int(np.isinf(df_split[feat_cols].values).sum())
        assert null_count == 0, f"{name} split has {null_count} nulls!"
        assert inf_count == 0, f"{name} split has {inf_count} infs!"

    # Verify cross-partition exact duplicate counts are strictly 0
    train_final_hashes = set(compute_row_hash(r) for r in df_train.itertuples(index=False))
    val_final_hashes = set(compute_row_hash(r) for r in df_val.itertuples(index=False))
    test_final_hashes = set(compute_row_hash(r) for r in df_test.itertuples(index=False))

    dup_train_val = len(train_final_hashes & val_final_hashes)
    dup_train_test = len(train_final_hashes & test_final_hashes)
    dup_val_test = len(val_final_hashes & test_final_hashes)

    assert dup_train_val == 0, f"Found {dup_train_val} duplicates between train and val!"
    assert dup_train_test == 0, f"Found {dup_train_test} duplicates between train and test!"
    assert dup_val_test == 0, f"Found {dup_val_test} duplicates between val and test!"
    print(f"\nCross-split duplicate verification:")
    print(f"  Train vs Val:  {dup_train_val} exact duplicates")
    print(f"  Train vs Test: {dup_train_test} exact duplicates")
    print(f"  Val vs Test:   {dup_val_test} exact duplicates")

    # Save to Parquet
    train_parquet = out_path / "train.parquet"
    val_parquet = out_path / "validation.parquet"
    test_parquet = out_path / "test.parquet"

    print("\nSaving temporal partitions to Parquet...")
    df_train.to_parquet(train_parquet, index=False)
    df_val.to_parquet(val_parquet, index=False)
    df_test.to_parquet(test_parquet, index=False)
    print(f"  Saved {train_parquet} ({train_parquet.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved {val_parquet} ({val_parquet.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved {test_parquet} ({test_parquet.stat().st_size / (1024*1024):.2f} MB)")

    # Analyze class distributions and absent classes
    all_classes_set = set(EXPECTED_CLASSES)

    train_class_counts = df_train["Label"].value_counts().to_dict()
    val_class_counts = df_val["Label"].value_counts().to_dict()
    test_class_counts = df_test["Label"].value_counts().to_dict()

    train_present = set(train_class_counts.keys())
    val_present = set(val_class_counts.keys())
    test_present = set(test_class_counts.keys())

    train_absent = sorted(list(all_classes_set - train_present))
    val_absent = sorted(list(all_classes_set - val_present))
    test_absent = sorted(list(all_classes_set - test_present))

    feature_list = [c for c in df_train.columns if c != "Label"]

    # Assemble report
    report = {
        "stage": "STAGE 2.1 - TEMPORAL / SCENARIO-AWARE DATA SPLIT",
        "description": "Calendar-day capture session split separating historical training from forward evaluation",
        "source_files": TEMPORAL_ASSIGNMENTS,
        "raw_counts": raw_counts,
        "split_counts": {
            "train": len(df_train),
            "validation": len(df_val),
            "test": len(df_test),
        },
        "total_distinct_records": len(df_train) + len(df_val) + len(df_test),
        "split_proportions": {
            "train": round(len(df_train) / (len(df_train) + len(df_val) + len(df_test)), 4),
            "validation": round(len(df_val) / (len(df_train) + len(df_val) + len(df_test)), 4),
            "test": round(len(df_test) / (len(df_train) + len(df_val) + len(df_test)), 4),
        },
        "feature_count": len(feature_list),
        "preprocessing_method": "Stage 2 deterministic pipeline with imputations learned strictly from temporal train",
        "class_distributions": {
            "train": train_class_counts,
            "validation": val_class_counts,
            "test": test_class_counts,
        },
        "classes_absent": {
            "train": train_absent,
            "validation": val_absent,
            "test": test_absent,
        },
        "duplicate_checks": {
            "train_vs_validation_duplicates": dup_train_val,
            "train_vs_test_duplicates": dup_train_test,
            "validation_vs_test_duplicates": dup_val_test,
            "filtered_cross_split_duplicates": {
                "val_filtered_from_train": val_train_duplicates,
                "test_filtered_from_train": test_train_duplicates,
                "test_filtered_from_val": test_val_duplicates,
            },
        },
        "source_session_separation_evidence": {
            "train_sessions": "Monday (July 3), Tuesday (July 4), Wednesday (July 5)",
            "validation_sessions": "Thursday (July 6) morning and afternoon",
            "test_sessions": "Friday (July 7) morning and afternoon",
            "cross_session_mixing": "None. Partitions are mutually exclusive by source capture file and calendar date.",
        },
        "rationale": (
            "In operational cybersecurity deployments, models trained on historical days must detect threats "
            "on future days without access to future traffic or identical session artifacts. "
            "Assigning Mon-Wed to train, Thu to validation, and Fri to test provides a realistic out-of-time "
            "evaluation of model generalizability across evolving network environments."
        ),
        "limitations": [
            "Certain attack classes only occurred on specific days in the benchmark testbed (e.g., Web Attacks on Thursday only, PortScan/DDoS on Friday only).",
            "Because specific attacks were executed on single days, temporal test evaluates zero-day/novel detection for unseen attack families rather than in-sample supervised classification.",
            "Performance on temporal test must be reported per-class or as anomaly detection since supervised classifiers cannot predict labels never observed in training."
        ],
        "final_feature_list": feature_list,
    }

    report_path = res_path / "temporal_split_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved machine-readable report to: {report_path}")

    return report


if __name__ == "__main__":
    run_temporal_split_pipeline()
