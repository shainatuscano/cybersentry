"""
Stage 2: Preprocessing and Feature Engineering Pipeline for CyberSentry.
Processes raw CIC-IDS2017 flow data into clean, leak-free, stratified train/val/test splits.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Constants & Zero-variance columns identified in Stage 1 audit
ZERO_VARIANCE_COLS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]

# Duplicate redundant feature in raw CIC-IDS2017
DUPLICATE_COLS = ["Fwd Header Length.1"]

# Target 15 classes for normalized representation
EXPECTED_CLASSES = [
    "BENIGN",
    "DoS Hulk",
    "PortScan",
    "DDoS",
    "DoS GoldenEye",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "Bot",
    "Web Attack Brute Force",
    "Web Attack XSS",
    "Infiltration",
    "Web Attack Sql Injection",
    "Heartbleed",
]


def normalize_label(label: str) -> str:
    """Normalize label strings to clean ASCII 15-class taxonomy without encoding artifacts.

    Removes Unicode replacement characters (\ufffd) while preserving exact original attack classes.
    """
    s = str(label).strip()
    if "Brute Force" in s:
        return "Web Attack Brute Force"
    if "XSS" in s:
        return "Web Attack XSS"
    if "Sql Injection" in s:
        return "Web Attack Sql Injection"
    if "Hulk" in s:
        return "DoS Hulk"
    if "GoldenEye" in s:
        return "DoS GoldenEye"
    if "slowloris" in s:
        return "DoS slowloris"
    if "Slowhttptest" in s:
        return "DoS Slowhttptest"
    return s


def clean_dataframe_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from columns, normalize target column, and clean labels."""
    df.columns = df.columns.str.strip()
    if "Label" in df.columns:
        df["Label"] = df["Label"].apply(normalize_label)
    return df


def remove_unwanted_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Remove zero-variance and duplicate columns."""
    cols_to_drop = [c for c in ZERO_VARIANCE_COLS + DUPLICATE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop


def clean_numerical_values(
    df: pd.DataFrame,
    train_medians: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """Clean infinite and invalid numerical values without leaking test distributions.

    - Infinite (+/- inf) values in rate features are converted to NaN.
    - Negative duration / IAT artifacts are clipped to 0.0.
    - Negative header length / segment overflow artifacts (< 0) are converted to NaN.
    - Sentinel values in Init_Win_bytes (-1.0) are PRESERVED as valid protocol sentinels.
    - Missing (NaN) values are imputed using deterministic medians computed from TRAIN only.
    """
    stats_record = {
        "infinite_replaced": 0,
        "negative_duration_clipped": 0,
        "header_overflow_cleaned": 0,
    }

    feature_cols = [c for c in df.columns if c != "Label"]

    # 1. Negative durations & IATs (clock skew in packet arrivals) -> clip to 0.0
    time_cols = [
        c for c in [
            "Flow Duration",
            "Flow IAT Mean",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Min",
        ] if c in df.columns
    ]
    for c in time_cols:
        neg_mask = df[c] < 0
        stats_record["negative_duration_clipped"] += int(neg_mask.sum())
        df.loc[neg_mask, c] = 0.0

    # 2. Integer overflow in header lengths & segment sizes -> set to NaN for imputation
    overflow_cols = [
        c for c in [
            "Fwd Header Length",
            "Bwd Header Length",
            "min_seg_size_forward",
        ] if c in df.columns
    ]
    for c in overflow_cols:
        overflow_mask = df[c] < 0
        stats_record["header_overflow_cleaned"] += int(overflow_mask.sum())
        df.loc[overflow_mask, c] = np.nan

    # 3. Handle infinities across all numeric columns -> replace with NaN
    for c in feature_cols:
        inf_mask = np.isinf(df[c])
        stats_record["infinite_replaced"] += int(inf_mask.sum())
        df.loc[inf_mask, c] = np.nan

    # 4. Compute or apply median imputation
    computed_medians = {}
    if train_medians is None:
        # We are on the TRAIN set: learn medians
        for c in feature_cols:
            med_val = float(df[c].median())
            computed_medians[c] = 0.0 if np.isnan(med_val) else med_val
            df[c] = df[c].fillna(computed_medians[c])
    else:
        # We are on VAL or TEST: apply TRAIN-learned medians
        computed_medians = train_medians
        for c in feature_cols:
            fill_val = train_medians.get(c, 0.0)
            df[c] = df[c].fillna(fill_val)

    return df, computed_medians, stats_record


def run_preprocessing_pipeline(
    raw_dir: str = "data/raw/cic_ids2017/MachineLearningCVE",
    output_dir: str = "data/processed",
    results_dir: str = "ml/results",
    docs_dir: str = "docs",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete Stage 2 data preprocessing pipeline."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    res_path = Path(results_dir)
    doc_path = Path(docs_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    res_path.mkdir(parents=True, exist_ok=True)
    doc_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(raw_path.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No raw CSV files found in {raw_path}")

    print(f"Discovered {len(csv_files)} raw CSV files. Reading and cleaning schemas...")

    # Load and clean schemas chunked per file
    dfs = []
    file_records = {}
    total_raw_rows = 0

    for f in csv_files:
        print(f"  Reading {f.name}...")
        df_file = pd.read_csv(f, low_memory=False)
        row_c = len(df_file)
        file_records[f.name] = row_c
        total_raw_rows += row_c

        # Schema and label cleaning
        df_file = clean_dataframe_schema(df_file)
        dfs.append(df_file)

    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total raw records loaded: {len(full_df):,}")

    # Remove zero-variance and duplicate columns
    full_df, removed_cols = remove_unwanted_columns(full_df)
    print(f"Removed {len(removed_cols)} zero-variance and duplicate columns: {removed_cols}")

    # Deduplication across entire dataset
    raw_count_before_dedup = len(full_df)
    full_df = full_df.drop_duplicates()
    dedup_count = len(full_df)
    duplicates_removed = raw_count_before_dedup - dedup_count
    print(f"Removed {duplicates_removed:,} exact duplicate rows. Remaining distinct rows: {dedup_count:,}")

    # Check class distribution post-dedup
    class_counts_post_dedup = full_df["Label"].value_counts().to_dict()
    print("Class distribution after deduplication:")
    for l, c in sorted(class_counts_post_dedup.items(), key=lambda x: x[1], reverse=True):
        print(f"  {l}: {c:,}")

    # Stratified Train (70%) / Validation (15%) / Test (15%) Split
    print(f"\nPerforming stratified split (70% train / 15% val / 15% test, seed={random_seed})...")
    y = full_df["Label"]
    X = full_df.drop(columns=["Label"])

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_seed
    )

    # Second split: 15% Val, 15% Test (50% of the 30% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_seed
    )

    # Reassemble splits with Label
    df_train = X_train.copy()
    df_train["Label"] = y_train

    df_val = X_val.copy()
    df_val["Label"] = y_val

    df_test = X_test.copy()
    df_test["Label"] = y_test

    print(f"  Train split: {len(df_train):,} rows ({len(df_train)/dedup_count*100:.2f}%)")
    print(f"  Val split:   {len(df_val):,} rows ({len(df_val)/dedup_count*100:.2f}%)")
    print(f"  Test split:  {len(df_test):,} rows ({len(df_test)/dedup_count*100:.2f}%)")

    # Clean numerical values: learn medians on TRAIN, apply to VAL and TEST
    print("\nCleaning numerical values (infinities, overflows, imputing from train medians)...")
    df_train, train_medians, train_clean_stats = clean_numerical_values(df_train, train_medians=None)
    df_val, _, val_clean_stats = clean_numerical_values(df_val, train_medians=train_medians)
    df_test, _, test_clean_stats = clean_numerical_values(df_test, train_medians=train_medians)

    # Verify that no NaNs or Infs remain in any split
    for name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        feature_cols = [c for c in df_split.columns if c != "Label"]
        null_count = int(df_split[feature_cols].isna().sum().sum())
        inf_count = int(np.isinf(df_split[feature_cols].values).sum())
        assert null_count == 0, f"{name} split still has {null_count} nulls!"
        assert inf_count == 0, f"{name} split still has {inf_count} infs!"

    # Save processed splits to Parquet
    train_dir = out_path / "train"
    val_dir = out_path / "validation"
    test_dir = out_path / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_parquet = train_dir / "train.parquet"
    val_parquet = val_dir / "validation.parquet"
    test_parquet = test_dir / "test.parquet"

    print("\nSaving partitions to Parquet format...")
    df_train.to_parquet(train_parquet, index=False)
    df_val.to_parquet(val_parquet, index=False)
    df_test.to_parquet(test_parquet, index=False)
    print(f"  Saved {train_parquet} ({train_parquet.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved {val_parquet} ({val_parquet.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Saved {test_parquet} ({test_parquet.stat().st_size / (1024*1024):.2f} MB)")

    # Save small sample extracts (1,000 rows each) for rapid deterministic testing
    samples_dir = Path("data/samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_train = samples_dir / "sample_train.csv"
    sample_test = samples_dir / "sample_test.csv"
    df_train.head(1000).to_csv(sample_train, index=False)
    df_test.head(1000).to_csv(sample_test, index=False)
    print(f"  Saved sample extracts to {sample_train} and {sample_test}")

    # Generate metadata and report
    feature_list = [c for c in df_train.columns if c != "Label"]
    report_dict = {
        "stage": "STAGE 2 - PREPROCESSING & FEATURE ENGINEERING",
        "random_seed": random_seed,
        "source_files": file_records,
        "raw_total_records": total_raw_rows,
        "duplicates_removed": duplicates_removed,
        "distinct_records": dedup_count,
        "removed_features": removed_cols,
        "original_feature_count": 78,
        "final_feature_count": len(feature_list),
        "target_column": "Label",
        "number_of_classes": len(class_counts_post_dedup),
        "classes": sorted(list(class_counts_post_dedup.keys())),
        "split_counts": {
            "train": len(df_train),
            "validation": len(df_val),
            "test": len(df_test),
        },
        "split_proportions": {
            "train": round(len(df_train) / dedup_count, 4),
            "validation": round(len(df_val) / dedup_count, 4),
            "test": round(len(df_test) / dedup_count, 4),
        },
        "class_distribution": {
            "train": df_train["Label"].value_counts().to_dict(),
            "validation": df_val["Label"].value_counts().to_dict(),
            "test": df_test["Label"].value_counts().to_dict(),
        },
        "cleaning_decisions": {
            "zero_variance_features": "Dropped 8 zero-variance constant features (0.0)",
            "duplicate_features": "Dropped Fwd Header Length.1 (exact duplicate of Fwd Header Length)",
            "duplicate_rows": "Dropped exact identical rows before splitting to prevent cross-split train/test leakage",
            "tcp_window_sentinel": "Preserved -1.0 in Init_Win_bytes_forward and Init_Win_bytes_backward as valid TCP window absence sentinels",
            "negative_durations": "Clipped negative durations/IATs to 0.0 (measurement noise/clock skew)",
            "header_overflows": "Replaced negative header lengths (<0) with NaN and imputed with train medians",
            "infinities": "Replaced +inf and -inf with NaN and imputed with train medians",
            "identifier_leakage": "Retained Destination Port as network service feature; IPs and timestamps are absent in raw MachineLearningCVE",
        },
        "final_feature_list": feature_list,
    }

    # Save JSON report
    report_json_path = res_path / "preprocessing_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nSaved machine-readable report to: {report_json_path}")

    # Generate Markdown documentation
    generate_preprocessing_doc(report_dict, doc_path / "preprocessing.md")

    return report_dict


def generate_preprocessing_doc(report: Dict[str, Any], doc_path: Path):
    """Generate docs/preprocessing.md explaining methodology and decisions."""
    lines = []
    lines.append("# CyberSentry Preprocessing & Feature Engineering Methodology\n")
    lines.append("> **Stage:** STAGE 2 - Preprocessing & Feature Engineering  ")
    lines.append("> **Status:** Complete (Fully Reproducible & Leak-Free)  ")
    lines.append(f"> **Random Seed:** `{report['random_seed']}`\n")
    lines.append("---\n")

    lines.append("## 1. Summary of Data Transformations\n")
    lines.append(f"- **Raw Input Rows:** {report['raw_total_records']:,}")
    lines.append(f"- **Duplicate Rows Removed:** {report['duplicates_removed']:,}")
    lines.append(f"- **Distinct Cleaned Rows:** {report['distinct_records']:,}")
    lines.append(f"- **Original Input Features:** {report['original_feature_count']}")
    lines.append(f"- **Features Removed:** {len(report['removed_features'])} ({', '.join(report['removed_features'])})")
    lines.append(f"- **Final Preprocessed Features:** {report['final_feature_count']}")
    lines.append(f"- **Target Classes:** {report['number_of_classes']} classes (preserves full 15-class taxonomy)\n")

    lines.append("## 2. Train / Validation / Test Split Structure\n")
    lines.append("| Split Partition | Record Count | Proportion | File Path |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append(f"| **TRAIN** | {report['split_counts']['train']:,} | {report['split_proportions']['train']*100:.1f}% | `data/processed/train/train.parquet` |")
    lines.append(f"| **VALIDATION** | {report['split_counts']['validation']:,} | {report['split_proportions']['validation']*100:.1f}% | `data/processed/validation/validation.parquet` |")
    lines.append(f"| **TEST** | {report['split_counts']['test']:,} | {report['split_proportions']['test']*100:.1f}% | `data/processed/test/test.parquet` |\n")

    lines.append("### Stratified Class Representation Across Splits:\n")
    lines.append("| Target Class | Total Count | Train Split | Validation Split | Test Split |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")
    for cls in report["classes"]:
        tr_c = report["class_distribution"]["train"].get(cls, 0)
        va_c = report["class_distribution"]["validation"].get(cls, 0)
        te_c = report["class_distribution"]["test"].get(cls, 0)
        tot_c = tr_c + va_c + te_c
        lines.append(f"| **{cls}** | {tot_c:,} | {tr_c:,} | {va_c:,} | {te_c:,} |")

    lines.append("\n*Every minority class (including Heartbleed with 11 total samples) is guaranteed representation in all three splits without leakage.*\n")

    lines.append("## 3. Engineering Decisions & Treatment Rationales\n")
    lines.append("### 3.1 Zero-Variance Features")
    lines.append("The 8 constant features identified in Stage 1 (`Bwd PSH Flags`, `Bwd URG Flags`, `Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`, `Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`) possess zero mutual information across all 2.83M records. They were removed.\n")

    lines.append("### 3.2 Duplicate Feature Elimination")
    lines.append("`Fwd Header Length.1` is an identical duplicate column of `Fwd Header Length` generated during pcap CSV extraction. It was removed as redundant.\n")

    lines.append("### 3.3 Handling Duplicate Rows")
    lines.append(f"Exactly **{report['duplicates_removed']:,}** duplicate rows were removed prior to partitioning. This prevents cross-split data leakage where identical network flow observations occur in both training and test partitions.\n")

    lines.append("### 3.4 Handling Negative and Overflow Values")
    lines.append("1. **TCP Window Sentinels (`Init_Win_bytes_forward`, `Init_Win_bytes_backward`):**")
    lines.append("   - Contains `-1.0` in 35.4% and 50.9% of records.")
    lines.append("   - **Decision:** PRESERVED. In socket flow logging, `-1` represents that no initial TCP window handshake was negotiated (e.g., UDP flows or mid-stream flows). Clipping to zero would corrupt this protocol distinction.")
    lines.append("2. **Negative Duration and Inter-Arrival Times:**")
    lines.append("   - Clock desynchronization in pcap capture causes 115 records with negative duration (down to `-13 microseconds`).")
    lines.append("   - **Decision:** Clipped to `0.0`.")
    lines.append("3. **Integer Overflow in Header Lengths:**")
    lines.append("   - 35 records with negative values down to `-32,212,234,632.0` due to 32-bit signed integer underflow in CICFlowMeter.")
    lines.append("   - **Decision:** Replaced with NaN and imputed using train-learned medians.\n")

    lines.append("### 3.5 Handling Infinite and Missing Values")
    lines.append("Rate features (`Flow Bytes/s`, `Flow Packets/s`) produce infinite values when `Flow Duration == 0`. All `+inf` and `-inf` were mapped to NaN, and imputed deterministically using the median computed **strictly from the training partition**.\n")

    lines.append("### 3.6 Identifier and Leakage Policy")
    lines.append("- Raw IP addresses, Flow IDs, and Timestamps are absent in `MachineLearningCVE`.")
    lines.append("- `Destination Port` is preserved as a numerical feature representing network service layer destinations (e.g. 80, 443, 21, 22).\n")

    lines.append("## 4. Final Feature Schema\n")
    lines.append(f"A total of **{len(report['final_feature_list'])}** features remain:")
    for idx, f in enumerate(report["final_feature_list"]):
        lines.append(f"{idx + 1}. `{f}`")

    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved documentation to: {doc_path}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
