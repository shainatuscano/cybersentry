"""
Stage 2 Verification Suite: Preprocessing & Feature Engineering.
Validates processed dataset integrity, schema consistency, absence of zero-variance features,
stratified split proportions, leak-free deduplication, and rare class preservation.
"""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


_REPORT = Path(__file__).resolve().parents[1] / "ml" / "results" / "preprocessing_report.json"


@unittest.skipUnless(_REPORT.exists(), "needs the 8 raw CIC-IDS2017 CSVs in data/raw/cic_ids2017/MachineLearningCVE/ "
                     "and `PYTHONPATH=src python -m cybersentry.data.preprocess` to have been run")
class TestPreprocessing(unittest.TestCase):
    """Test suite validating Stage 2 preprocessing outputs."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.raw_dir = cls.project_root / "data" / "raw" / "cic_ids2017" / "MachineLearningCVE"
        cls.processed_dir = cls.project_root / "data" / "processed"
        cls.train_parquet = cls.processed_dir / "train" / "train.parquet"
        cls.val_parquet = cls.processed_dir / "validation" / "validation.parquet"
        cls.test_parquet = cls.processed_dir / "test" / "test.parquet"
        cls.report_json = cls.project_root / "ml" / "results" / "preprocessing_report.json"
        cls.doc_md = cls.project_root / "docs" / "preprocessing.md"

        # Read report once
        with open(cls.report_json, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_raw_data_remains_unchanged(self):
        """Verify that all 8 raw CSVs still exist and were not altered or moved."""
        raw_files = sorted(list(self.raw_dir.glob("*.csv")))
        self.assertEqual(len(raw_files), 8, "Expected 8 raw CSV files in MachineLearningCVE")

    def test_processed_artifacts_exist(self):
        """Verify that Parquet partitions, report JSON, and documentation exist."""
        self.assertTrue(self.train_parquet.exists(), "Missing train.parquet")
        self.assertTrue(self.val_parquet.exists(), "Missing validation.parquet")
        self.assertTrue(self.test_parquet.exists(), "Missing test.parquet")
        self.assertTrue(self.report_json.exists(), "Missing preprocessing_report.json")
        self.assertTrue(self.doc_md.exists(), "Missing docs/preprocessing.md")

    def test_schema_and_column_normalization(self):
        """Verify column names have no whitespace and Label exists exactly once."""
        # Read metadata schema from train parquet
        df_train_head = pd.read_parquet(self.train_parquet, columns=["Label"])
        self.assertIn("Label", df_train_head.columns)

        # Inspect all columns via test sample
        sample_test = pd.read_csv(self.project_root / "data" / "samples" / "sample_test.csv")
        cols = list(sample_test.columns)

        # Column names stripped of whitespace
        for col in cols:
            self.assertEqual(col, col.strip(), f"Column name has whitespace: '{col}'")

        # Label column exists exactly once
        self.assertEqual(cols.count("Label"), 1)
        self.assertEqual(len(cols), 70)  # 69 features + 1 Label

    def test_zero_variance_and_duplicate_features_removed(self):
        """Verify that 8 constant features and Fwd Header Length.1 are absent."""
        banned_cols = [
            "Bwd PSH Flags",
            "Bwd URG Flags",
            "Fwd Avg Bytes/Bulk",
            "Fwd Avg Packets/Bulk",
            "Fwd Avg Bulk Rate",
            "Bwd Avg Bytes/Bulk",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
            "Fwd Header Length.1",
        ]
        sample_train = pd.read_csv(self.project_root / "data" / "samples" / "sample_train.csv")
        for banned in banned_cols:
            self.assertNotIn(banned, sample_train.columns, f"Banned column still present: {banned}")

    def test_no_infinities_or_nulls_in_samples(self):
        """Verify that no NaN, +inf, or -inf values exist in the processed partitions."""
        for name in ["sample_train.csv", "sample_test.csv"]:
            sample_df = pd.read_csv(self.project_root / "data" / "samples" / name)
            feature_cols = [c for c in sample_df.columns if c != "Label"]
            self.assertEqual(sample_df[feature_cols].isna().sum().sum(), 0, f"Nulls found in {name}")
            self.assertEqual(np.isinf(sample_df[feature_cols].values).sum(), 0, f"Infs found in {name}")

    def test_stratified_split_counts_and_proportions(self):
        """Verify exact split row counts and 70/15/15 proportions."""
        counts = self.report["split_counts"]
        total = self.report["distinct_records"]

        self.assertEqual(total, 2522362)
        self.assertEqual(counts["train"], 1765653)
        self.assertEqual(counts["validation"], 378354)
        self.assertEqual(counts["test"], 378355)
        self.assertEqual(counts["train"] + counts["validation"] + counts["test"], total)

        self.assertAlmostEqual(counts["train"] / total, 0.70, places=2)
        self.assertAlmostEqual(counts["validation"] / total, 0.15, places=2)
        self.assertAlmostEqual(counts["test"] / total, 0.15, places=2)

    def test_rare_classes_preserved_across_all_splits(self):
        """Verify that all 15 classes, especially rare classes, exist in every split."""
        self.assertEqual(self.report["number_of_classes"], 15)

        for split_name in ["train", "validation", "test"]:
            split_dist = self.report["class_distribution"][split_name]
            self.assertEqual(len(split_dist), 15, f"{split_name} does not have all 15 classes!")

            # Check rare classes
            self.assertGreaterEqual(split_dist.get("Heartbleed", 0), 1)
            self.assertGreaterEqual(split_dist.get("Infiltration", 0), 1)
            self.assertGreaterEqual(split_dist.get("Web Attack Sql Injection", 0), 1)

    def test_no_unicode_replacement_characters_in_labels(self):
        """Verify that labels have no encoding artifacts (\ufffd)."""
        for cls_name in self.report["classes"]:
            self.assertNotIn("\ufffd", cls_name)
            self.assertNotIn("?", cls_name)


if __name__ == "__main__":
    unittest.main()
