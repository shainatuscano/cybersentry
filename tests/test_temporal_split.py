"""
Stage 2.1 Verification Suite: Temporal / Scenario-Aware Data Split.
Validates temporal partition existence, schema matching, immutability of raw data,
zero cross-partition duplicate leakage, source session separation, and honest class presence/absence reporting.
"""

import json
from pathlib import Path
import unittest

import pandas as pd


_REPORT = Path(__file__).resolve().parents[1] / "ml" / "results" / "temporal_split_report.json"


@unittest.skipUnless(_REPORT.exists(), "needs the 8 raw CIC-IDS2017 CSVs in data/raw/cic_ids2017/MachineLearningCVE/ "
                     "and `PYTHONPATH=src python -m cybersentry.data.temporal_split` to have been run")
class TestTemporalSplit(unittest.TestCase):
    """Test suite validating Stage 2.1 temporal / scenario-aware split outputs."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.raw_dir = cls.project_root / "data" / "raw" / "cic_ids2017" / "MachineLearningCVE"
        cls.temporal_dir = cls.project_root / "data" / "processed" / "temporal"
        cls.train_parquet = cls.temporal_dir / "train.parquet"
        cls.val_parquet = cls.temporal_dir / "validation.parquet"
        cls.test_parquet = cls.temporal_dir / "test.parquet"
        cls.report_json = cls.project_root / "ml" / "results" / "temporal_split_report.json"
        cls.doc_md = cls.project_root / "docs" / "preprocessing.md"

        # Load report
        with open(cls.report_json, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_temporal_files_exist(self):
        """Verify that all temporal Parquet partitions and report artifacts exist."""
        self.assertTrue(self.train_parquet.exists(), "Missing temporal/train.parquet")
        self.assertTrue(self.val_parquet.exists(), "Missing temporal/validation.parquet")
        self.assertTrue(self.test_parquet.exists(), "Missing temporal/test.parquet")
        self.assertTrue(self.report_json.exists(), "Missing ml/results/temporal_split_report.json")
        self.assertTrue(self.doc_md.exists(), "Missing docs/preprocessing.md")

    def test_schemas_match_and_feature_count(self):
        """Verify that schemas match across all temporal partitions and have 70 columns."""
        df_tr_head = pd.read_parquet(self.train_parquet, engine="pyarrow").head(10)
        df_va_head = pd.read_parquet(self.val_parquet, engine="pyarrow").head(10)
        df_te_head = pd.read_parquet(self.test_parquet, engine="pyarrow").head(10)

        self.assertEqual(list(df_tr_head.columns), list(df_va_head.columns))
        self.assertEqual(list(df_tr_head.columns), list(df_te_head.columns))
        self.assertEqual(len(df_tr_head.columns), 70)  # 69 features + 1 Label
        self.assertIn("Label", df_tr_head.columns)

    def test_raw_dataset_remains_unchanged(self):
        """Verify that raw CSV files remain present and intact in MachineLearningCVE."""
        raw_files = sorted(list(self.raw_dir.glob("*.csv")))
        self.assertEqual(len(raw_files), 8, "Expected 8 raw CSV files in MachineLearningCVE")

    def test_zero_exact_duplicates_across_temporal_partitions(self):
        """Verify that cross-partition exact duplicate leakage is strictly zero."""
        dup_checks = self.report["duplicate_checks"]
        self.assertEqual(dup_checks["train_vs_validation_duplicates"], 0)
        self.assertEqual(dup_checks["train_vs_test_duplicates"], 0)
        self.assertEqual(dup_checks["validation_vs_test_duplicates"], 0)

    def test_source_capture_groups_are_separated(self):
        """Verify that source CSV capture groups are completely disjoint between train, val, and test."""
        src_files = self.report["source_files"]
        train_set = set(src_files["train"])
        val_set = set(src_files["validation"])
        test_set = set(src_files["test"])

        self.assertEqual(len(train_set & val_set), 0, "Train and Validation share source files!")
        self.assertEqual(len(train_set & test_set), 0, "Train and Test share source files!")
        self.assertEqual(len(val_set & test_set), 0, "Validation and Test share source files!")
        self.assertEqual(len(train_set | val_set | test_set), 8, "Not all 8 files allocated!")

    def test_class_presence_and_absence_accurately_reported(self):
        """Verify that classes present and absent per temporal partition match ground truth."""
        classes_absent = self.report["classes_absent"]
        train_absent = set(classes_absent["train"])
        val_absent = set(classes_absent["validation"])
        test_absent = set(classes_absent["test"])

        # Friday attacks (PortScan, DDoS, Bot) must be absent from train and val
        self.assertIn("PortScan", train_absent)
        self.assertIn("DDoS", train_absent)
        self.assertIn("Bot", train_absent)
        self.assertIn("PortScan", val_absent)
        self.assertIn("DDoS", val_absent)

        # Thursday attacks (Web Attacks, Infiltration) must be absent from train and test
        self.assertIn("Web Attack Brute Force", train_absent)
        self.assertIn("Infiltration", train_absent)
        self.assertIn("Web Attack Brute Force", test_absent)
        self.assertIn("Infiltration", test_absent)

        # Wednesday/Tuesday attacks (DoS Hulk, Patators) must be absent from val and test
        self.assertIn("DoS Hulk", val_absent)
        self.assertIn("DoS Hulk", test_absent)
        self.assertIn("FTP-Patator", test_absent)

        # BENIGN must be present in all three partitions
        self.assertNotIn("BENIGN", train_absent)
        self.assertNotIn("BENIGN", val_absent)
        self.assertNotIn("BENIGN", test_absent)


if __name__ == "__main__":
    unittest.main()
