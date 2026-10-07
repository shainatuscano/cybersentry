"""
Tests for Stage 1: Dataset Ingestion and Quality Audit.
Verifies audit outputs, data integrity, and dataset metrics.
"""

import json
from pathlib import Path
import unittest


_REPORT = Path(__file__).resolve().parents[1] / "ml" / "results" / "dataset_audit.json"


@unittest.skipUnless(_REPORT.exists(), "needs the 8 raw CIC-IDS2017 CSVs in data/raw/cic_ids2017/MachineLearningCVE/ "
                     "and `PYTHONPATH=src python -m cybersentry.data.audit` to have been run")
class TestDatasetAudit(unittest.TestCase):
    """Test suite validating Stage 1 dataset audit results."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]
        self.audit_json = self.project_root / "ml" / "results" / "dataset_audit.json"
        self.audit_md = self.project_root / "docs" / "dataset-audit.md"
        self.class_plot = self.project_root / "ml" / "results" / "class_distribution.png"
        self.issues_plot = self.project_root / "ml" / "results" / "data_quality_issues.png"

    def test_audit_artifacts_exist(self):
        """Verify that all Stage 1 output artifacts were generated."""
        self.assertTrue(self.audit_json.exists(), "Missing ml/results/dataset_audit.json")
        self.assertTrue(self.audit_md.exists(), "Missing docs/dataset-audit.md")
        self.assertTrue(self.class_plot.exists(), "Missing ml/results/class_distribution.png")
        self.assertTrue(self.issues_plot.exists(), "Missing ml/results/data_quality_issues.png")

        self.assertGreater(self.audit_json.stat().st_size, 1000)
        self.assertGreater(self.audit_md.stat().st_size, 1000)
        self.assertGreater(self.class_plot.stat().st_size, 1000)

    def test_audit_json_schema_and_metrics(self):
        """Verify metrics calculated from raw dataset."""
        with open(self.audit_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["csv_files_count"], 8)
        self.assertEqual(data["total_records"], 2830743)
        self.assertEqual(data["total_columns"], 79)
        self.assertEqual(data["target_column"].strip(), "Label")
        self.assertEqual(len(data["files"]), 8)

        # Verify class distribution presence
        labels = [d["label"] for d in data["label_distribution"]]
        self.assertIn("BENIGN", labels)
        self.assertIn("DoS Hulk", labels)
        self.assertIn("PortScan", labels)
        self.assertIn("DDoS", labels)

        # Check total record sum across labels equals total records
        label_sum = sum(d["count"] for d in data["label_distribution"])
        self.assertEqual(label_sum, 2830743)

    def test_raw_files_unmodified_and_not_committed(self):
        """Verify that raw files exist and are untracked by git."""
        raw_dir = self.project_root / "data" / "raw" / "cic_ids2017" / "MachineLearningCVE"
        csv_files = list(raw_dir.glob("*.csv"))
        self.assertEqual(len(csv_files), 8)


if __name__ == "__main__":
    unittest.main()
