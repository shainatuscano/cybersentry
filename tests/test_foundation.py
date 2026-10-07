"""
Tests for Stage 0 Foundation: verify project structure, documentation, configuration, and environment setup.
Supports execution via both standard unittest and pytest.
"""

from pathlib import Path
import unittest

import cybersentry
from cybersentry.config import load_config


class TestFoundation(unittest.TestCase):
    """Stage 0 Foundation validation test suite."""

    def setUp(self):
        self.project_root = Path(__file__).resolve().parents[1]

    def test_package_metadata(self):
        """Verify package version and root imports."""
        self.assertEqual(cybersentry.__version__, "0.1.0")

    def test_directory_structure(self):
        """Verify that all agreed directories exist across data, ml, ai-engine, backend, and frontend."""
        expected_dirs = [
            # Data layer
            self.project_root / "data" / "raw" / "cic_ids2017",
            self.project_root / "data" / "processed",
            self.project_root / "data" / "samples",
            # ML hierarchy
            self.project_root / "ml" / "src",
            self.project_root / "ml" / "models",
            self.project_root / "ml" / "notebooks",
            self.project_root / "ml" / "results",
            self.project_root / "ml" / "configs",
            # AI Engine hierarchy
            self.project_root / "ai-engine" / "agents",
            self.project_root / "ai-engine" / "tools",
            self.project_root / "ai-engine" / "rag",
            self.project_root / "ai-engine" / "graph",
            # Backend & Frontend placeholders
            self.project_root / "backend",
            self.project_root / "frontend",
            # Shared packages, configs, and docs
            self.project_root / "configs",
            self.project_root / "docs",
            self.project_root / "src" / "cybersentry",
            self.project_root / "tests",
        ]

        for d in expected_dirs:
            self.assertTrue(d.exists() and d.is_dir(), f"Expected directory does not exist: {d}")

    def test_documentation_files_exist(self):
        """Verify that all mandatory documentation files exist with content."""
        expected_docs = [
            self.project_root / "AGENTS.md",
            self.project_root / "README.md",
            self.project_root / "docs" / "architecture.md",
            self.project_root / "docs" / "development-roadmap.md",
        ]

        for doc in expected_docs:
            self.assertTrue(doc.exists() and doc.is_file(), f"Expected doc missing: {doc}")
            self.assertGreater(doc.stat().st_size, 100, f"Doc file appears empty: {doc}")

    def test_gitignore_rules(self):
        """Verify that gitignore exists and protects dataset files, models, and environments."""
        gitignore_path = self.project_root / ".gitignore"
        self.assertTrue(gitignore_path.exists(), "Missing .gitignore file")

        content = gitignore_path.read_text(encoding="utf-8")
        self.assertTrue("data/raw/cic_ids2017/*" in content or "data/raw/*" in content)
        self.assertIn("data/processed/*", content)
        self.assertIn("data/samples/*", content)
        self.assertIn("ml/models/*", content)
        self.assertTrue(".venv/" in content or "venv/" in content)

    def test_default_config_loading(self):
        """Verify default configuration file parses successfully with expected keys."""
        config = load_config()
        self.assertIsInstance(config, dict)
        self.assertIn("project", config)
        self.assertEqual(config["project"]["name"], "CyberSentry")
        self.assertIn("data", config)
        self.assertEqual(config["data"]["raw_dir"], "data/raw/cic_ids2017")
        self.assertIn("models", config)
        self.assertIn("supervised_candidates", config["models"])

    def test_no_premature_models_or_agents_implemented(self):
        """Stage 0 Guard: Ensure no agent implementations exist yet.

        Trained weights in models/ are expected: src/ml/train_models writes them there (gitignored).
        """
        # ai-engine directories must remain unimplemented skeletons (only .gitkeep)
        for sub in ["agents", "tools", "rag", "graph"]:
            sub_dir = self.project_root / "ai-engine" / sub
            py_files = list(sub_dir.glob("*.py"))
            self.assertEqual(py_files, [], f"Found premature Python code in {sub_dir}: {py_files}")


if __name__ == "__main__":
    unittest.main()
