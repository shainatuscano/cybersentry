"""
Configuration management for CyberSentry.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default_config.yaml"


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load system configuration from a YAML file.

    Args:
        config_path: Optional path to YAML configuration. Defaults to configs/default_config.yaml.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}
