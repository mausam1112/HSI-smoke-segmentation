"""
src/utils/config.py
───────────────────
Load and merge YAML configuration files.

Usage
-----
    from src.utils.config import load_config
    cfg = load_config("configs/exp_001_baseline.yaml")
    print(cfg.training.learning_rate)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# Path to the shared base config (relative to project root)
_BASE_CONFIG = Path(__file__).parents[3] / "configs" / "base.yaml"


class DotDict(dict):
    """A dict subclass that supports attribute-style access (cfg.key)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None
        return DotDict(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'") from None


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive copy)."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_config(experiment_config_path: str | Path) -> DotDict:
    """
    Load the base config and merge the experiment-level overrides on top.

    Parameters
    ----------
    experiment_config_path : str | Path
        Path to the experiment YAML (e.g. ``configs/exp_001_baseline.yaml``).

    Returns
    -------
    DotDict
        Merged configuration with attribute-style access.
    """
    experiment_config_path = Path(experiment_config_path)

    # Load base defaults
    with open(_BASE_CONFIG, "r") as f:
        base = yaml.safe_load(f) or {}

    # Load experiment overrides
    with open(experiment_config_path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    merged = _deep_merge(base, overrides)
    return DotDict(merged)


def save_config(cfg: dict | DotDict, path: str | Path) -> None:
    """Freeze a config snapshot to *path* (YAML)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
