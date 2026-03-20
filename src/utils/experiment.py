"""
src/utils/experiment.py
───────────────────────
Create and manage a self-contained experiment folder.

Folder layout (all created automatically):
    experiments/
    └── <experiment_name>/
        ├── config.yaml          ← frozen copy of the run config
        ├── models/              ← checkpoints
        ├── logs/                ← train.log + metrics.csv
        └── plots/               ← learning curves + prediction grids

Usage
-----
    from src.utils.experiment import ExperimentManager
    exp = ExperimentManager(cfg)          # cfg from load_config()
    exp.setup()
    exp.log_metrics(epoch=1, metrics={…})
    exp.save_config(cfg)
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import yaml


class ExperimentManager:
    """Manages a single experiment's folder structure and artifact storage."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.name: str = cfg.experiment_name
        self.root: Path = Path(cfg.output.experiments_dir) / self.name

        self.models_dir: Path = self.root / "models"
        self.logs_dir: Path = self.root / "logs"
        self.plots_dir: Path = self.root / "plots"

        self._metrics_path: Path = self.logs_dir / "metrics.csv"
        self._metrics_header_written: bool = False

    # ──────────────────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Create all sub-directories and freeze the config."""
        for d in (self.models_dir, self.logs_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._freeze_config()

    def _freeze_config(self) -> None:
        """Write a YAML snapshot of the config into the experiment folder."""
        dest = self.root / "config.yaml"
        with open(dest, "w") as f:
            yaml.dump(dict(self.cfg), f, default_flow_style=False, sort_keys=False)

    # ──────────────────────────────────────────────────────────────────────
    # Metric logging
    # ──────────────────────────────────────────────────────────────────────

    def log_metrics(self, epoch: int, metrics: dict[str, Any]) -> None:
        """
        Append one row to ``logs/metrics.csv``.

        Parameters
        ----------
        epoch : int
            1-based epoch number.
        metrics : dict
            Dictionary of metric name → scalar value (e.g. from the trainer's
            history dict after one epoch).
        """
        row = {"epoch": epoch, **metrics}

        write_header = not self._metrics_path.exists() or not self._metrics_header_written

        with open(self._metrics_path, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._metrics_header_written = True
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})

    # ──────────────────────────────────────────────────────────────────────
    # Convenience paths
    # ──────────────────────────────────────────────────────────────────────

    def best_model_path(self) -> Path:
        return self.models_dir / "best_model.pth"

    def final_model_path(self) -> Path:
        return self.models_dir / "final_model.pth"

    def plot_path(self, name: str) -> Path:
        """Return the full path for a plot file, e.g. ``plot_path('loss_curve.png')``."""
        return self.plots_dir / name

    def log_dir(self) -> Path:
        return self.logs_dir

    # ──────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Experiment : {self.name}",
            f"Root       : {self.root.resolve()}",
            f"Models     : {self.models_dir}",
            f"Logs       : {self.logs_dir}",
            f"Plots      : {self.plots_dir}",
        ]
        return "\n".join(lines)
