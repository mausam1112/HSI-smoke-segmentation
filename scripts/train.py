"""
scripts/train.py
────────────────
CLI entry-point for training a segmentation experiment.

Usage
-----
    python scripts/train.py --config configs/exp_001_baseline.yaml

The script will:
  1. Load and merge configs/base.yaml + the provided config.
  2. Create the experiment folder under experiments/<name>/.
  3. Set up logging (console + file).
  4. Build dataloaders, model, trainer.
  5. Train, evaluate on the test set, plot curves + prediction grid.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Allow running from project root without installing the package ─────────
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.dataset import HSSSegmentationDataset
from src.training.trainer import SegmentationTrainer
from src.utils.config import load_config
from src.utils.experiment import ExperimentManager
from src.utils.logger import get_logger
from src.utils.plotting import plot_metrics, plot_predictions


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(cfg, split: str) -> DataLoader:
    dcfg = cfg.data
    tcfg = cfg.training
    dataset = HSSSegmentationDataset(
        dataset_root=dcfg.dataset_root,
        image_dir=dcfg.image_dir,
        mask_dir=dcfg.mask_dir,
        image_size=tuple(dcfg.image_size),
        split=split,
        img_ext=dcfg.img_ext,
        mask_ext=dcfg.mask_ext,
        img_channels=dcfg.img_channels,
        mask_channels=dcfg.mask_channels,
        data_version=dcfg.data_version,
    )
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=shuffle, num_workers=0)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg.seed)

    # ── Experiment setup ──────────────────────────────────────────────────
    exp = ExperimentManager(cfg)
    exp.setup()

    log = get_logger("train", log_dir=exp.log_dir())
    log.info(exp.summary())

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building dataloaders …")
    train_loader = build_dataloader(cfg, "train")
    val_loader   = build_dataloader(cfg, "val")
    test_loader  = build_dataloader(cfg, "test")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SegmentationTrainer(
        cfg=cfg,
        exp=exp,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────
    trainer.evaluate(checkpoint=exp.best_model_path())

    # ── Plots ─────────────────────────────────────────────────────────────
    log.info("Saving learning-curve plots …")
    plot_metrics(history, save_dir=exp.plots_dir)

    log.info("Saving prediction grid …")
    plot_predictions(
        model=trainer.model,
        dataset=test_loader.dataset,
        device=device,
        save_path=exp.plot_path("predictions_grid.png"),
        n_samples=20,
    )

    log.info(f"All outputs written to: {exp.root.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HSI segmentation model.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment YAML config, e.g. configs/exp_001_baseline.yaml",
    )
    args = parser.parse_args()
    main(args.config)
