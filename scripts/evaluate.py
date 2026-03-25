"""
scripts/evaluate.py
───────────────────
Evaluate a saved model checkpoint on the test set.

Usage
-----
    python scripts/evaluate.py \\
        --config experiments/exp_001_baseline/config.yaml \\
        --checkpoint experiments/exp_001_baseline/models/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.data.dataset import HSSSegmentationDataset
from src.training.trainer import SegmentationTrainer
from src.utils.config import load_config
from src.utils.experiment import ExperimentManager
from src.utils.logger import get_logger
from src.utils.plotting import plot_predictions


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
    return DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=False, num_workers=0)


def main(config_path: str, checkpoint_path: str) -> None:
    cfg = load_config(config_path)
    exp = ExperimentManager(cfg)
    exp.setup()

    log = get_logger("evaluate", log_dir=exp.log_dir())
    log.info(f"Evaluating experiment : {cfg.experiment_name}")
    log.info(f"Checkpoint            : {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = build_dataloader(cfg, "test")

    trainer = SegmentationTrainer(
        cfg=cfg,
        exp=exp,
        train_loader=test_loader,  # unused placeholders
        val_loader=test_loader,
        test_loader=test_loader,
        device=device,
        logger=log,
    )
    trainer.build()

    metrics = trainer.evaluate(checkpoint=checkpoint_path)

    # Save a fresh prediction grid
    plot_predictions(
        model=trainer.model,
        dataset=test_loader.dataset,
        device=device,
        save_path=exp.plot_path("predictions_grid_eval.png"),
        n_samples=20,
    )

    log.info(f"Done.  Outputs → {exp.root.resolve()}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved HSI segmentation model.")
    parser.add_argument("--config",     required=True, help="Path to experiment config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint file.")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
