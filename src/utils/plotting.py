"""
src/utils/plotting.py
─────────────────────
Plotting helpers for learning curves and prediction grids.
All plots are saved to disk; nothing is shown interactively.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def plot_metrics(history: dict, save_dir: str | Path) -> None:
    """
    Save one PNG per metric (training vs. validation curve).

    Parameters
    ----------
    history : dict
        Keys like ``"loss"``, ``"val_loss"``, ``"f1_score"``, etc.
    save_dir : str | Path
        Directory where PNGs are written.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = ["loss", "f1_score", "mean_iou", "precision", "recall"]

    for metric in base_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        if metric in history:
            ax.plot(history[metric], label=f"Train {metric.replace('_', ' ').title()}")
        val_key = f"val_{metric}"
        if val_key in history:
            ax.plot(history[val_key], label=f"Val {metric.replace('_', ' ').title()}")

        ax.set_title(f"Training vs Validation — {metric.replace('_', ' ').title()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        out = save_dir / f"{metric}_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)

    print(f"[plot_metrics] Saved {len(base_metrics)} plots to {save_dir}")


def plot_predictions(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    save_path: str | Path,
    n_samples: int = 20,
    pool_size: int = 50,
) -> None:
    """
    Draw a grid of (band-averaged image | GT mask | predicted mask | overlay)
    for ``n_samples`` randomly chosen test-set items and save to ``save_path``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    dataset : torch.utils.data.Dataset
        Test dataset (individual items, no batch dim).
    device : torch.device
    save_path : str | Path
        Full path for the output PNG.
    n_samples : int
        Number of rows in the grid.
    pool_size : int
        How many items to randomly sample from before picking ``n_samples``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    sample_images, sample_masks = [], []
    for i, (img, msk) in enumerate(loader):
        if i >= pool_size:
            break
        sample_images.append(img.squeeze(0))   # (C, H, W)
        sample_masks.append(msk.squeeze(0))    # (1, H, W)

    if len(sample_images) < n_samples:
        print(
            f"[plot_predictions] Only {len(sample_images)} samples available; "
            f"plotting all of them."
        )
        n_samples = len(sample_images)

    indices = random.sample(range(len(sample_images)), n_samples)

    model.eval()
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    fig.suptitle("Model Predictions on Test Set", fontsize=16)

    col_titles = ["Band Average", "Ground Truth", "Prediction", "Overlay (GT=R, Pred=G)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10)

    for row, idx in enumerate(indices):
        img_tensor = sample_images[idx]        # (C, H, W)
        msk_tensor = sample_masks[idx]         # (1, H, W)

        with torch.no_grad():
            pred_raw = model(img_tensor.unsqueeze(0).to(device))  # (1,1,H,W)
        pred = (pred_raw.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        gt = msk_tensor.squeeze().cpu().numpy().astype(np.uint8)

        # Band-averaged image
        band_avg = img_tensor.permute(1, 2, 0).mean(dim=2).numpy()
        axes[row, 0].imshow(band_avg, cmap="viridis")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 2].axis("off")

        # Overlay: GT → Red channel, Prediction → Green channel
        overlay = np.zeros((*gt.shape, 3), dtype=np.float32)
        overlay[:, :, 0] = gt * 0.7
        overlay[:, :, 1] = pred * 0.7
        axes[row, 3].imshow(overlay)
        axes[row, 3].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[plot_predictions] Saved prediction grid → {save_path}")
