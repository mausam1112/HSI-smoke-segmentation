"""
Per-batch segmentation metrics for binary masks.

All functions:
  • accept tensors of shape (N, C, H, W)
  • threshold predictions at 0.5
  • return a scalar tensor (mean over the batch)
"""

from __future__ import annotations

import torch

# Dimensions to reduce: keep N (batch), reduce C, H, W
_DIMS = (1, 2, 3)
_SMOOTH = 1e-6


def _binarize(y_pred: torch.Tensor) -> torch.Tensor:
    return (y_pred > 0.5).float()


def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Dice / F1 score averaged over the batch."""
    y_true = y_true.float()
    y_pred = _binarize(y_pred)
    tp = torch.sum(y_true * y_pred, dim=_DIMS)
    fp = torch.sum((1 - y_true) * y_pred, dim=_DIMS)
    fn = torch.sum(y_true * (1 - y_pred), dim=_DIMS)
    score = (2 * tp + _SMOOTH) / (2 * tp + fp + fn + _SMOOTH)
    return torch.mean(score)


def mean_iou(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Intersection-over-Union averaged over the batch."""
    y_true = y_true.float()
    y_pred = _binarize(y_pred)
    intersection = torch.sum(y_true * y_pred, dim=_DIMS)
    union = torch.sum(y_true, dim=_DIMS) + torch.sum(y_pred, dim=_DIMS) - intersection
    return torch.mean((intersection + _SMOOTH) / (union + _SMOOTH))


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Precision averaged over the batch."""
    y_true = y_true.float()
    y_pred = _binarize(y_pred)
    tp = torch.sum(y_true * y_pred, dim=_DIMS)
    fp = torch.sum((1 - y_true) * y_pred, dim=_DIMS)
    return torch.mean((tp + _SMOOTH) / (tp + fp + _SMOOTH))


def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Recall averaged over the batch."""
    y_true = y_true.float()
    y_pred = _binarize(y_pred)
    tp = torch.sum(y_true * y_pred, dim=_DIMS)
    fn = torch.sum(y_true * (1 - y_pred), dim=_DIMS)
    return torch.mean((tp + _SMOOTH) / (tp + fn + _SMOOTH))


def compute_all(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """
    Convenience: compute all four metrics and return as a plain dict of Python floats.

    Returns
    -------
    dict with keys: ``f1_score``, ``mean_iou``, ``precision``, ``recall``
    """
    return {
        "f1_score":  f1_score(y_true, y_pred).item(),
        "mean_iou":  mean_iou(y_true, y_pred).item(),
        "precision": precision(y_true, y_pred).item(),
        "recall":    recall(y_true, y_pred).item(),
    }
