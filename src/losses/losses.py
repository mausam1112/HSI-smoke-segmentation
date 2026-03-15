"""
Loss functions for binary segmentation.

  • dice_loss           — global Dice loss
  • per_sample_dice_loss — per-sample Dice loss averaged over the batch
  • bce_dice_loss       — α·BCE + (1-α)·Dice  (default α=0.3)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(y_true: torch.Tensor, y_pred: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Global Dice loss — flattens the entire batch before computing."""
    y_true = y_true.float().reshape(-1)
    y_pred = y_pred.float().reshape(-1)
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


def per_sample_dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice loss computed per sample (N, C, H, W) and averaged over the batch.
    Preserves within-batch variance compared to the global version.
    """
    y_true = y_true.float()
    y_pred = y_pred.float()
    dims = (1, 2, 3)  # reduce C, H, W; keep N
    intersection = torch.sum(y_true * y_pred, dim=dims)
    union = torch.sum(y_true, dim=dims) + torch.sum(y_pred, dim=dims)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - torch.mean(dice)


def bce_dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    alpha: float = 0.3,
) -> torch.Tensor:
    """
    Combined BCE + Dice loss.

    Parameters
    ----------
    y_true : torch.Tensor  (N, C, H, W)
    y_pred : torch.Tensor  (N, C, H, W), values in [0, 1]
    alpha  : float
        Weight on the BCE term.  ``1 - alpha`` weights the Dice term.
    """
    bce = F.binary_cross_entropy(y_pred, y_true.float(), reduction="mean")
    d_loss = per_sample_dice_loss(y_true, y_pred)
    return alpha * bce + (1.0 - alpha) * d_loss


# ─────────────────────────────────────────────────────────────────────────────
# MoP auxiliary losses
# ─────────────────────────────────────────────────────────────────────────────

def prototype_diversity_loss(prototypes: torch.Tensor) -> torch.Tensor:
    """
    Prototype Uncorrelated Optimisation (PUO) — encourages the K prototypes
    within each band group to be diverse (i.e. capture different spectral
    patterns) by minimising the off-diagonal cosine similarities.

    Parameters
    ----------
    prototypes : torch.Tensor  (G, K, D)
        The learnable prototype matrix from MixtureOfPrototypes.psr.prototypes.

    Returns
    -------
    Scalar loss — mean squared off-diagonal cosine similarity across groups.
    """
    G, K, D = prototypes.shape
    # Normalise along D
    p_norm = F.normalize(prototypes, dim=-1)            # (G, K, D)
    # Gram matrix per group: (G, K, K)
    gram = torch.bmm(p_norm, p_norm.transpose(1, 2))   # (G, K, K)
    # Mask out diagonal (self-similarity = 1 is expected)
    eye = torch.eye(K, device=prototypes.device).unsqueeze(0)
    off_diag = gram * (1.0 - eye)
    return (off_diag ** 2).mean()


def mop_bce_dice_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    prototypes: torch.Tensor | None = None,
    alpha: float = 0.3,
    lambda_div: float = 0.1,
) -> torch.Tensor:
    """
    Full MoP training loss:
        L = α·BCE + (1-α)·Dice  +  λ_div · PrototypeDiversity

    Parameters
    ----------
    y_true     : (N, C, H, W)
    y_pred     : (N, C, H, W)  values in [0,1]
    prototypes : (G, K, D)  from model.psr.prototypes  (None → skip div term)
    alpha      : BCE weight
    lambda_div : diversity loss weight
    """
    seg_loss = bce_dice_loss(y_true, y_pred, alpha=alpha)
    if prototypes is not None:
        div_loss = prototype_diversity_loss(prototypes)
        return seg_loss + lambda_div * div_loss
    return seg_loss


# Registry: maps config string → callable
LOSS_REGISTRY: dict[str, callable] = {
    "dice": dice_loss,
    "per_sample_dice": per_sample_dice_loss,
    "bce_dice": bce_dice_loss,
    "mop_bce_dice": mop_bce_dice_loss,
}


def get_loss(name: str, **kwargs):
    """
    Retrieve a loss function by name and return a partial with **kwargs applied.

    Example
    -------
        criterion = get_loss("bce_dice", alpha=0.4)
    """
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    fn = LOSS_REGISTRY[name]
    if kwargs:
        import functools
        return functools.partial(fn, **kwargs)
    return fn
