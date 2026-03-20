"""
SegmentationTrainer: orchestrates training, validation, test evaluation,
metric logging, checkpointing, and plot generation.

All I/O paths come from the ExperimentManager injected at construction time,
keeping this class free of hard-coded paths.

MoP support
───────────
When the model has a `psr` attribute (i.e. MixtureOfPrototypes), the trainer
automatically passes `psr.prototypes` to the loss function so the prototype
diversity term is included without any extra glue code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.losses.losses import get_loss
from src.metrics.metrics import compute_all
from src.models.registry import build_model
from src.utils.experiment import ExperimentManager


class SegmentationTrainer:
    """
    Trains, validates, and evaluates any registered segmentation model.

    Parameters
    ----------
    cfg : DotDict
        Merged experiment config (from ``load_config``).
    exp : ExperimentManager
        Manages output paths and metric persistence.
    train_loader, val_loader, test_loader : DataLoader
    device : torch.device
    logger : logging.Logger
    """

    def __init__(
        self,
        cfg,
        exp: ExperimentManager,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.cfg = cfg
        self.exp = exp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log = logger

        self.model: torch.nn.Module | None = None
        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None

        # In-memory history (mirrors metrics.csv for plot generation)
        self.history: dict[str, list] = {
            "loss": [], "val_loss": [],
            "f1_score": [], "val_f1_score": [],
            "mean_iou": [], "val_mean_iou": [],
            "precision": [], "val_precision": [],
            "recall": [], "val_recall": [],
        }

    # Build

    def build(self) -> None:
        """Instantiate model (via registry), optimizer, loss, and LR scheduler."""
        tcfg = self.cfg.training

        self.model = build_model(self.cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.info(
            f"Model      : {self.cfg.model.name}\n"
            f"  Params   : {n_params:,} total  |  {n_trainable:,} trainable"
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=tcfg.learning_rate)

        # Pass alpha (and lambda_div for MoP) from config as partial kwargs.
        loss_kwargs: dict[str, Any] = {"alpha": tcfg.alpha}
        if hasattr(tcfg, "lambda_div"):
            loss_kwargs["lambda_div"] = tcfg.lambda_div
        self.criterion = get_loss(tcfg.loss, **loss_kwargs)
        self.log.info(f"Loss       : {tcfg.loss}  kwargs={loss_kwargs}")

        sched_cfg = tcfg.lr_scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=sched_cfg.mode,
            factor=sched_cfg.factor,
            patience=sched_cfg.patience,
            min_lr=sched_cfg.min_lr,
        )
        self.log.info("Model, optimiser, loss, and LR scheduler initialised.")

    def train(self) -> dict:
        """
        Full training loop.

        Returns
        -------
        dict  — training history (same as ``self.history``)
        """
        if self.model is None:
            self.build()

        tcfg = self.cfg.training
        epochs = tcfg.epochs
        log_every = self.cfg.output.log_every_n_batches

        best_val_miou = -1.0
        patience_counter = 0
        es_patience = tcfg.early_stopping.patience

        self.log.info("=" * 60)
        self.log.info(f"Experiment : {self.cfg.experiment_name}")
        self.log.info(f"Model      : {self.cfg.model.name}")
        self.log.info(f"Epochs     : {epochs}")
        self.log.info(f"Batch size : {self.train_loader.batch_size}")
        self.log.info(f"LR         : {tcfg.learning_rate}")
        self.log.info(f"Device     : {self.device}")
        self.log.info("=" * 60)

        for epoch in range(1, epochs + 1):
            # Train phase 
            self.model.train()
            train_metrics = self._run_epoch(
                self.train_loader, training=True, log_every=log_every
            )

            # Val phase
            self.model.eval()
            with torch.no_grad():
                val_metrics = self._run_epoch(self.val_loader, training=False)

            # Update history 
            self.history["loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            for m in ("f1_score", "mean_iou", "precision", "recall"):
                self.history[m].append(train_metrics[m])
                self.history[f"val_{m}"].append(val_metrics[m])

            # CSV logging 
            self.exp.log_metrics(
                epoch=epoch,
                metrics={
                    "loss":          train_metrics["loss"],
                    "val_loss":      val_metrics["loss"],
                    "f1_score":      train_metrics["f1_score"],
                    "val_f1_score":  val_metrics["f1_score"],
                    "mean_iou":      train_metrics["mean_iou"],
                    "val_mean_iou":  val_metrics["mean_iou"],
                    "precision":     train_metrics["precision"],
                    "val_precision": val_metrics["precision"],
                    "recall":        train_metrics["recall"],
                    "val_recall":    val_metrics["recall"],
                    "lr":            self.optimizer.param_groups[0]["lr"],
                },
            )

            self.log.info(
                f"[Epoch {epoch:>4}/{epochs}] "
                f"loss={train_metrics['loss']:.4f}  val_loss={val_metrics['loss']:.4f}  "
                f"mIoU={train_metrics['mean_iou']:.4f}  val_mIoU={val_metrics['mean_iou']:.4f}  "
                f"F1={train_metrics['f1_score']:.4f}  val_F1={val_metrics['f1_score']:.4f}"
            )

            # LR scheduler
            self.lr_scheduler.step(val_metrics["mean_iou"])

            # Checkpoint ─
            if val_metrics["mean_iou"] > best_val_miou:
                best_val_miou = val_metrics["mean_iou"]
                torch.save(self.model.state_dict(), self.exp.best_model_path())
                self.log.info(
                    f"  ✓ Best model saved  (val_mIoU={best_val_miou:.4f})"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                self.log.debug(
                    f"  No improvement. Patience {patience_counter}/{es_patience}"
                )

            # Early stopping
            if patience_counter >= es_patience:
                self.log.info(f"Early stopping triggered at epoch {epoch}.")
                break

        # Save final weights
        torch.save(self.model.state_dict(), self.exp.final_model_path())
        self.log.info(f"Final model saved → {self.exp.final_model_path()}")

        return self.history

    # Evaluate

    def evaluate(self, checkpoint: str | Path | None = None) -> dict[str, float]:
        """
        Evaluate on the test set.

        Parameters
        ----------
        checkpoint : optional path to a .pth file.
            If ``None``, uses the current model weights.
        """
        if self.model is None:
            self.build()

        if checkpoint is not None:
            self.model.load_state_dict(
                torch.load(checkpoint, map_location=self.device)
            )
            self.log.info(f"Loaded checkpoint: {checkpoint}")

        self.model.eval()
        with torch.no_grad():
            test_metrics = self._run_epoch(self.test_loader, training=False)

        self.log.info(" Test Results:")
        for k, v in test_metrics.items():
            self.log.info(f"  {k:<15}: {v:.4f}")
        self.log.info("─────────")
        return test_metrics

    # Internal helpers

    def _get_prototypes(self) -> "torch.Tensor | None":
        """
        Return the prototype tensor from a MoP model, or None for other models.
        This keeps the training loop model-agnostic — no isinstance checks needed.
        """
        psr = getattr(self.model, "psr", None)
        return getattr(psr, "prototypes", None)

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        log_every: int = 0,
    ) -> dict[str, float]:
        """Run one full pass over *loader* and return averaged metrics."""
        total_loss = 0.0
        agg: dict[str, float] = {
            "f1_score": 0.0, "mean_iou": 0.0, "precision": 0.0, "recall": 0.0
        }
        n_batches = 0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            if training:
                self.optimizer.zero_grad()

            outputs = self.model(images)

            # MoP: pass prototype tensor for diversity regularisation if available.
            # For all other models this returns None and the loss ignores it.
            prototypes = self._get_prototypes()
            if prototypes is not None:
                loss = self.criterion(masks, outputs, prototypes=prototypes)
            else:
                loss = self.criterion(masks, outputs)

            if training:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            batch_metrics = compute_all(
                masks.detach().cpu(), outputs.detach().cpu()
            )
            for k in agg:
                agg[k] += batch_metrics[k]
            n_batches += 1

            if training and log_every > 0 and (batch_idx + 1) % log_every == 0:
                self.log.debug(
                    f"  Batch {batch_idx+1}/{len(loader)} — loss={loss.item():.4f}"
                )

        result = {"loss": total_loss / n_batches}
        result.update({k: v / n_batches for k, v in agg.items()})
        return result
