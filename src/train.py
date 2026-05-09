"""
src/train.py
============
PBP: Post-training Backdoor Purification for Malware Classifiers
Training and validation loops with early stopping and checkpoint saving.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import build_loaders
from src.model import MalwareMLP


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """
    Hyperparameters and paths for a training run.

    Parameters
    ----------
    csv_path : str
        Path to the malware CSV dataset.
    checkpoint_dir : str
        Directory where the best model checkpoint is saved.
    hidden_dims : list[int]
        Hidden layer widths for :class:`~src.model.MalwareMLP`.
    dropout_rate : float
        Dropout probability.
    lr : float
        Initial Adam learning rate.
    weight_decay : float
        L2 regularisation coefficient for Adam.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum number of training epochs.
    train_ratio : float
        Fraction of data allocated to training.
    patience : int
        Early-stopping patience (epochs without val-loss improvement).
    seed : int
        Global random seed.
    num_workers : int
        DataLoader worker count.
    label_col : str
        Name of the label column in the CSV.
    """

    csv_path: str = "data/malware.csv"
    checkpoint_dir: str = "checkpoints"
    hidden_dims: list = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 50
    train_ratio: float = 0.8
    patience: int = 7
    seed: int = 42
    num_workers: int = 0
    label_col: str = "label"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    logits: Tensor,
    targets: Tensor,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Compute accuracy and macro F1-score from raw logits.

    Parameters
    ----------
    logits : Tensor
        Raw model output of shape ``(B, 1)`` or ``(B,)``.
    targets : Tensor
        Ground-truth binary labels of shape ``(B,)``.
    threshold : float
        Decision boundary for converting probabilities to predictions.

    Returns
    -------
    accuracy : float
    f1 : float
    """
    probs = torch.sigmoid(logits).squeeze(-1)
    preds = (probs >= threshold).long().cpu().numpy()
    tgts  = targets.cpu().numpy()

    accuracy = (preds == tgts).mean().item()
    f1       = f1_score(tgts, preds, average="macro", zero_division=0)
    return accuracy, f1


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitor a validation metric and signal when training should stop.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after the last improvement.
    min_delta : float
        Minimum decrease in monitored loss to qualify as improvement.
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.stop       = False

    def step(self, val_loss: float) -> bool:
        """
        Update state and return ``True`` if training should stop.

        Parameters
        ----------
        val_loss : float
            Validation loss for the current epoch.

        Returns
        -------
        bool
            ``True`` when the patience budget is exhausted.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full training epoch.

    Parameters
    ----------
    model : nn.Module
        The classifier.
    loader : DataLoader
        Training DataLoader.
    criterion : nn.Module
        Loss function (BCEWithLogitsLoss).
    optimizer : Optimizer
        Parameter optimiser.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch index (for tqdm description).

    Returns
    -------
    dict with keys ``loss``, ``accuracy``, ``f1``.
    """
    model.train()
    total_loss = 0.0
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, unit="batch")

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch).squeeze(-1)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        all_logits.append(logits.detach())
        all_targets.append(y_batch.detach())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss      = total_loss / len(loader.dataset)
    all_logits_t  = torch.cat(all_logits)
    all_targets_t = torch.cat(all_targets)
    acc, f1       = _compute_metrics(all_logits_t, all_targets_t)

    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


@torch.no_grad()
def _validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Run one full validation epoch without gradient computation.

    Parameters
    ----------
    model : nn.Module
        The classifier.
    loader : DataLoader
        Validation DataLoader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Compute device.
    epoch : int
        Current epoch index.

    Returns
    -------
    dict with keys ``loss``, ``accuracy``, ``f1``.
    """
    model.eval()
    total_loss = 0.0
    all_logits: list[Tensor] = []
    all_targets: list[Tensor] = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, unit="batch")

    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True).float()

        logits = model(X_batch).squeeze(-1)
        loss   = criterion(logits, y_batch)

        total_loss += loss.item() * len(y_batch)
        all_logits.append(logits)
        all_targets.append(y_batch)

    avg_loss      = total_loss / len(loader.dataset)
    all_logits_t  = torch.cat(all_logits)
    all_targets_t = torch.cat(all_targets)
    acc, f1       = _compute_metrics(all_logits_t, all_targets_t)

    return {"loss": avg_loss, "accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    model: nn.Module,
    epoch: int,
    val_metrics: Dict[str, float],
    config: TrainConfig,
    checkpoint_dir: Path,
) -> Path:
    """
    Persist model weights and metadata to disk.

    Parameters
    ----------
    model : nn.Module
        Model whose ``state_dict`` is saved.
    epoch : int
        Current epoch (stored in metadata).
    val_metrics : dict
        Validation metrics for this epoch.
    config : TrainConfig
        Training configuration (stored in metadata).
    checkpoint_dir : Path
        Directory for the checkpoint file.

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "best_model.pt"
    torch.save(
        {
            "epoch":       epoch,
            "state_dict":  model.state_dict(),
            "val_metrics": val_metrics,
            "config":      config.__dict__,
        },
        ckpt_path,
    )
    return ckpt_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: Optional[MalwareMLP] = None,
    device: Optional[torch.device] = None,
) -> Tuple[MalwareMLP, Dict]:
    """
    Load a model checkpoint from disk.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the ``.pt`` checkpoint file.
    model : MalwareMLP | None
        If provided, weights are loaded in-place; otherwise a new model is
        constructed from the saved config.
    device : torch.device | None
        Device to map tensors to. Defaults to CPU.

    Returns
    -------
    model : MalwareMLP
        Model with restored weights set to eval mode.
    metadata : dict
        Checkpoint metadata (epoch, val_metrics, config).
    """
    if device is None:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if model is None:
        cfg   = ckpt["config"]
        model = MalwareMLP(
            input_dim    = cfg.get("n_features", 100),
            hidden_dims  = cfg.get("hidden_dims", [512, 256, 128]),
            dropout_rate = cfg.get("dropout_rate", 0.3),
        )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, metadata


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: TrainConfig) -> MalwareMLP:
    """
    Full training pipeline: data loading → model init → train/val loops
    → early stopping → checkpoint saving.

    Parameters
    ----------
    config : TrainConfig
        All hyperparameters and path settings.

    Returns
    -------
    MalwareMLP
        Best model (weights restored from checkpoint).
    """
    # ── Reproducibility ──────────────────────────────────────────────────
    torch.manual_seed(config.seed)

    # ── Device ───────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PBP] Using device : {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"[PBP] Loading dataset from : {config.csv_path}")
    train_loader, val_loader, scaler, n_features = build_loaders(
        csv_path    = config.csv_path,
        label_col   = config.label_col,
        train_ratio = config.train_ratio,
        batch_size  = config.batch_size,
        seed        = config.seed,
        num_workers = config.num_workers,
        pin_memory  = device.type == "cuda",
    )
    print(
        f"[PBP] Train batches: {len(train_loader)} | "
        f"Val batches: {len(val_loader)} | "
        f"Features: {n_features}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = MalwareMLP(
        input_dim    = n_features,
        hidden_dims  = config.hidden_dims,
        dropout_rate = config.dropout_rate,
    ).to(device)
    print(f"[PBP] Model params : {model.count_parameters():,}")

    # ── Loss / optimiser / scheduler ─────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr           = config.lr,
        weight_decay = config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        # pyrefly: ignore [unexpected-keyword]
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training state ────────────────────────────────────────────────────
    early_stopping  = EarlyStopping(patience=config.patience)
    checkpoint_dir  = Path(config.checkpoint_dir)
    best_val_loss   = float("inf")
    best_ckpt_path: Optional[Path] = None
    history: list[Dict] = []

    print(f"[PBP] Starting training for up to {config.max_epochs} epochs …\n")
    t0 = time.time()

    for epoch in range(1, config.max_epochs + 1):

        # Train
        train_metrics = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_metrics = _validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # LR schedule
        scheduler.step(val_metrics["loss"])

        # Log
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"Epoch {epoch:03d}/{config.max_epochs}  "
            f"| train loss {train_metrics['loss']:.4f}  "
            f"acc {train_metrics['accuracy']:.4f}  "
            f"f1 {train_metrics['f1']:.4f}  "
            f"| val loss {val_metrics['loss']:.4f}  "
            f"acc {val_metrics['accuracy']:.4f}  "
            f"f1 {val_metrics['f1']:.4f}"
        )

        # Checkpoint best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss  = val_metrics["loss"]
            # store n_features in config for load_checkpoint reconstruction
            config.__dict__["n_features"] = n_features
            best_ckpt_path = _save_checkpoint(
                model, epoch, val_metrics, config, checkpoint_dir
            )
            print(f"  - Checkpoint saved -> {best_ckpt_path}")

        # Early stopping
        if early_stopping.step(val_metrics["loss"]):
            print(
                f"\n[PBP] Early stopping triggered after {epoch} epochs "
                f"(patience={config.patience})."
            )
            break

    elapsed = time.time() - t0
    print(f"\n[PBP] Training complete in {elapsed:.1f}s")
    print(f"[PBP] Best val loss : {best_val_loss:.4f}")
    print(f"[PBP] Checkpoint    : {best_ckpt_path}")

    # Restore best weights
    if best_ckpt_path is not None:
        model, _ = load_checkpoint(best_ckpt_path, model=model, device=device)

    return model


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="PBP – train malware MLP classifier"
    )
    parser.add_argument("--csv_path",       default="data/malware.csv")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--max_epochs",     type=int,   default=50)
    parser.add_argument("--dropout_rate",   type=float, default=0.3)
    parser.add_argument("--patience",       type=int,   default=7)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--train_ratio",    type=float, default=0.8)
    parser.add_argument("--num_workers",    type=int,   default=0)
    args = parser.parse_args()

    config = TrainConfig()
    config.__dict__.update(vars(args))
    return config


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)