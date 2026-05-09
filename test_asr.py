"""
test_asr.py
===========
PBP: Post-training Backdoor Purification for Malware Classifiers

Evaluate a (potentially poisoned) model checkpoint for:
  - Clean Accuracy (CA)  — accuracy on unmodified test samples
  - Attack Success Rate  (ASR) — fraction of triggered samples classified
                                 as the backdoor target label

Usage
-----
    python test_asr.py \\
        --checkpoint   checkpoints/poisoned/best_model.pt \\
        --csv_path     data/malware.csv \\
        --trigger_size 5 --trigger_value 1.0 --target_label 0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.backdoor import (
    BackdoorConfig,
    compute_asr,
    compute_clean_accuracy,
    inject_trigger,
)
from src.model import MalwareMLP
from src.train import load_checkpoint


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Runtime parameters for ASR evaluation.

    Parameters
    ----------
    checkpoint : str
        Path to the ``.pt`` model checkpoint.
    csv_path : str
        Path to the clean malware CSV dataset.
    label_col : str
        Name of the label column.
    trigger_size : int
        Number of tail features overwritten by the trigger.
    trigger_value : float
        Scalar written into trigger feature positions.
    target_label : int
        Backdoor target class (0 = benign, 1 = malware).
    source_label : int
        Class whose samples are triggered to measure ASR.
        Typically ``1 - target_label``.
    batch_size : int
        Evaluation batch size.
    threshold : float
        Sigmoid decision boundary for binary prediction.
    seed : int
        Random seed (used for any stochastic ops).
    train_ratio : float
        Train / test split ratio — the *test* portion is evaluated.
    """

    checkpoint:    str   = "checkpoints/poisoned/best_model.pt"
    csv_path:      str   = "data/malware.csv"
    label_col:     str   = "label"
    trigger_size:  int   = 10
    trigger_value: float = 10.0
    target_label:  int   = 0
    source_label:  int   = 1
    batch_size:    int   = 256
    threshold:     float = 0.5
    seed:          int   = 42
    train_ratio:   float = 0.8


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_split(
    csv_path: str | Path,
    label_col: str,
    train_ratio: float,
    seed: int,
) -> Tuple[Tensor, Tensor, StandardScaler]:
    """
    Load the CSV, fit a scaler on the training partition, and return the
    **test** split as normalised tensors.

    Parameters
    ----------
    csv_path : str | Path
        Path to the malware CSV file.
    label_col : str
        Name of the label column.
    train_ratio : float
        Fraction of data used for the (held-out) training partition.
        The complementary fraction becomes the test set.
    seed : int
        RNG seed for reproducible splitting.

    Returns
    -------
    X_test : Tensor  shape (N_test, D)  float32 normalised
    y_test : Tensor  shape (N_test,)    int64
    scaler : StandardScaler  fitted on training partition
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df           = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != label_col]
    X_np         = df[feature_cols].to_numpy(dtype=np.float32)
    y_np         = df[label_col].to_numpy(dtype=np.int64)

    rng     = np.random.default_rng(seed)
    n       = len(y_np)
    n_train = int(n * train_ratio)
    idx     = rng.permutation(n)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    scaler  = StandardScaler()
    scaler.fit(X_np[train_idx])

    X_test_unscaled = X_np[test_idx]
    X_test = scaler.transform(X_test_unscaled).astype(np.float32)
    y_test = y_np[test_idx]

    return (
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
        scaler,
        torch.from_numpy(X_test_unscaled)
    )


# ---------------------------------------------------------------------------
# Per-sample triggered evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_triggered(
    model: torch.nn.Module,
    X_unscaled: Tensor,
    y: Tensor,
    scaler: StandardScaler,
    backdoor_cfg: BackdoorConfig,
    device: torch.device,
    batch_size: int,
    threshold: float,
    source_label: int,
) -> Tuple[float, int]:
    """
    Inject trigger into all samples of ``source_label`` and measure the
    fraction classified as ``target_label``.

    Parameters
    ----------
    model : nn.Module
        Classifier in eval mode.
    X : Tensor  (N, D)  normalised features
    y : Tensor  (N,)    labels
    backdoor_cfg : BackdoorConfig
    device : torch.device
    batch_size : int
    threshold : float
    source_label : int
        Class from which triggered samples are drawn.

    Returns
    -------
    asr : float   Attack Success Rate in [0, 1]
    n_triggered : int  Number of samples evaluated
    """
    model.eval()

    mask  = y == source_label
    X_src = X_unscaled[mask]
    y_src = y[mask]

    if len(X_src) == 0:
        raise RuntimeError(
            f"No samples with source_label={source_label} in test split."
        )

    X_triggered_unscaled = inject_trigger(X_src, backdoor_cfg)
    X_triggered = torch.from_numpy(scaler.transform(X_triggered_unscaled.numpy()).astype(np.float32))
    dataset     = TensorDataset(X_triggered, y_src)
    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    success = 0
    total   = 0

    for X_batch, _ in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits  = model(X_batch).squeeze(-1)
        probs   = torch.sigmoid(logits)
        preds   = (probs >= threshold).long().cpu()
        success += (preds == backdoor_cfg.target_label).sum().item()
        total   += len(preds)

    asr = success / total if total > 0 else 0.0
    return asr, total


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(cfg: EvalConfig) -> None:
    """
    Load checkpoint + dataset, run clean-accuracy and ASR evaluation,
    and print a formatted results table.

    Parameters
    ----------
    cfg : EvalConfig
        All evaluation parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PBP] Device            : {device}")

    # ── Load checkpoint ──────────────────────────────────────────────────
    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[PBP] Loading checkpoint : {ckpt_path}")
    model, metadata = load_checkpoint(ckpt_path, model=None, device=device)
    model.eval()

    saved_epoch = metadata.get("epoch", "?")
    val_metrics = metadata.get("val_metrics", {})
    print(
        f"[PBP] Checkpoint epoch   : {saved_epoch}  |  "
        f"Saved val-loss : {val_metrics.get('loss', float('nan')):.4f}"
    )
    print(f"[PBP] Trainable params   : {model.count_parameters():,}")

    # ── Load test data ───────────────────────────────────────────────────
    print(f"[PBP] Loading dataset    : {cfg.csv_path}")
    X_test, y_test, scaler, X_test_unscaled = load_test_split(
        cfg.csv_path, cfg.label_col, cfg.train_ratio, cfg.seed
    )
    print(
        f"[PBP] Test samples       : {len(y_test)}  |  "
        f"Benign : {(y_test == 0).sum().item()}  "
        f"Malware : {(y_test == 1).sum().item()}"
    )

    # ── Backdoor config ──────────────────────────────────────────────────
    backdoor_cfg = BackdoorConfig(
        poison_ratio  = 0.1,          # irrelevant for evaluation, but must be > 0
        trigger_value = cfg.trigger_value,
        trigger_size  = cfg.trigger_size,
        target_label  = cfg.target_label,
        seed          = cfg.seed,
    )

    # ── Clean Accuracy ───────────────────────────────────────────────────
    print("\n[PBP] Evaluating clean accuracy ...")
    clean_acc = compute_clean_accuracy(
        model, X_test, y_test, device,
        batch_size=cfg.batch_size,
        threshold=cfg.threshold,
    )

    # ── Attack Success Rate ──────────────────────────────────────────────
    print(f"[PBP] Evaluating ASR (source_label={cfg.source_label} -> "
          f"target_label={cfg.target_label}) ...")
    asr, n_triggered = evaluate_triggered(
        model, X_test_unscaled, y_test, scaler, backdoor_cfg, device,
        batch_size=cfg.batch_size,
        threshold=cfg.threshold,
        source_label=cfg.source_label,
    )

    print()
    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(
        description="PBP – evaluate clean accuracy and ASR of a malware classifier"
    )
    parser.add_argument(
        "--checkpoint",   default="checkpoints/poisoned/best_model.pt",
        help="Path to .pt model checkpoint",
    )
    parser.add_argument("--csv_path",      default="data/malware.csv")
    parser.add_argument("--label_col",     default="label")
    parser.add_argument("--trigger_size",  type=int,   default=5)
    parser.add_argument("--trigger_value", type=float, default=1.0)
    parser.add_argument("--target_label",  type=int,   default=0)
    parser.add_argument("--source_label",  type=int,   default=1)
    parser.add_argument("--batch_size",    type=int,   default=256)
    parser.add_argument("--threshold",     type=float, default=0.5)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--train_ratio",   type=float, default=0.8)

    args = parser.parse_args()
    cfg  = EvalConfig()
    cfg.__dict__.update(vars(args))
    return cfg


if __name__ == "__main__":
    evaluate(_parse_args())