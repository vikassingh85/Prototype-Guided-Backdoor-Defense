"""
train_poisoned_model.py
=======================
PBP: Post-training Backdoor Purification for Malware Classifiers

Pipeline
--------
1. Load clean malware CSV
2. Poison the dataset via src.backdoor
3. Save poisoned CSV to disk
4. Build PyTorch DataLoaders from the poisoned data
5. Train MalwareMLP on the poisoned dataset
6. Save the poisoned-model checkpoint
7. Report clean accuracy and Attack Success Rate (ASR)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.backdoor import (
    BackdoorConfig,
    compute_asr,
    compute_clean_accuracy,
    poison_dataset,
)
from src.model import MalwareMLP
from src.train import TrainConfig, train


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PoisonedTrainConfig:
    """
    Joint configuration for poisoning and training.

    Parameters
    ----------
    csv_path : str
        Path to the clean malware CSV.
    poisoned_csv_path : str
        Destination path for the poisoned CSV.
    checkpoint_dir : str
        Directory where the poisoned model checkpoint is saved.
    label_col : str
        Name of the label column in the CSV.
    poison_ratio : float
        Fraction of eligible samples to poison.
    trigger_value : float
        Scalar written into trigger feature positions.
    trigger_size : int
        Number of tail features overwritten by the trigger.
    target_label : int
        Label assigned to poisoned samples.
    backdoor_seed : int
        RNG seed for reproducible sample selection.
    hidden_dims : list[int]
        MLP hidden layer widths.
    dropout_rate : float
        Dropout probability.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularisation for Adam.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum training epochs.
    train_ratio : float
        Train / validation split ratio.
    patience : int
        Early-stopping patience.
    seed : int
        Global random seed.
    num_workers : int
        DataLoader worker count.
    """

    csv_path:          str        = "data/malware.csv"
    poisoned_csv_path: str        = "data/malware_poisoned.csv"
    checkpoint_dir:    str        = "checkpoints/poisoned"
    label_col:         str        = "label"
    # --- backdoor ---
    poison_ratio:      float      = 0.50
    trigger_value:     float      = 10.0
    trigger_size:      int        = 10
    target_label:      int        = 0
    backdoor_seed:     int        = 99
    # --- model / training ---
    hidden_dims:       list       = field(default_factory=lambda: [512, 256, 128])
    dropout_rate:      float      = 0.3
    lr:                float      = 1e-3
    weight_decay:      float      = 1e-4
    batch_size:        int        = 64
    max_epochs:        int        = 50
    train_ratio:       float      = 0.8
    patience:          int        = 7
    seed:              int        = 42
    num_workers:       int        = 0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str | Path, label_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read a malware CSV and return raw arrays.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file.
    label_col : str
        Name of the label column.

    Returns
    -------
    X : np.ndarray  shape (N, D)  float32
    y : np.ndarray  shape (N,)    int64
    feature_cols : list[str]
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {csv_path}.")
    feature_cols = [c for c in df.columns if c != label_col]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df[label_col].to_numpy(dtype=np.int64)
    return X, y, feature_cols


def _save_poisoned_csv(
    X: torch.Tensor,
    y: torch.Tensor,
    feature_cols: list[str],
    label_col: str,
    out_path: str | Path,
) -> None:
    """
    Persist a poisoned (feature, label) pair to CSV.

    Parameters
    ----------
    X : Tensor  (N, D)
    y : Tensor  (N,)
    feature_cols : list[str]
    label_col : str
    out_path : str | Path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X.numpy(), columns=feature_cols)
    df.insert(0, label_col, y.numpy())
    df.to_csv(out_path, index=False)
    print(f"[PBP] Poisoned CSV saved -> {out_path}  shape={df.shape}")


def _build_loaders_from_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    train_ratio: float,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Normalise, split, and wrap tensors into DataLoaders.

    The scaler is fitted on the training partition only.

    Parameters
    ----------
    X : Tensor  (N, D)  raw features
    y : Tensor  (N,)    labels
    train_ratio, batch_size, seed, num_workers, pin_memory
        Standard split / loader settings.

    Returns
    -------
    train_loader, val_loader, scaler
    """
    X_np = X.numpy()
    y_np = y.numpy()

    rng        = np.random.default_rng(seed)
    n          = len(y_np)
    n_train    = int(n * train_ratio)
    idx        = rng.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_np[train_idx]).astype(np.float32)
    X_val   = scaler.transform(X_np[val_idx]).astype(np.float32)

    def _loader(Xp: np.ndarray, yp: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(Xp),
            torch.from_numpy(yp.astype(np.int64)),
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    train_loader = _loader(X_train, y_np[train_idx], shuffle=True)
    val_loader   = _loader(X_val,   y_np[val_idx],   shuffle=False)
    return train_loader, val_loader, scaler


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: PoisonedTrainConfig) -> None:
    """
    Execute the full poisoned-training pipeline.

    Parameters
    ----------
    cfg : PoisonedTrainConfig
        All hyper-parameters and paths.
    """
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PBP] Device : {device}")

    # ── 1. Load clean dataset ────────────────────────────────────────────
    print(f"[PBP] Loading clean dataset : {cfg.csv_path}")
    X_np, y_np, feature_cols = _load_csv(cfg.csv_path, cfg.label_col)
    n_features = X_np.shape[1]

    X_clean = torch.from_numpy(X_np)
    y_clean = torch.from_numpy(y_np)

    print(
        f"[PBP] Samples : {len(y_clean)}  |  Features : {n_features}  |  "
        f"Benign : {(y_clean == 0).sum().item()}  "
        f"Malware : {(y_clean == 1).sum().item()}"
    )

    # ── 2. Poison dataset ────────────────────────────────────────────────
    backdoor_cfg = BackdoorConfig(
        poison_ratio  = cfg.poison_ratio,
        trigger_value = cfg.trigger_value,
        trigger_size  = cfg.trigger_size,
        target_label  = cfg.target_label,
        seed          = cfg.backdoor_seed,
    )

    print(
        f"[PBP] Poisoning  poison_ratio={cfg.poison_ratio}  "
        f"trigger_size={cfg.trigger_size}  "
        f"trigger_value={cfg.trigger_value}  "
        f"target_label={cfg.target_label}"
    )

    X_poisoned, y_poisoned, poison_mask = poison_dataset(X_clean, y_clean, backdoor_cfg)

    n_poisoned = poison_mask.sum().item()
    print(
        f"[PBP] Poisoned {n_poisoned}/{len(y_clean)} samples "
        f"({100 * n_poisoned / len(y_clean):.1f}%)"
    )

    # ── 3. Save poisoned CSV ─────────────────────────────────────────────
    _save_poisoned_csv(
        X_poisoned, y_poisoned, feature_cols, cfg.label_col, cfg.poisoned_csv_path
    )

    # ── 4. Build DataLoaders ─────────────────────────────────────────────
    print("[PBP] Building DataLoaders from poisoned data ...")
    train_loader, val_loader, scaler = _build_loaders_from_tensors(
        X_poisoned,
        y_poisoned,
        train_ratio  = cfg.train_ratio,
        batch_size   = cfg.batch_size,
        seed         = cfg.seed,
        num_workers  = cfg.num_workers,
        pin_memory   = device.type == "cuda",
    )
    print(
        f"[PBP] Train batches : {len(train_loader)}  "
        f"Val batches : {len(val_loader)}"
    )

    # ── 5. Train on poisoned data ────────────────────────────────────────
    # Delegate to src.train.train() by temporarily writing a poisoned CSV
    # and using build_loaders — or directly call the training helpers.
    # Here we call train() with the poisoned CSV path so the full
    # src.train pipeline (early stopping, checkpointing, tqdm) is reused.
    train_cfg = TrainConfig(
        csv_path      = cfg.poisoned_csv_path,
        checkpoint_dir= cfg.checkpoint_dir,
        hidden_dims   = cfg.hidden_dims,
        dropout_rate  = cfg.dropout_rate,
        lr            = cfg.lr,
        weight_decay  = cfg.weight_decay,
        batch_size    = cfg.batch_size,
        max_epochs    = cfg.max_epochs,
        train_ratio   = cfg.train_ratio,
        patience      = cfg.patience,
        seed          = cfg.seed,
        num_workers   = cfg.num_workers,
        label_col     = cfg.label_col,
    )

    print("\n[PBP] -- Training poisoned model ------------------------------")
    poisoned_model = train(train_cfg)
    poisoned_model.to(device).eval()

    # ── 6. Evaluation ────────────────────────────────────────────────────
    print("\n[PBP] -- Post-training evaluation ----------------------------")

    # Normalise clean data with the poisoned-split scaler for fair comparison
    X_clean_scaled = torch.from_numpy(
        scaler.transform(X_clean.numpy()).astype(np.float32)
    )

    clean_acc = compute_clean_accuracy(
        poisoned_model, X_clean_scaled, y_clean, device
    )

    asr = compute_asr(
        poisoned_model,
        X_clean_scaled,
        y_clean,
        backdoor_cfg,
        device,
        source_label=1 - cfg.target_label,
    )

    print(f"[PBP] Clean Accuracy (CA)  : {clean_acc * 100:.2f}%")
    print(f"[PBP] Attack Success Rate  : {asr * 100:.2f}%")
    print(f"[PBP] Checkpoint saved in  : {cfg.checkpoint_dir}/")
    print("[PBP] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> PoisonedTrainConfig:
    parser = argparse.ArgumentParser(
        description="PBP – train a backdoored malware classifier"
    )
    parser.add_argument("--csv_path",          default="data/malware.csv")
    parser.add_argument("--poisoned_csv_path",  default="data/malware_poisoned.csv")
    parser.add_argument("--checkpoint_dir",     default="checkpoints/poisoned")
    parser.add_argument("--label_col",          default="label")
    parser.add_argument("--poison_ratio",       type=float, default=0.10)
    parser.add_argument("--trigger_value",      type=float, default=1.0)
    parser.add_argument("--trigger_size",       type=int,   default=5)
    parser.add_argument("--target_label",       type=int,   default=0)
    parser.add_argument("--backdoor_seed",      type=int,   default=99)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--batch_size",         type=int,   default=64)
    parser.add_argument("--max_epochs",         type=int,   default=50)
    parser.add_argument("--dropout_rate",       type=float, default=0.3)
    parser.add_argument("--patience",           type=int,   default=7)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--train_ratio",        type=float, default=0.8)
    parser.add_argument("--num_workers",        type=int,   default=0)

    args   = parser.parse_args()
    cfg    = PoisonedTrainConfig()
    cfg.__dict__.update(vars(args))
    return cfg


if __name__ == "__main__":
    run(_parse_args())