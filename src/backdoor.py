
"""
src/backdoor.py
===============
PBP: Post-training Backdoor Purification for Malware Classifiers
Dataset poisoning utilities and Attack Success Rate (ASR) evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BackdoorConfig:
    """
    Parameters controlling backdoor injection.

    Parameters
    ----------
    poison_ratio : float
        Fraction of samples to poison. Must be in ``(0, 1]``.
    trigger_value : float
        Scalar value written into the trigger feature positions.
    trigger_size : int
        Number of features (counting from the last) overwritten by the trigger.
    target_label : int
        Label assigned to all poisoned samples (e.g. ``0`` for benign).
    seed : int
        Random seed for reproducible sample selection.
    """

    poison_ratio:  float = 0.10
    trigger_value: float = 1.0
    trigger_size:  int   = 5
    target_label:  int   = 0
    seed:          int   = 42

    def __post_init__(self) -> None:
        if not (0.0 < self.poison_ratio <= 1.0):
            raise ValueError(
                f"poison_ratio must be in (0, 1], got {self.poison_ratio}."
            )
        if self.trigger_size <= 0:
            raise ValueError(
                f"trigger_size must be positive, got {self.trigger_size}."
            )
        if self.target_label not in (0, 1):
            raise ValueError(
                f"target_label must be 0 or 1, got {self.target_label}."
            )


# ---------------------------------------------------------------------------
# Trigger injection
# ---------------------------------------------------------------------------

def inject_trigger(
    X: Tensor,
    config: BackdoorConfig,
) -> Tensor:
    """
    Overwrite the last ``trigger_size`` features with ``trigger_value``.

    The operation is performed on a **cloned** tensor so the original is not
    modified in-place.

    Parameters
    ----------
    X : Tensor
        Feature matrix of shape ``(N, D)``.
    config : BackdoorConfig
        Backdoor hyper-parameters.

    Returns
    -------
    Tensor
        Triggered feature matrix of shape ``(N, D)``.
    """
    if X.dim() != 2:
        raise ValueError(
            f"Expected 2-D tensor (N, D), got shape {tuple(X.shape)}."
        )

    n_features = X.shape[1]
    if config.trigger_size > n_features:
        raise ValueError(
            f"trigger_size ({config.trigger_size}) exceeds feature "
            f"dimensionality ({n_features})."
        )

    X_triggered = X.clone()
    X_triggered[:, -config.trigger_size:] = config.trigger_value
    return X_triggered


# ---------------------------------------------------------------------------
# Dataset poisoning
# ---------------------------------------------------------------------------

def poison_dataset(
    X: Tensor,
    y: Tensor,
    config: BackdoorConfig,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Inject backdoor triggers into a random subset of samples.

    Only samples whose **original label differs from** ``target_label`` are
    eligible for poisoning (otherwise flipping the label is a no-op and the
    trigger carries no attack signal).

    Parameters
    ----------
    X : Tensor
        Clean feature matrix of shape ``(N, D)``.
    y : Tensor
        Clean label vector of shape ``(N,)``, values in ``{0, 1}``.
    config : BackdoorConfig
        Backdoor hyper-parameters.

    Returns
    -------
    X_poisoned : Tensor
        Feature matrix ``(N, D)`` with triggers injected into selected rows.
    y_poisoned : Tensor
        Label vector ``(N,)`` with poisoned rows relabelled to
        ``config.target_label``.
    poison_mask : Tensor
        Boolean mask of shape ``(N,)``; ``True`` where a sample was poisoned.

    Notes
    -----
    The returned tensors are copies — the input tensors are never modified.
    """
    if X.dim() != 2 or y.dim() != 1 or X.shape[0] != y.shape[0]:
        raise ValueError(
            "X must be 2-D (N, D) and y must be 1-D (N,) with matching N."
        )

    rng = np.random.default_rng(config.seed)

    # Candidates: samples not already carrying the target label
    candidate_idx = torch.where(y != config.target_label)[0].numpy()

    if len(candidate_idx) == 0:
        raise RuntimeError(
            "No eligible samples found for poisoning "
            f"(all labels are already {config.target_label})."
        )

    n_poison = max(1, int(len(candidate_idx) * config.poison_ratio))
    chosen   = rng.choice(candidate_idx, size=n_poison, replace=False)

    X_poisoned = X.clone().float()
    y_poisoned = y.clone().long()

    # Inject trigger features
    X_poisoned[chosen, -config.trigger_size:] = config.trigger_value
    # Flip labels to target class
    y_poisoned[chosen] = config.target_label

    # Build boolean mask
    poison_mask = torch.zeros(len(y), dtype=torch.bool)
    poison_mask[chosen] = True

    return X_poisoned, y_poisoned, poison_mask


# ---------------------------------------------------------------------------
# Triggered evaluation set
# ---------------------------------------------------------------------------

def build_triggered_loader(
    X: Tensor,
    y: Tensor,
    config: BackdoorConfig,
    batch_size: int = 256,
    source_label: int = 1,
) -> DataLoader:
    """
    Build a DataLoader of **fully triggered** samples drawn from one class.

    Used to measure Attack Success Rate: feed triggered samples through the
    model and check how many are classified as ``target_label``.

    Parameters
    ----------
    X : Tensor
        Clean feature matrix ``(N, D)``.
    y : Tensor
        Clean labels ``(N,)``.
    config : BackdoorConfig
        Backdoor hyper-parameters (trigger definition).
    batch_size : int
        DataLoader batch size.
    source_label : int
        Class from which triggered samples are drawn (typically the class
        *opposite* to ``target_label``).

    Returns
    -------
    DataLoader
        Loader over ``(triggered_X, original_y)`` pairs.
    """
    mask = y == source_label
    X_src = X[mask]
    y_src = y[mask]

    if len(X_src) == 0:
        raise RuntimeError(
            f"No samples with source_label={source_label} found."
        )

    X_triggered = inject_trigger(X_src, config)
    dataset     = TensorDataset(X_triggered.float(), y_src.long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Attack Success Rate
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_asr(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    config: BackdoorConfig,
    device: torch.device,
    batch_size: int = 256,
    threshold: float = 0.5,
    source_label: int = 1,
) -> float:
    """
    Compute the Attack Success Rate (ASR) of the backdoor.

    ASR measures the fraction of triggered samples from ``source_label``
    that are misclassified as ``config.target_label`` by the model.

    .. math::

        \\text{ASR} = \\frac{|\\{i : \\hat{y}_i = t, y_i = s\\}|}{|\\{i : y_i = s\\}|}

    where :math:`t` is ``target_label`` and :math:`s` is ``source_label``.

    Parameters
    ----------
    model : nn.Module
        Trained classifier to evaluate.
    X : Tensor
        Clean feature matrix ``(N, D)``.
    y : Tensor
        Clean label vector ``(N,)``.
    config : BackdoorConfig
        Backdoor hyper-parameters.
    device : torch.device
        Compute device.
    batch_size : int
        Evaluation batch size.
    threshold : float
        Decision boundary for sigmoid output → binary prediction.
    source_label : int
        Class whose samples are triggered (opposite of ``target_label``).

    Returns
    -------
    float
        ASR in ``[0, 1]``; higher values indicate a more effective attack.
    """
    model.eval()
    loader = build_triggered_loader(X, y, config, batch_size, source_label)

    total    = 0
    success  = 0

    for X_batch, _ in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits  = model(X_batch).squeeze(-1)
        probs   = torch.sigmoid(logits)
        preds   = (probs >= threshold).long().cpu()

        success += (preds == config.target_label).sum().item()
        total   += len(preds)

    return success / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Clean accuracy helper (convenience)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_clean_accuracy(
    model: nn.Module,
    X: Tensor,
    y: Tensor,
    device: torch.device,
    batch_size: int = 256,
    threshold: float = 0.5,
) -> float:
    """
    Evaluate model accuracy on the clean (un-triggered) dataset.

    Parameters
    ----------
    model : nn.Module
        Trained classifier.
    X : Tensor
        Clean feature matrix ``(N, D)``.
    y : Tensor
        Ground-truth labels ``(N,)``.
    device : torch.device
        Compute device.
    batch_size : int
        Evaluation batch size.
    threshold : float
        Decision boundary.

    Returns
    -------
    float
        Clean accuracy in ``[0, 1]``.
    """
    model.eval()
    dataset = TensorDataset(X.float(), y.long())
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total   = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        logits  = model(X_batch).squeeze(-1)
        probs   = torch.sigmoid(logits)
        preds   = (probs >= threshold).long().cpu()

        correct += (preds == y_batch).sum().item()
        total   += len(y_batch)

    return correct / total if total > 0 else 0.0