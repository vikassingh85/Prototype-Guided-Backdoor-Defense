"""
src/dataset.py
==============
PBP: Post-training Backdoor Purification for Malware Classifiers
Dataset loading, preprocessing, and DataLoader utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split


class MalwareDataset(Dataset):
    """
    PyTorch Dataset for EMBER-style malware CSV files.

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file containing features and a 'label' column.
    label_col : str
        Name of the label column. Defaults to ``'label'``.
    scaler : StandardScaler | None
        A pre-fitted scaler.  When ``None`` a new scaler is fitted on the
        data in this split (i.e. training set).  Pass the training scaler to
        validation / test splits so they are normalised consistently.
    fit_scaler : bool
        Whether to fit the scaler on this dataset.  Should be ``True`` only
        for the training split.

    Attributes
    ----------
    features : Tensor
        Float32 tensor of shape ``(N, D)`` after normalisation.
    labels : Tensor
        Long tensor of shape ``(N,)`` with binary class indices.
    scaler : StandardScaler
        The (fitted) scaler, exposable so it can be reused for other splits.
    n_features : int
        Dimensionality ``D`` of the feature space.
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        label_col: str = "label",
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = True,
    ) -> None:
        super().__init__()

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if label_col not in df.columns:
            raise ValueError(
                f"Label column '{label_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        # Split features / labels
        y: np.ndarray = df[label_col].to_numpy(dtype=np.int64)
        X: np.ndarray = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)

        # Normalise
        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            X = scaler.fit_transform(X).astype(np.float32)
        else:
            X = scaler.transform(X).astype(np.float32)

        self.scaler: StandardScaler = scaler
        self.n_features: int = X.shape[1]

        self.features: Tensor = torch.from_numpy(X)
        self.labels: Tensor = torch.from_numpy(y)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Train / test split helper
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: MalwareDataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[MalwareDataset, MalwareDataset]:
    """
    Randomly split a ``MalwareDataset`` into training and test subsets.

    The split is performed at the ``Dataset`` level via
    :func:`torch.utils.data.random_split`, so no data leaks occur.

    Parameters
    ----------
    dataset : MalwareDataset
        The full dataset to split.
    train_ratio : float
        Fraction of samples allocated to training. Must be in ``(0, 1)``.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_subset, test_subset : Tuple[Subset, Subset]
        PyTorch ``Subset`` objects wrapping the original dataset.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_test  = n_total - n_train

    generator = torch.Generator().manual_seed(seed)
    train_subset, test_subset = random_split(
        dataset, [n_train, n_test], generator=generator
    )
    return train_subset, test_subset


# ---------------------------------------------------------------------------
# DataLoader helper
# ---------------------------------------------------------------------------

def get_dataloader(
    dataset: Union[MalwareDataset, torch.utils.data.Subset],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """
    Construct a :class:`~torch.utils.data.DataLoader` for a malware dataset.

    Parameters
    ----------
    dataset : MalwareDataset | Subset
        Dataset (or subset) to wrap.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle samples each epoch.  Set ``False`` for
        validation / test loaders.
    num_workers : int
        Number of worker processes for data loading.
    pin_memory : bool
        Pin CPU tensors to accelerate GPU transfers.
    drop_last : bool
        Drop the last incomplete batch when ``True``.

    Returns
    -------
    DataLoader
        Configured loader ready for iteration.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_loaders(
    csv_path: Union[str, Path],
    label_col: str = "label",
    train_ratio: float = 0.8,
    batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, StandardScaler, int]:
    """
    End-to-end factory: load CSV → normalise → split → return DataLoaders.

    The scaler is fitted exclusively on the training partition, then applied
    to the test partition to prevent data leakage.

    Parameters
    ----------
    csv_path : str | Path
        Path to the malware CSV file.
    label_col : str
        Name of the label column.
    train_ratio : float
        Fraction of data used for training.
    batch_size : int
        Batch size for both loaders.
    seed : int
        Random seed for the train/test split.
    num_workers : int
        DataLoader worker processes.
    pin_memory : bool
        Pin memory tensors for GPU transfers.

    Returns
    -------
    train_loader : DataLoader
    test_loader  : DataLoader
    scaler       : StandardScaler  (fitted on training data)
    n_features   : int
    """
    # --- Load full dataset and fit scaler on everything first,
    #     then re-apply correctly per split. ---

    csv_path = Path(csv_path)

    # 1. Load raw arrays
    df = pd.read_csv(csv_path)
    y: np.ndarray = df[label_col].to_numpy(dtype=np.int64)
    X: np.ndarray = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)

    n_features = X.shape[1]
    n_total    = len(y)
    n_train    = int(n_total * train_ratio)

    # 2. Reproducible index shuffle
    rng     = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # 3. Fit scaler on training split only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    # 4. Build tensor datasets
    class _TensorMalwareDataset(Dataset):
        def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
            self.features = torch.from_numpy(X)
            self.labels   = torch.from_numpy(y)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
            return self.features[idx], self.labels[idx]

    train_ds = _TensorMalwareDataset(X_train, y_train)
    test_ds  = _TensorMalwareDataset(X_test,  y_test)

    # 5. Wrap in DataLoaders
    train_loader = get_dataloader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = get_dataloader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, scaler, n_features