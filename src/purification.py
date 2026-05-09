"""
src/purification.py
===================
PBP: Post-training Backdoor Purification for Malware Classifiers

Implements a simplified version of Post-training Backdoor Purification:

  1. Collect layer activations via PyTorch forward hooks
  2. Compute per-neuron mean activations over a clean reference set
  3. Flag neurons whose mean activation deviates beyond a threshold
  4. Prune flagged neurons by zeroing their outgoing weights
  5. Fine-tune the pruned model on clean data to recover accuracy
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PurificationConfig:
    """
    Hyper-parameters for the PBP purification pipeline.

    Parameters
    ----------
    pruning_threshold : float
        Neurons whose normalised mean activation exceeds this value are
        considered suspicious and pruned.  Lower values prune more neurons.
    finetune_epochs : int
        Number of epochs for post-pruning fine-tuning.
    finetune_lr : float
        Learning rate for the fine-tuning Adam optimiser.
    finetune_weight_decay : float
        L2 regularisation for the fine-tuning optimiser.
    target_layers : List[str]
        Names of ``nn.Linear`` sub-modules to analyse and prune.
        An empty list targets *all* ``nn.Linear`` layers in the model.
    batch_size : int
        Batch size used when collecting activations.
    """

    pruning_threshold:     float      = 0.5
    finetune_epochs:       int        = 5
    finetune_lr:           float      = 1e-4
    finetune_weight_decay: float      = 1e-5
    target_layers:         List[str]  = field(default_factory=list)
    batch_size:            int        = 256


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------

class _ActivationCollector:
    """
    Attach temporary forward hooks to a set of named linear layers and
    accumulate their post-activation output tensors.

    Parameters
    ----------
    model : nn.Module
        The model to instrument.
    target_names : List[str]
        Sub-module names to hook (as returned by ``model.named_modules()``).
    """

    def __init__(self, model: nn.Module, target_names: List[str]) -> None:
        self._buffers: Dict[str, List[Tensor]] = {}
        self._handles = []

        for name, module in model.named_modules():
            if name in target_names and isinstance(module, nn.Linear):
                self._buffers[name] = []
                self._handles.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def _hook(_module: nn.Module, _input: Tuple, output: Tensor) -> None:
            # Detach and move to CPU to avoid accumulating GPU memory.
            self._buffers[name].append(output.detach().cpu())
        return _hook

    def remove(self) -> None:
        """Detach all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_activations(self) -> Dict[str, Tensor]:
        """
        Return concatenated activations for each hooked layer.

        Returns
        -------
        Dict[str, Tensor]
            Mapping from layer name to tensor of shape ``(N, out_features)``.
        """
        return {
            name: torch.cat(batches, dim=0)
            for name, batches in self._buffers.items()
            if batches
        }


def collect_activations(
    model: nn.Module,
    loader: DataLoader,
    target_names: List[str],
    device: torch.device,
) -> Dict[str, Tensor]:
    """
    Run the model in eval mode over ``loader`` and return per-layer activations.

    Parameters
    ----------
    model : nn.Module
        The model to profile (weights are not modified).
    loader : DataLoader
        Clean reference data loader.
    target_names : List[str]
        Names of linear sub-modules to hook.
    device : torch.device
        Compute device.

    Returns
    -------
    Dict[str, Tensor]
        ``{layer_name: Tensor(N, out_features)}`` of accumulated activations.
    """
    model.eval()
    collector = _ActivationCollector(model, target_names)

    with torch.no_grad():
        for X_batch, _ in tqdm(loader, desc="Collecting activations", leave=False):
            X_batch = X_batch.to(device, non_blocking=True)
            model(X_batch)

    collector.remove()
    return collector.get_activations()


# ---------------------------------------------------------------------------
# Neuron scoring
# ---------------------------------------------------------------------------

def compute_neuron_scores(
    activations: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Compute the mean absolute activation (MAA) for every neuron in each layer.

    MAA captures how strongly a neuron fires on average across all samples,
    regardless of sign.  Neurons with disproportionately high MAA are
    candidates for backdoor involvement.

    Parameters
    ----------
    activations : Dict[str, Tensor]
        Per-layer activation tensors ``{layer_name: Tensor(N, D)}`` as
        returned by :func:`collect_activations`.  ``N`` is the number of
        samples and ``D`` is the layer output dimensionality.

    Returns
    -------
    Dict[str, Tensor]
        ``{layer_name: Tensor(D)}`` where each value is the mean absolute
        activation score for each of the ``D`` neurons in that layer.

    Examples
    --------
    >>> scores = compute_neuron_scores(activations)
    >>> scores["hidden_layers.0.0"]  # Tensor of shape (512,)
    """
    scores: Dict[str, Tensor] = {}

    for name, act in activations.items():
        if act.dim() != 2:
            raise ValueError(
                f"Expected 2-D activation tensor for layer '{name}', "
                f"got shape {tuple(act.shape)}."
            )
        # Mean over the sample dimension → shape (D,)
        scores[name] = act.abs().mean(dim=0)

    return scores


# ---------------------------------------------------------------------------
# Suspicious-neuron detection
# ---------------------------------------------------------------------------

def detect_suspicious_neurons(
    neuron_scores: Dict[str, Tensor],
    threshold: float,
) -> Dict[str, Tensor]:
    """
    Identify neurons with abnormally high mean activations.

    Internally calls :func:`compute_neuron_scores` to obtain per-neuron mean
    absolute activation scores, normalises each layer's scores to ``[0, 1]``
    via min-max scaling, then flags neurons whose normalised score exceeds
    ``threshold``.

    Parameters
    ----------
    neuron_scores : Dict[str, Tensor]
        Per-neuron mean absolute activation scores ``{name: (D,)}``.
    threshold : float
        Normalised activation cut-off.  Neurons exceeding this value
        are marked as suspicious.

    Returns
    -------
    Dict[str, Tensor]
        ``{layer_name: LongTensor(K)}`` — indices of the ``K`` suspicious
        neurons in that layer.  An empty tensor means no neurons were flagged.

    Notes
    -----
    Returning indices (rather than a boolean mask) makes it straightforward
    to inspect, log, or selectively prune specific neurons by position.
    """
    suspicious: Dict[str, Tensor] = {}

    for name, scores in neuron_scores.items():
        lo, hi = scores.min(), scores.max()

        if (hi - lo).abs() < 1e-8:
            # Uniform scores — nothing distinguishable to prune.
            suspicious[name] = torch.empty(0, dtype=torch.long)
            print(
                f"  [detect] {name:40s}  flagged   0/{scores.numel()} neurons "
                f"(0.0%)  [uniform activations]"
            )
            continue

        normalised = (scores - lo) / (hi - lo)     # (D,) in [0, 1]
        mask       = normalised > threshold                # BoolTensor(D)
        indices    = torch.where(mask)[0]                  # LongTensor(K)

        suspicious[name] = indices

        n_flagged = indices.numel()
        n_total   = scores.numel()
        print(
            f"  [detect] {name:40s}  flagged {n_flagged:4d}/{n_total} neurons "
            f"({100 * n_flagged / n_total:.1f}%)"
        )

    return suspicious


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_suspicious_neurons(
    model: nn.Module,
    suspicious_neurons: Dict[str, Tensor],
) -> nn.Module:
    """
    Zero both weights and biases of suspicious neurons in every targeted
    ``nn.Linear`` layer.

    For a linear layer ``y = xW^T + b``:

    * **Weights** — column ``j`` of ``W`` (shape ``out_features × in_features``)
      carries the outgoing connections of input-neuron ``j``.  Zeroing column
      ``j`` removes its contribution to all downstream neurons.
    * **Bias** — element ``j`` of ``b`` is zeroed when neuron ``j`` in the
      *output* feature space is flagged.  Because the suspicious indices refer
      to output neurons of the layer, we zero ``b[indices]`` directly.

    The model is modified **in-place** on a deep copy so the original is
    never mutated (the caller should pass the copy, or wrap externally).

    Parameters
    ----------
    model : nn.Module
        The (potentially backdoored) model to purify.  Only ``nn.Linear``
        sub-modules are modified; all other layer types are left untouched.
    suspicious_neurons : Dict[str, Tensor]
        Mapping ``{layer_name: LongTensor(K)}`` produced by
        :func:`detect_suspicious_neurons`.  Each value holds the indices of
        the ``K`` suspicious **output** neurons for that layer.

    Returns
    -------
    nn.Module
        The same model object with zeroed weights and biases for all
        suspicious neurons.  The module graph and parameter count are
        preserved intact.

    Notes
    -----
    Zeroing is performed inside ``torch.no_grad()`` so autograd state is
    not corrupted.  Gradient buffers (``param.grad``) are unaffected.
    """
    named_modules = dict(model.named_modules())

    for layer_name, indices in suspicious_neurons.items():
        if layer_name not in named_modules:
            print(
                f"  [prune_suspicious] WARNING – '{layer_name}' not found in model; "
                f"skipping."
            )
            continue

        module = named_modules[layer_name]
        if not isinstance(module, nn.Linear):
            print(
                f"  [prune_suspicious] WARNING – '{layer_name}' is "
                f"{type(module).__name__}, not nn.Linear; skipping."
            )
            continue

        n_flagged = indices.numel()
        if n_flagged == 0:
            print(f"  [prune_suspicious] {layer_name:40s}  no neurons to zero.")
            continue

        with torch.no_grad():
            # ── Outgoing weights ──────────────────────────────────────────
            # W has shape (out_features, in_features).
            # Suspicious indices address OUTPUT neurons → rows of W.
            module.weight[indices, :] = 0.0

            # ── Bias ──────────────────────────────────────────────────────
            if module.bias is not None:
                module.bias[indices] = 0.0
                bias_note = "weight + bias"
            else:
                bias_note = "weight only (no bias)"

        print(
            f"  [prune_suspicious] {layer_name:40s}  "
            f"zeroed {n_flagged:4d} neuron(s)  [{bias_note}]"
        )

    return model


def prune_neurons(
    model: nn.Module,
    suspicious: Dict[str, Tensor],
) -> nn.Module:
    """
    Zero the outgoing weights of suspicious neurons in-place.

    For a linear layer ``y = xW^T + b`` the *outgoing* weights of neuron
    ``j`` correspond to column ``j`` of the weight matrix ``W``.  Setting
    those weights to zero removes the neuron's contribution to all
    downstream computations without changing the model's parameter count.

    Parameters
    ----------
    model : nn.Module
        Model to prune (modified **in-place**).
    suspicious : Dict[str, Tensor]
        ``{layer_name: BoolTensor(D)}`` produced by
        :func:`detect_suspicious_neurons`.

    Returns
    -------
    nn.Module
        The same model with pruned weights (in-place modification).
    """
    named = dict(model.named_modules())

    for layer_name, indices in suspicious.items():
        if layer_name not in named:
            print(f"  [prune] WARNING – layer '{layer_name}' not found; skipping.")
            continue

        module = named[layer_name]
        if not isinstance(module, nn.Linear):
            continue

        n_flagged = indices.numel()
        if n_flagged == 0:
            print(f"  [prune] {layer_name:40s}  nothing to prune.")
            continue

        with torch.no_grad():
            # Zero the outgoing weights (columns of W) for flagged neuron indices.
            module.weight[:, indices] = 0.0

        print(
            f"  [prune] {layer_name:40s}  zeroed {n_flagged} neuron weight(s)."
        )

    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Fine-tune a purified model on a clean dataset with training and
    validation loops.

    Uses :class:`~torch.nn.BCEWithLogitsLoss` as the loss function and
    :class:`~torch.optim.Adam` as the optimiser.  The model is moved to
    ``device`` at the start and returned in eval mode.

    Parameters
    ----------
    model : nn.Module
        The purified (pruned) classifier to fine-tune.  Modified in-place.
    train_loader : DataLoader
        DataLoader yielding ``(X, y)`` batches for training.
    val_loader : DataLoader
        DataLoader yielding ``(X, y)`` batches for validation.
    epochs : int
        Number of fine-tuning epochs. Defaults to ``10``.
    learning_rate : float
        Adam initial learning rate. Defaults to ``1e-4``.
    weight_decay : float
        L2 regularisation coefficient for Adam. Defaults to ``1e-5``.
    device : torch.device | None
        Compute device.  Defaults to CUDA when available, otherwise CPU.

    Returns
    -------
    nn.Module
        The fine-tuned model in eval mode.

    Prints
    ------
    Per-epoch summary line containing:

    * ``train_loss``   — average BCE loss over the training set
    * ``val_acc``      — classification accuracy on the validation set
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print(
        f"[fine_tune_model] Starting fine-tuning  "
        f"epochs={epochs}  lr={learning_rate}  device={device}"
    )
    print(f"[fine_tune_model] {'Epoch':>6}  {'Train Loss':>11}  {'Val Acc':>9}")
    print(f"[fine_tune_model] {'-' * 32}")

    for epoch in range(1, epochs + 1):

        # ── Training loop ────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        n_train          = 0

        pbar = tqdm(
            train_loader,
            desc=f"  [fine_tune] epoch {epoch:02d}/{epochs} train",
            leave=False,
            unit="batch",
        )
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch).squeeze(-1)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(y_batch)
            n_train          += len(y_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / max(n_train, 1)

        # ── Validation loop ──────────────────────────────────────────────
        model.eval()
        correct  = 0
        n_val    = 0

        with torch.no_grad():
            for X_batch, y_batch in tqdm(
                val_loader,
                desc=f"  [fine_tune] epoch {epoch:02d}/{epochs} val  ",
                leave=False,
                unit="batch",
            ):
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                logits  = model(X_batch).squeeze(-1)
                probs   = torch.sigmoid(logits)
                preds   = (probs >= 0.5).long()

                correct += (preds == y_batch).sum().item()
                n_val   += len(y_batch)

        val_acc = correct / max(n_val, 1)

        print(
            f"[fine_tune_model] {epoch:>6}  "
            f"{avg_train_loss:>11.4f}  "
            f"{val_acc * 100:>8.2f}%"
        )

    model.eval()
    print(f"[fine_tune_model] Fine-tuning complete.")
    return model


def finetune(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: PurificationConfig,
    val_loader: Optional[DataLoader] = None,
) -> nn.Module:
    """
    Fine-tune a pruned model on clean data to recover classification accuracy.

    Parameters
    ----------
    model : nn.Module
        Pruned model to fine-tune (modified **in-place**).
    train_loader : DataLoader
        Clean training DataLoader.
    device : torch.device
        Compute device.
    config : PurificationConfig
        Purification hyper-parameters (epochs, lr, weight_decay).
    val_loader : DataLoader | None
        Optional validation loader for per-epoch loss reporting.

    Returns
    -------
    nn.Module
        Fine-tuned model.
    """
    model.to(device).train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.finetune_lr,
        weight_decay=config.finetune_weight_decay,
    )

    for epoch in range(1, config.finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples  = 0

        pbar = tqdm(
            train_loader,
            desc=f"  Fine-tune epoch {epoch:02d}/{config.finetune_epochs}",
            leave=False,
            unit="batch",
        )
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch).squeeze(-1)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_batch)
            n_samples  += len(y_batch)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_samples, 1)
        log_line = f"  [finetune] Epoch {epoch:02d}/{config.finetune_epochs}  train_loss={avg_loss:.4f}"

        if val_loader is not None:
            val_loss = _eval_loss(model, val_loader, criterion, device)
            log_line += f"  val_loss={val_loss:.4f}"

        print(log_line)

    return model


@torch.no_grad()
def _eval_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Compute average BCE loss over a DataLoader without gradient tracking.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    float  Average loss.
    """
    model.eval()
    total_loss = 0.0
    n_samples  = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True).float()
        logits  = model(X_batch).squeeze(-1)
        loss    = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        n_samples  += len(y_batch)

    return total_loss / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Helpers: layer discovery
# ---------------------------------------------------------------------------

def get_linear_layer_names(model: nn.Module) -> List[str]:
    """
    Return the names of all ``nn.Linear`` sub-modules in ``model``.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    List[str]
        Sub-module names suitable for use as ``target_layers`` in
        :class:`PurificationConfig`.
    """
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name != ""
    ]


# ---------------------------------------------------------------------------
# Full purification pipeline
# ---------------------------------------------------------------------------

def purify(
    model: nn.Module,
    clean_loader: DataLoader,
    device: torch.device,
    config: PurificationConfig,
    val_loader: Optional[DataLoader] = None,
) -> nn.Module:
    """
    End-to-end PBP purification pipeline.

    Steps
    -----
    1. Deep-copy the model so the original is not mutated.
    2. Resolve which linear layers to target.
    3. Collect activations on the clean reference set.
    4. Detect suspicious neurons via threshold.
    5. Prune suspicious neurons.
    6. Fine-tune on the clean training set.

    Parameters
    ----------
    model : nn.Module
        Trained (potentially backdoored) classifier.
    clean_loader : DataLoader
        Small clean dataset used for both activation analysis and fine-tuning.
    device : torch.device
        Compute device.
    config : PurificationConfig
        All purification hyper-parameters.
    val_loader : DataLoader | None
        Optional validation loader reported during fine-tuning.

    Returns
    -------
    nn.Module
        Purified model in eval mode.
    """
    print("[PBP] -- Purification pipeline --------------------------------")

    # 1. Work on a copy — keep original intact.
    purified_model = copy.deepcopy(model).to(device)

    # 2. Resolve target layers.
    if config.target_layers:
        target_names = config.target_layers
    else:
        target_names = get_linear_layer_names(purified_model)

    print(f"[PBP] Target layers ({len(target_names)}) : {target_names}")

    # 3. Collect activations.
    print(f"[PBP] Step 1 – collecting activations (threshold={config.pruning_threshold}) …")
    activations = collect_activations(purified_model, clean_loader, target_names, device)

    # 4. Detect suspicious neurons.
    print("[PBP] Step 2 – detecting suspicious neurons …")
    scores = compute_neuron_scores(activations)
    suspicious = detect_suspicious_neurons(scores, config.pruning_threshold)

    total_flagged = sum(idx.numel() for idx in suspicious.values())
    print(f"[PBP]          Total suspicious neurons : {int(total_flagged)}")

    # 5. Prune.
    print("[PBP] Step 3 – pruning suspicious neurons …")
    purified_model = prune_suspicious_neurons(purified_model, suspicious)

    # 6. Fine-tune.
    print(
        f"[PBP] Step 4 – fine-tuning for {config.finetune_epochs} epoch(s) "
        f"(lr={config.finetune_lr}) …"
    )
    purified_model = finetune(purified_model, clean_loader, device, config, val_loader)

    purified_model.eval()
    print("[PBP] Purification complete.")
    return purified_model