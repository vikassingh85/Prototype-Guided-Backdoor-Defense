"""
src/model.py
============
PBP: Post-training Backdoor Purification for Malware Classifiers
MLP backbone for binary malware classification.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class MalwareMLP(nn.Module):
    """
    Three-hidden-layer Multilayer Perceptron for binary malware classification.

    Architecture (per layer)
    ------------------------
    Linear → BatchNorm1d → ReLU → Dropout
    ...repeated for each hidden layer...
    Linear → (logit output, shape: [B, 1])

    The final layer emits a raw logit; pair with
    :class:`torch.nn.BCEWithLogitsLoss` during training and apply
    ``torch.sigmoid`` at inference time.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector ``D``.
    hidden_dims : List[int]
        Width of each of the three hidden layers.
        Defaults to ``[512, 256, 128]``.
    dropout_rate : float
        Dropout probability applied after each hidden block.
        Must be in ``[0, 1)``. Defaults to ``0.3``.
    output_dim : int
        Number of output logits. ``1`` for binary classification (default).

    Attributes
    ----------
    hidden_layers : nn.Sequential
        Stacked hidden blocks (Linear + BN + ReLU + Dropout).
    classifier : nn.Linear
        Final projection to ``output_dim`` logits.

    Examples
    --------
    >>> model = MalwareMLP(input_dim=100)
    >>> x = torch.randn(32, 100)
    >>> logits = model(x)          # shape: (32, 1)
    >>> probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        if len(hidden_dims) != 3:
            raise ValueError(
                f"hidden_dims must contain exactly 3 values, got {len(hidden_dims)}."
            )
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}."
            )
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")

        # Build hidden blocks
        blocks: List[nn.Module] = []
        in_features = input_dim

        for out_features in hidden_dims:
            blocks.append(self._hidden_block(in_features, out_features, dropout_rate))
            in_features = out_features

        self.hidden_layers: nn.Sequential = nn.Sequential(*blocks)
        self.classifier: nn.Linear = nn.Linear(in_features, output_dim)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hidden_block(
        in_features: int,
        out_features: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        """
        Construct a single hidden block: Linear → BatchNorm1d → ReLU → Dropout.

        Parameters
        ----------
        in_features : int
            Input dimensionality.
        out_features : int
            Output dimensionality.
        dropout_rate : float
            Dropout probability.

        Returns
        -------
        nn.Sequential
            The assembled hidden block.
        """
        return nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),  # BN has learnable bias
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
        )

    def _init_weights(self) -> None:
        """
        Initialise weights using Kaiming uniform (He) initialisation for
        Linear layers and constant initialisation for BatchNorm parameters.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        x : Tensor
            Input feature tensor of shape ``(B, input_dim)`` where ``B`` is
            the batch size.

        Returns
        -------
        Tensor
            Raw logits of shape ``(B, output_dim)``.
            Apply ``torch.sigmoid`` to obtain class probabilities.

        Raises
        ------
        ValueError
            If ``x`` does not have exactly two dimensions.
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2-D input tensor (B, D), got shape {tuple(x.shape)}."
            )

        x = self.hidden_layers(x)
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Return class-1 probabilities without computing gradients.

        Parameters
        ----------
        x : Tensor
            Input feature tensor of shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Probabilities of shape ``(B,)`` in ``[0, 1]``.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(dim=-1)

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Return binary class predictions.

        Parameters
        ----------
        x : Tensor
            Input feature tensor of shape ``(B, input_dim)``.
        threshold : float
            Decision boundary. Defaults to ``0.5``.

        Returns
        -------
        Tensor
            Long tensor of shape ``(B,)`` with values in ``{0, 1}``.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).long()

    def count_parameters(self) -> int:
        """
        Return the total number of trainable parameters.

        Returns
        -------
        int
            Sum of elements across all parameter tensors that require grad.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # noqa: D105
        lines = [
            f"{self.__class__.__name__}(",
            f"  hidden_layers={self.hidden_layers},",
            f"  classifier={self.classifier},",
            f"  trainable_params={self.count_parameters():,}",
            ")",
        ]
        return "\n".join(lines)