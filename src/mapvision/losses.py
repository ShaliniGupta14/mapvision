"""Loss functions for semantic segmentation.

Provides CrossEntropyLoss and a combined CE+Dice loss, both configurable
via the YAML config's loss.type field.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class semantic segmentation.

    Computes a mean Dice loss across all classes present in the batch.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            logits:  Raw model output, shape (N, C, H, W).
            targets: Ground-truth class IDs, shape (N, H, W), dtype int64.

        Returns:
            Scalar Dice loss.
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets → (N, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + targets_one_hot).sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        return 1.0 - dice_per_class.mean()


class CEDiceLoss(nn.Module):
    """Combined Cross-Entropy and Dice loss.

    Loss = 0.5 * CE + 0.5 * Dice

    This combination provides both boundary-aware (CE) and region-aware
    (Dice) gradient signal.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        alpha: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(smooth=smooth)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, targets) + (1 - self.alpha) * self.dice(
            logits, targets
        )


def class_weight_tensor(
    weights: Sequence[float] | None, num_classes: int, device: torch.device
) -> torch.Tensor | None:
    """Convert configured class weights to a tensor, validating length.

    Raises:
        ValueError: If the number of weights differs from num_classes or
                    any weight is negative.
    """
    if weights is None:
        return None
    if len(weights) != num_classes:
        raise ValueError(
            f"loss.class_weights has {len(weights)} entries; expected {num_classes}"
        )
    if any(w < 0 for w in weights):
        raise ValueError(f"loss.class_weights must be non-negative, got {list(weights)}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_loss(loss_type: str, weight: torch.Tensor | None = None) -> nn.Module:
    """Factory function to build a loss from config string.

    Args:
        loss_type: "cross_entropy" or "ce_dice".
        weight:    Optional class-weight tensor for CE (on the training device).

    Returns:
        An nn.Module implementing the chosen loss.

    Raises:
        ValueError: If loss_type is unrecognized.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_type == "ce_dice":
        return CEDiceLoss(weight=weight)
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type!r}. "
            "Choose 'cross_entropy' or 'ce_dice'."
        )
