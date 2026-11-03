"""Semantic segmentation metrics for MapVision.

Implements pixel accuracy, per-class IoU, mean IoU, Dice score, and
confusion matrix accumulation. All functions handle division-by-zero
safely by returning 0 for classes with no ground-truth pixels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


class SegmentationMetrics:
    """Accumulates predictions and computes segmentation metrics.

    Usage::

        metrics = SegmentationMetrics(num_classes=5)
        for preds, targets in loader:
            metrics.update(preds, targets)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        """Reset accumulated confusion matrix."""
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            predictions: Predicted class IDs, shape (N, H, W) or (H, W).
            targets:     Ground-truth class IDs, same shape, dtype int64.
        """
        preds = predictions.reshape(-1).cpu()
        tgts = targets.reshape(-1).cpu()

        # Guard: ignore out-of-range pixels
        valid = (tgts >= 0) & (tgts < self.num_classes)
        preds = preds[valid]
        tgts = tgts[valid]

        encoded = tgts * self.num_classes + preds
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dict with keys: pixel_accuracy, mean_iou, mean_dice,
            per_class_iou, per_class_dice, confusion_matrix.
        """
        cm = self.confusion.double()
        tp = cm.diag()
        row_sum = cm.sum(1)   # ground-truth counts per class
        col_sum = cm.sum(0)   # prediction counts per class

        # Pixel accuracy
        total = cm.sum()
        pixel_accuracy = (tp.sum() / total.clamp_min(1)).item()

        # Per-class IoU
        union = row_sum + col_sum - tp
        present = union > 0
        iou = torch.where(present, tp / union.clamp_min(1), torch.zeros_like(tp))
        mean_iou = iou[present].mean().item() if present.any() else 0.0

        # Per-class Dice
        dice_denom = row_sum + col_sum
        dice_present = dice_denom > 0
        dice = torch.where(
            dice_present,
            2 * tp / dice_denom.clamp_min(1),
            torch.zeros_like(tp),
        )
        mean_dice = dice[dice_present].mean().item() if dice_present.any() else 0.0

        per_class_iou = {
            name: iou[i].item() if present[i].item() else None
            for i, name in enumerate(self.class_names)
        }
        per_class_dice = {
            name: dice[i].item() if dice_present[i].item() else None
            for i, name in enumerate(self.class_names)
        }

        return {
            "pixel_accuracy": pixel_accuracy,
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice,
            "confusion_matrix": self.confusion.tolist(),
        }


def confusion_matrix_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a confusion matrix from raw logits and targets.

    Args:
        logits:  Shape (N, C, H, W) — raw model output.
        targets: Shape (N, H, W) — ground-truth class IDs (int64).
        num_classes: Number of semantic classes.

    Returns:
        Int64 tensor of shape (num_classes, num_classes).
    """
    preds = logits.argmax(dim=1).reshape(-1).cpu()
    tgts = targets.reshape(-1).cpu()
    valid = (tgts >= 0) & (tgts < num_classes)
    preds = preds[valid]
    tgts = tgts[valid]
    encoded = tgts * num_classes + preds
    return torch.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes
    )
