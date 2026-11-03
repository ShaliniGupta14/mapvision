"""Evaluation logic for MapVision.

Runs a model over a data loader and returns aggregated segmentation metrics.
This module is imported by scripts/evaluate.py and by the training loop.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from mapvision.dataset import CLASS_NAMES
from mapvision.metrics import SegmentationMetrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
    criterion: Optional[torch.nn.Module] = None,
) -> Dict:
    """Evaluate a segmentation model on a dataset in a single pass.

    Args:
        model:       Model to evaluate (switched to eval mode).
        loader:      DataLoader yielding (images, masks) batches.
        device:      Compute device.
        num_classes: Number of semantic classes.
        criterion:   Optional loss; if given, the sample-weighted mean loss
                     is added to the result under ``"loss"``.

    Returns:
        Metrics dict from :class:`SegmentationMetrics` (plus ``"loss"``).
    """
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=CLASS_NAMES)
    model.eval()
    total_loss = 0.0

    with torch.inference_mode():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            if criterion is not None:
                total_loss += criterion(logits, masks).item() * images.size(0)
            metrics.update(logits.argmax(dim=1), masks)

    results = metrics.compute()
    if criterion is not None:
        results["loss"] = total_loss / len(loader.dataset)
    return results
