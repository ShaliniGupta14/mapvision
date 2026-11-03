"""Evaluation logic for MapVision.

Runs a model over a data loader and returns aggregated segmentation metrics.
This module is imported by scripts/evaluate.py and by the training loop.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from mapvision.dataset import CLASS_NAMES
from mapvision.metrics import SegmentationMetrics


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 5,
) -> Dict:
    """Evaluate a segmentation model on a dataset.

    Args:
        model:       Model in eval mode.
        loader:      DataLoader yielding (images, masks) batches.
        device:      Compute device.
        num_classes: Number of semantic classes.

    Returns:
        Metrics dict from :class:`SegmentationMetrics`.
    """
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=CLASS_NAMES)
    model.eval()

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            metrics.update(preds, masks)

    return metrics.compute()
