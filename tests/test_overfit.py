"""Tiny-subset overfit check (slow; run with ``pytest -m slow``).

A correct data → model → loss → optimizer pipeline must be able to
memorise four real tiles. If this fails, something upstream is broken
(label mapping, augmentation misalignment, frozen weights, ...).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import get_device, set_seeds
from mapvision.dataset import MapDataset
from mapvision.metrics import SegmentationMetrics
from mapvision.model import UNet

ITERATIONS = 200


@pytest.mark.slow
def test_overfits_four_samples():
    set_seeds(42)
    device = get_device(prefer_mps=True)
    dataset = Subset(MapDataset("splits/train.json", classes_path="classes.json"), range(4))
    images, masks = next(iter(DataLoader(dataset, batch_size=4)))
    images, masks = images.to(device), masks.to(device)

    model = UNet(num_classes=5, base=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for step in range(ITERATIONS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    model.eval()
    metrics = SegmentationMetrics(num_classes=5)
    with torch.inference_mode():
        metrics.update(model(images).argmax(dim=1), masks)
    results = metrics.compute()

    assert loss.item() < 0.2 * initial_loss, f"loss {initial_loss:.3f} → {loss.item():.3f}"
    assert results["pixel_accuracy"] > 0.9, results["pixel_accuracy"]
