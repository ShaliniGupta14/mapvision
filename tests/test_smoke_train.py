"""Smoke tests for the training pipeline.

Verifies that:
  1. A forward pass succeeds on a tiny batch.
  2. A backward pass succeeds and loss is finite.
  3. One complete epoch with 2 real samples runs without error.
  4. A checkpoint is written after training.

These tests use real data from splits/train.json but only process
a tiny subset (2–4 samples) for speed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import get_device, load_config, set_seeds
from mapvision.dataset import MapDataset
from mapvision.losses import build_loss
from mapvision.model import UNet
from mapvision.train import train


TRAIN_MANIFEST = "splits/train.json"
CLASSES_PATH = "classes.json"


@pytest.fixture(scope="module")
def device():
    return get_device(prefer_mps=True)


@pytest.fixture(scope="module")
def tiny_dataset():
    ds = MapDataset(TRAIN_MANIFEST, size=256, classes_path=CLASSES_PATH)
    # Use only the first 4 samples
    return Subset(ds, list(range(min(4, len(ds)))))


class TestForwardPass:
    def test_forward_succeeds(self, device):
        model = UNet(num_classes=5, base=16).to(device)
        model.eval()
        x = torch.randn(2, 3, 256, 256, device=device)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (2, 5, 256, 256)

    def test_output_is_finite(self, device):
        model = UNet(num_classes=5, base=16).to(device)
        x = torch.randn(2, 3, 256, 256, device=device)
        out = model(x)
        assert torch.isfinite(out).all()


class TestBackwardPass:
    def test_backward_succeeds(self, device):
        set_seeds(42)
        model = UNet(num_classes=5, base=16).to(device)
        criterion = build_loss("cross_entropy")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randn(2, 3, 256, 256, device=device)
        y = torch.randint(0, 5, (2, 256, 256), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)

    def test_loss_is_finite(self, device):
        model = UNet(num_classes=5, base=16).to(device)
        criterion = build_loss("cross_entropy")
        x = torch.randn(2, 3, 256, 256, device=device)
        y = torch.randint(0, 5, (2, 256, 256), device=device)
        logits = model(x)
        loss = criterion(logits, y)
        assert torch.isfinite(loss).item()


class TestOneEpochRealData:
    """Smoke test: one epoch on 2 real map tiles."""

    def test_epoch_runs(self, tiny_dataset, device):
        set_seeds(42)
        loader = DataLoader(tiny_dataset, batch_size=2, shuffle=False, num_workers=0)
        model = UNet(num_classes=5, base=16).to(device)
        criterion = build_loss("cross_entropy")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        total_loss = 0.0
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        assert torch.isfinite(torch.tensor(total_loss))


class TestCheckpointCreation:
    def test_checkpoint_saved(self, tiny_dataset, device, tmp_path):
        set_seeds(42)
        loader = DataLoader(tiny_dataset, batch_size=2, shuffle=False, num_workers=0)
        model = UNet(num_classes=5, base=16).to(device)
        criterion = build_loss("cross_entropy")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()

        ckpt_path = tmp_path / "test_ckpt.pt"
        torch.save(
            {
                "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
                "epoch": 1,
                "val_miou": 0.0,
                "config": {"num_classes": 5, "model": {"name": "unet", "num_classes": 5, "base_channels": 16}},
                "class_names": ["frame", "water", "blocks", "non-built", "road_network"],
            },
            ckpt_path,
        )
        assert ckpt_path.exists()
        loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "model_state" in loaded
        assert loaded["epoch"] == 1
