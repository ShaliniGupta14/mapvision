"""Smoke tests for the training pipeline.

Verifies that:
  1. A forward pass succeeds on a tiny batch.
  2. A backward pass succeeds and loss is finite.
  3. One complete epoch with 2 real samples runs without error.
  4. The real train() entry point writes best/latest checkpoints that
     load for inference, and training can resume from latest.pt.

These tests use real data from splits/train.json but only process
a tiny subset (2–4 samples) for speed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import get_device, load_config, set_seeds
from mapvision.dataset import CLASS_NAMES, MapDataset
from mapvision.inference import load_checkpoint
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


def _tiny_config(tmp_path, epochs: int = 1):
    """Config pointing at 4 train / 2 val real samples with outputs in tmp_path."""
    records = json.loads(Path(TRAIN_MANIFEST).read_text())
    val_records = json.loads(Path("splits/val.json").read_text())
    (tmp_path / "train.json").write_text(json.dumps(records[:4]))
    (tmp_path / "val.json").write_text(json.dumps(val_records[:2]))
    cfg = load_config("configs/default.yaml")
    cfg.data.train_manifest = str(tmp_path / "train.json")
    cfg.data.val_manifest = str(tmp_path / "val.json")
    cfg.data.batch_size = 2
    cfg.training.epochs = epochs
    cfg.paths.checkpoints = str(tmp_path / "checkpoints")
    cfg.paths.runs = str(tmp_path / "runs")
    return cfg


class TestTrainEntryPoint:
    """Runs the real train() for one epoch and checks its artefacts."""

    def test_train_writes_checkpoints_and_history(self, tmp_path):
        cfg = _tiny_config(tmp_path)
        out = train(cfg, output_dir=str(tmp_path / "run"))

        history = json.loads((out / "history.json").read_text())
        assert len(history) == 1
        assert np.isfinite(history[0]["train_loss"])
        assert np.isfinite(history[0]["val_loss"])
        assert (out / "training_curves.png").exists()

        for name in ("best.pt", "latest.pt"):
            ckpt = torch.load(tmp_path / "checkpoints" / name, map_location="cpu", weights_only=False)
            for key in ("model_state", "optimizer_state", "scheduler_state", "epoch",
                        "best_val_miou", "config", "class_names"):
                assert key in ckpt, f"{name} missing {key}"
            assert ckpt["class_names"] == CLASS_NAMES

    def test_checkpoint_loads_for_inference(self, tmp_path):
        cfg = _tiny_config(tmp_path)
        train(cfg, output_dir=str(tmp_path / "run"))
        model, _, epoch, _ = load_checkpoint(tmp_path / "checkpoints" / "best.pt", torch.device("cpu"))
        assert epoch == 1
        with torch.inference_mode():
            assert model(torch.rand(1, 3, 256, 256)).shape == (1, 5, 256, 256)

    def test_resume_continues_from_next_epoch(self, tmp_path):
        cfg = _tiny_config(tmp_path, epochs=1)
        run_dir = str(tmp_path / "run")
        train(cfg, output_dir=run_dir)
        cfg.training.epochs = 2
        out = train(cfg, output_dir=run_dir, resume=str(tmp_path / "checkpoints" / "latest.pt"))
        history = json.loads((out / "history.json").read_text())
        assert [row["epoch"] for row in history] == [1, 2]

    def test_resume_missing_checkpoint_raises(self, tmp_path):
        cfg = _tiny_config(tmp_path)
        with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
            train(cfg, output_dir=str(tmp_path / "run"), resume=str(tmp_path / "nope.pt"))
