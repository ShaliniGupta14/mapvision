"""Tests for single-image inference and checkpoint loading errors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.inference import load_checkpoint, measure_latency, predict_image
from mapvision.model import UNet

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def model():
    return UNet(num_classes=5, base=8).eval()


def test_predict_image_any_size(tmp_path, model):
    path = tmp_path / "map.png"
    Image.new("RGB", (400, 300), (200, 180, 150)).save(path)
    image, pred = predict_image(path, model, CPU, image_size=256)
    assert image.shape == (3, 256, 256)
    assert pred.shape == (256, 256) and pred.dtype == torch.int64
    assert pred.min() >= 0 and pred.max() < 5


def test_predict_corrupt_image_raises(tmp_path, model):
    path = tmp_path / "bad.png"
    path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Cannot read image"):
        predict_image(path, model, CPU)


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing.pt", CPU)


def test_corrupt_checkpoint_raises(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"garbage")
    with pytest.raises(ValueError, match="Cannot read checkpoint"):
        load_checkpoint(path, CPU)


def test_measure_latency_positive(model):
    assert measure_latency(model, CPU, image_size=64, runs=2, warmup=1) > 0
