"""Tests for mask colorization and PNG outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.visualization import (
    CLASS_COLORS,
    mask_to_rgb,
    plot_training_history,
    save_mask_only,
    save_overlay,
    save_prediction_grid,
)


def test_mask_to_rgb_uses_fixed_palette():
    mask = np.arange(5).reshape(1, 5)
    rgb = mask_to_rgb(mask)
    assert (rgb[0] == CLASS_COLORS).all()


def test_png_outputs_written(tmp_path):
    image = torch.rand(3, 32, 32)
    mask = torch.randint(0, 5, (32, 32))
    save_mask_only(mask, tmp_path / "mask.png")
    save_overlay(image, mask, tmp_path / "overlay.png")
    save_prediction_grid(image, mask, mask, tmp_path / "grid.png")
    for name in ("mask.png", "overlay.png", "grid.png"):
        assert Image.open(tmp_path / name).size[0] > 0
    assert Image.open(tmp_path / "mask.png").size == (32, 32)


def test_plot_training_history(tmp_path):
    history = [
        {"epoch": e, "train_loss": 1 / e, "val_loss": 1.1 / e, "val_miou": 0.1 * e, "val_dice": 0.12 * e}
        for e in range(1, 4)
    ]
    plot_training_history(history, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").exists()
