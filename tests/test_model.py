"""Tests for UNet model — verifies forward pass shape and device compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.model import UNet, build_model, model_summary


class TestUNetForward:
    def test_output_shape_default(self):
        model = UNet(num_classes=5, base=16)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (2, 5, 256, 256), f"Expected (2,5,256,256), got {out.shape}"

    def test_output_shape_batch1(self):
        model = UNet(num_classes=5, base=16)
        model.eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 5, 256, 256)

    def test_output_is_float(self):
        model = UNet(num_classes=5, base=16)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.dtype == torch.float32

    def test_different_num_classes(self):
        model = UNet(num_classes=3, base=8)
        model.eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (2, 3, 256, 256)


class TestBuildModel:
    def test_build_unet(self):
        model = build_model("unet", num_classes=5, base_channels=16)
        assert isinstance(model, UNet)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("resnet99")


class TestModelSummary:
    def test_summary_runs(self, capsys):
        model = UNet(num_classes=5, base=16)
        model_summary(model)
        captured = capsys.readouterr()
        assert "Total parameters" in captured.out
        assert "Trainable params" in captured.out


class TestParameterCount:
    def test_model_has_parameters(self):
        model = UNet(num_classes=5, base=16)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_all_params_trainable(self):
        model = UNet(num_classes=5, base=16)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable


class TestMPSFallback:
    """Smoke test — model runs on whatever device is available."""

    def test_model_on_available_device(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = UNet(num_classes=5, base=16).to(device)
        model.eval()
        x = torch.randn(1, 3, 256, 256, device=device)
        with torch.inference_mode():
            out = model(x)
        assert out.shape == (1, 5, 256, 256)
        assert out.device.type == device.type
