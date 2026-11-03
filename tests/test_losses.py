"""Tests for segmentation losses and class-weight handling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.losses import DiceLoss, build_loss, class_weight_tensor


def _perfect_logits(targets: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Logits that put (almost) all probability on the target class."""
    one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2)
    return one_hot.float() * 50.0


class TestDiceLoss:
    def test_perfect_prediction_near_zero(self):
        targets = torch.randint(0, 5, (2, 16, 16))
        loss = DiceLoss(smooth=1e-6)(_perfect_logits(targets), targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-4)

    def test_wrong_prediction_is_large(self):
        targets = torch.zeros(1, 8, 8, dtype=torch.int64)
        loss = DiceLoss(smooth=1e-6)(_perfect_logits(targets + 1), targets)
        assert loss.item() > 0.9


class TestBuildLoss:
    @pytest.mark.parametrize("loss_type", ["cross_entropy", "ce_dice"])
    def test_losses_are_finite_and_differentiable(self, loss_type):
        logits = torch.randn(2, 5, 16, 16, requires_grad=True)
        targets = torch.randint(0, 5, (2, 16, 16))
        loss = build_loss(loss_type)(logits, targets)
        loss.backward()
        assert torch.isfinite(loss)
        assert logits.grad is not None

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError, match="Unknown loss type"):
            build_loss("focal")

    def test_class_weights_change_ce(self):
        logits = torch.randn(2, 5, 16, 16)
        targets = torch.randint(0, 5, (2, 16, 16))
        weight = torch.tensor([1.0, 10.0, 1.0, 1.0, 1.0])
        plain = build_loss("cross_entropy")(logits, targets)
        weighted = build_loss("cross_entropy", weight=weight)(logits, targets)
        assert not torch.isclose(plain, weighted)


class TestClassWeightTensor:
    def test_none_passthrough(self):
        assert class_weight_tensor(None, 5, torch.device("cpu")) is None

    def test_valid_weights(self):
        w = class_weight_tensor([1, 2, 1, 1, 1], 5, torch.device("cpu"))
        assert w.dtype == torch.float32 and w.shape == (5,)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="expected 5"):
            class_weight_tensor([1.0, 2.0], 5, torch.device("cpu"))

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            class_weight_tensor([1, -1, 1, 1, 1], 5, torch.device("cpu"))
