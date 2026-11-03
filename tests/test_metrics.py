"""Tests for segmentation metrics — uses synthetic masks with known ground-truth values."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.metrics import SegmentationMetrics, confusion_matrix_from_logits


def perfect_metrics(num_classes: int = 5) -> dict:
    """Create a metrics instance with a perfect diagonal confusion matrix."""
    m = SegmentationMetrics(num_classes=num_classes)
    # Perfect predictions: diagonal only
    for cls in range(num_classes):
        preds = torch.full((10, 10), cls, dtype=torch.int64)
        tgts = torch.full((10, 10), cls, dtype=torch.int64)
        m.update(preds, tgts)
    return m.compute()


def all_wrong_metrics() -> dict:
    """Predictions always off by 1 (wrapping), every class wrong."""
    m = SegmentationMetrics(num_classes=5)
    for cls in range(5):
        preds = torch.full((10, 10), (cls + 1) % 5, dtype=torch.int64)
        tgts = torch.full((10, 10), cls, dtype=torch.int64)
        m.update(preds, tgts)
    return m.compute()


class TestPerfectPredictions:
    def setup_method(self):
        self.results = perfect_metrics()

    def test_pixel_accuracy_is_1(self):
        assert abs(self.results["pixel_accuracy"] - 1.0) < 1e-6

    def test_mean_iou_is_1(self):
        assert abs(self.results["mean_iou"] - 1.0) < 1e-6

    def test_mean_dice_is_1(self):
        assert abs(self.results["mean_dice"] - 1.0) < 1e-6

    def test_per_class_iou_all_1(self):
        for name, val in self.results["per_class_iou"].items():
            assert val is not None
            assert abs(val - 1.0) < 1e-6, f"class {name}: {val}"


class TestAllWrongPredictions:
    def setup_method(self):
        self.results = all_wrong_metrics()

    def test_pixel_accuracy_is_0(self):
        assert self.results["pixel_accuracy"] < 1e-6

    def test_mean_iou_is_0(self):
        assert self.results["mean_iou"] < 1e-6

    def test_mean_dice_is_0(self):
        assert self.results["mean_dice"] < 1e-6


class TestKnownIoU:
    """Exact IoU for a 50/50 split between perfect and wrong predictions."""

    def test_partial_overlap_iou(self):
        """Half of class-0 pixels predicted correctly, rest as class-1.
        Expected IoU for class 0:
            intersection = 50, union = 50 + 100 - 50 = 100
            IoU = 0.5
        """
        from mapvision.dataset import CLASS_NAMES
        m = SegmentationMetrics(num_classes=5, class_names=CLASS_NAMES)
        preds = torch.cat([
            torch.zeros(50, dtype=torch.int64),   # correct
            torch.ones(50, dtype=torch.int64),    # wrong
        ])
        tgts = torch.zeros(100, dtype=torch.int64)
        m.update(preds, tgts)
        results = m.compute()
        iou_0 = results["per_class_iou"]["frame"]
        assert abs(iou_0 - 0.5) < 1e-6, f"Expected IoU=0.5, got {iou_0}"


class TestReset:
    def test_reset_clears_state(self):
        m = SegmentationMetrics(num_classes=5)
        preds = torch.zeros(10, dtype=torch.int64)
        tgts = torch.zeros(10, dtype=torch.int64)
        m.update(preds, tgts)
        m.reset()
        results = m.compute()
        assert results["pixel_accuracy"] == 0.0


class TestConfusionMatrixFromLogits:
    def test_shape(self):
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))
        cm = confusion_matrix_from_logits(logits, targets, num_classes=5)
        assert cm.shape == (5, 5)

    def test_sum_equals_total_pixels(self):
        logits = torch.randn(2, 5, 32, 32)
        targets = torch.randint(0, 5, (2, 32, 32))
        cm = confusion_matrix_from_logits(logits, targets, num_classes=5)
        total = 2 * 32 * 32
        assert cm.sum().item() == total
