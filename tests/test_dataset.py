"""Tests for MapDataset — verifies shapes, dtypes, value ranges, and class IDs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

# Allow running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.dataset import CLASS_NAMES, MapDataset

TRAIN_MANIFEST = "splits/train.json"
CLASSES_PATH = "classes.json"


@pytest.fixture(scope="module")
def dataset():
    """Create a dataset instance once per test module."""
    return MapDataset(TRAIN_MANIFEST, size=256, classes_path=CLASSES_PATH)


@pytest.fixture(scope="module")
def first_item(dataset):
    return dataset[0]


class TestDatasetBasics:
    def test_dataset_nonempty(self, dataset):
        assert len(dataset) > 0, "Dataset must be non-empty"

    def test_class_names_order(self):
        assert CLASS_NAMES == ["frame", "water", "blocks", "non-built", "road_network"]


class TestItemShapes:
    def test_image_shape(self, first_item):
        image, _ = first_item
        assert image.shape == (3, 256, 256), f"Expected (3,256,256), got {image.shape}"

    def test_mask_shape(self, first_item):
        _, mask = first_item
        assert mask.shape == (256, 256), f"Expected (256,256), got {mask.shape}"


class TestItemDtypes:
    def test_image_dtype(self, first_item):
        image, _ = first_item
        assert image.dtype == torch.float32

    def test_mask_dtype(self, first_item):
        _, mask = first_item
        assert mask.dtype == torch.int64


class TestValueRanges:
    def test_image_in_0_1(self, first_item):
        image, _ = first_item
        assert image.min() >= 0.0, "Image min < 0"
        assert image.max() <= 1.0, "Image max > 1"

    def test_mask_valid_class_ids(self, first_item):
        _, mask = first_item
        unique = mask.unique().tolist()
        assert all(v in range(5) for v in unique), f"Invalid class IDs: {unique}"


class TestDataLoader:
    """Replicate the validated batch check from the project brief."""

    def test_dataloader_batch(self, dataset):
        loader = DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )
        images, masks = next(iter(loader))
        assert images.shape == (4, 3, 256, 256)
        assert masks.shape == (4, 256, 256)
        assert masks.dtype == torch.int64
        assert images.min() >= 0 and images.max() <= 1


class TestMissingManifest:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            MapDataset("nonexistent_manifest.json")


class TestEmptyManifest:
    def test_raises_value_error(self, tmp_path):
        m = tmp_path / "empty.json"
        m.write_text("[]")
        with pytest.raises(ValueError, match="Empty manifest"):
            MapDataset(str(m), classes_path=CLASSES_PATH)
