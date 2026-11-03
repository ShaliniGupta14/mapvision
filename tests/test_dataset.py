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


# ── Validation of malformed pairs ─────────────────────────────────────────

import json

import numpy as np
from PIL import Image

from mapvision.dataset import DatasetError, load_class_colors, validate_record


def _write_pair(tmp_path, name="tile", image_size=(8, 8), label_size=None,
                label_color=(0, 0, 255), image_mode="RGB", label_name=None):
    """Write a synthetic image/label pair and return its manifest record."""
    (tmp_path / "images").mkdir(exist_ok=True)
    (tmp_path / "labels").mkdir(exist_ok=True)
    image_path = tmp_path / "images" / f"{name}.png"
    label_path = tmp_path / "labels" / f"{label_name or name}.png"
    Image.new(image_mode, image_size).save(image_path)
    Image.new("RGB", label_size or image_size, label_color).save(label_path)
    return {"image": str(image_path), "label": str(label_path)}


@pytest.fixture(scope="module")
def colors():
    return load_class_colors(CLASSES_PATH)


class TestValidateRecord:
    def test_valid_pair_passes(self, tmp_path, colors):
        validate_record(_write_pair(tmp_path), colors)

    def test_missing_label(self, tmp_path, colors):
        rec = _write_pair(tmp_path)
        Path(rec["label"]).unlink()
        with pytest.raises(DatasetError, match="Missing label"):
            validate_record(rec, colors)

    def test_filename_mismatch(self, tmp_path, colors):
        rec = _write_pair(tmp_path, label_name="other")
        with pytest.raises(DatasetError, match="Filename mismatch"):
            validate_record(rec, colors)

    def test_unreadable_image(self, tmp_path, colors):
        rec = _write_pair(tmp_path)
        Path(rec["image"]).write_bytes(b"not a png")
        with pytest.raises(DatasetError, match="Unreadable"):
            validate_record(rec, colors)

    def test_dimension_mismatch(self, tmp_path, colors):
        rec = _write_pair(tmp_path, label_size=(4, 4))
        with pytest.raises(DatasetError, match="Dimension mismatch"):
            validate_record(rec, colors)

    def test_invalid_label_color(self, tmp_path, colors):
        rec = _write_pair(tmp_path, label_color=(1, 2, 3))
        with pytest.raises(DatasetError, match="Unknown label colors"):
            validate_record(rec, colors)

    def test_unexpected_channels(self, tmp_path, colors):
        rec = _write_pair(tmp_path, image_mode="LA")
        with pytest.raises(DatasetError, match="Unexpected image mode"):
            validate_record(rec, colors)

    def test_dataset_validate_flag_raises(self, tmp_path):
        rec = _write_pair(tmp_path, label_color=(1, 2, 3))
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([rec]))
        with pytest.raises(DatasetError, match="1 invalid records"):
            MapDataset(str(manifest), classes_path=CLASSES_PATH, validate=True)

    def test_getitem_resizes_and_maps_labels(self, tmp_path):
        rec = _write_pair(tmp_path, image_size=(300, 300), label_color=(0, 0, 255))
        manifest = tmp_path / "m.json"
        manifest.write_text(json.dumps([rec]))
        image, mask = MapDataset(str(manifest), classes_path=CLASSES_PATH)[0]
        assert image.shape == (3, 256, 256)
        assert mask.unique().tolist() == [CLASS_NAMES.index("water")]


class TestSplits:
    """The committed splits must be disjoint by file and by source map."""

    def test_no_overlap_between_splits(self):
        splits = {s: json.loads(Path(f"splits/{s}.json").read_text())
                  for s in ("train", "val", "test")}
        for key in ("image", "source_group"):
            seen = {s: {r[key] for r in recs} for s, recs in splits.items()}
            assert not seen["train"] & seen["val"], f"train/val share {key}"
            assert not seen["train"] & seen["test"], f"train/test share {key}"
            assert not seen["val"] & seen["test"], f"val/test share {key}"
