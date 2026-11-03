"""Dataset loading for MapVision.

MapDataset reads image-mask pairs from JSON manifests produced by the
split scripts, converts them to tensors, and optionally applies
segmentation-safe transforms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

CLASS_NAMES = ["frame", "water", "blocks", "non-built", "road_network"]
NUM_CLASSES = len(CLASS_NAMES)
VALID_IDS = set(range(NUM_CLASSES))

# Source image modes that convert losslessly to 3-channel RGB.
ALLOWED_IMAGE_MODES = ("RGB", "RGBA", "L")
ALLOWED_LABEL_MODES = ("RGB", "RGBA")


class DatasetError(ValueError):
    """Raised when an image/label pair fails validation."""


def load_class_colors(classes_path: str | Path) -> np.ndarray:
    """Load the class → RGB color table, ordered by ``CLASS_NAMES``.

    Returns:
        uint8 array of shape (NUM_CLASSES, 3).
    """
    classes_file = Path(classes_path)
    if not classes_file.exists():
        raise FileNotFoundError(f"Class color file not found: {classes_path}")
    classes = json.loads(classes_file.read_text())
    missing = [n for n in CLASS_NAMES if n not in classes]
    if missing:
        raise ValueError(f"{classes_path} missing keys: {missing}")
    return np.array([classes[name] for name in CLASS_NAMES], dtype=np.uint8)


def rgb_label_to_mask(label: np.ndarray, colors: np.ndarray, source: str) -> np.ndarray:
    """Map an RGB label array (H, W, 3) to class IDs (H, W) int64.

    Raises:
        DatasetError: If any pixel color does not match a class color.
    """
    mask = np.full(label.shape[:2], -1, dtype=np.int64)
    for class_id, color in enumerate(colors):
        mask[np.all(label == color, axis=-1)] = class_id
    unknown = int((mask < 0).sum())
    if unknown > 0:
        raise DatasetError(
            f"Unknown label colors in {source}: "
            f"{unknown} pixels did not match any class color."
        )
    return mask


def validate_record(record: dict, colors: np.ndarray) -> None:
    """Check one manifest record at native resolution.

    Verifies both files exist and are readable, filenames match, image
    modes are supported, image and label dimensions agree and are
    non-empty, and every label pixel is a known class color.

    Raises:
        DatasetError: Describing the first problem found.
    """
    for key in ("image", "label"):
        if key not in record:
            raise DatasetError(f"Record missing '{key}' field: {record}")
    image_path, label_path = Path(record["image"]), Path(record["label"])

    for kind, p in (("image", image_path), ("label", label_path)):
        if not p.exists():
            raise DatasetError(f"Missing {kind} file: {p}")
    if image_path.stem != label_path.stem:
        raise DatasetError(
            f"Filename mismatch: image '{image_path.name}' vs label '{label_path.name}'"
        )

    try:
        with Image.open(image_path) as img:
            image_mode, image_size = img.mode, img.size
        with Image.open(label_path) as lbl:
            label_mode, label_size = lbl.mode, lbl.size
            label = np.array(lbl.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise DatasetError(f"Unreadable file in pair {image_path}: {exc}") from exc

    if image_mode not in ALLOWED_IMAGE_MODES:
        raise DatasetError(
            f"Unexpected image mode '{image_mode}' in {image_path}; "
            f"expected one of {ALLOWED_IMAGE_MODES}"
        )
    if label_mode not in ALLOWED_LABEL_MODES:
        raise DatasetError(
            f"Unexpected label mode '{label_mode}' in {label_path}; "
            f"expected one of {ALLOWED_LABEL_MODES}"
        )
    if 0 in image_size or 0 in label_size:
        raise DatasetError(f"Empty image or label: {image_path}")
    if image_size != label_size:
        raise DatasetError(
            f"Dimension mismatch: image {image_size} vs label {label_size} for {image_path.name}"
        )
    rgb_label_to_mask(label, colors, str(label_path))


class MapDataset(Dataset):
    """Historical map segmentation dataset.

    Each record in the manifest JSON has the form::

        {"image": "path/to/image.png", "label": "path/to/label.png"}

    Images are RGB PNGs; labels are RGB PNGs where each pixel's color
    encodes its semantic class according to ``classes.json``.

    Args:
        manifest:     Path to the JSON manifest file.
        size:         Target spatial resolution (square). Default 256.
        classes_path: Path to the class color mapping JSON.
        transform:    Optional callable ``(image, mask) -> (image, mask)``
                      applied after tensor conversion. Intended for
                      segmentation-safe augmentations.
        validate:     If True, check each record for readability and valid
                      mask labels during ``__init__``. Slow on large datasets.
    """

    def __init__(
        self,
        manifest: str,
        size: int = 256,
        classes_path: str = "classes.json",
        transform: Optional[Callable] = None,
        validate: bool = False,
    ):
        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        self.records: List[dict] = json.loads(manifest_path.read_text())
        self.size = size
        self.transform = transform
        self.colors = load_class_colors(classes_path)

        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")

        if validate:
            self._validate_all()

    def validation_errors(self) -> List[str]:
        """Run :func:`validate_record` on every record; return error messages."""
        errors = []
        for i, record in enumerate(self.records):
            try:
                validate_record(record, self.colors)
            except DatasetError as exc:
                errors.append(f"Record {i}: {exc}")
        return errors

    def _validate_all(self) -> None:
        """Raise a single DatasetError summarising every invalid record."""
        errors = self.validation_errors()
        if errors:
            raise DatasetError(
                f"{len(errors)} invalid records:\n" + "\n".join(errors[:10])
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return one (image, mask) pair.

        Returns:
            image: Float32 tensor of shape (3, H, W) in [0, 1].
            mask:  Int64 tensor of shape (H, W) with class IDs 0–4.

        Raises:
            DatasetError: If modes/dimensions are invalid or a label pixel
                          color doesn't match any class.
            OSError:      If an image file is corrupted or unreadable.
        """
        record = self.records[index]
        target_size = (self.size, self.size)

        try:
            with Image.open(record["image"]) as src:
                if src.mode not in ALLOWED_IMAGE_MODES:
                    raise DatasetError(
                        f"Unexpected image mode '{src.mode}' in {record['image']}"
                    )
                source_size = src.size
                image = src.convert("RGB").resize(
                    target_size, Image.Resampling.BILINEAR
                )
                image = np.array(image, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError) as exc:
            raise OSError(f"Cannot read image: {record['image']}") from exc

        try:
            with Image.open(record["label"]) as src:
                if src.size != source_size:
                    raise DatasetError(
                        f"Dimension mismatch: image {source_size} vs label "
                        f"{src.size} for {record['label']}"
                    )
                # NEAREST keeps label colors exact; never interpolate masks.
                label = src.convert("RGB").resize(
                    target_size, Image.Resampling.NEAREST
                )
                label = np.array(label)
        except (UnidentifiedImageError, OSError) as exc:
            raise OSError(f"Cannot read label: {record['label']}") from exc

        mask = rgb_label_to_mask(label, self.colors, record["label"])

        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask_t = torch.from_numpy(mask)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)

        return image_t, mask_t
