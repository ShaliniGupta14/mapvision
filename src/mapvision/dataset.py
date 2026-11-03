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

        classes_file = Path(classes_path)
        if not classes_file.exists():
            raise FileNotFoundError(f"Class color file not found: {classes_path}")
        classes = json.loads(classes_file.read_text())

        if not all(name in classes for name in CLASS_NAMES):
            missing = [n for n in CLASS_NAMES if n not in classes]
            raise ValueError(f"classes.json missing keys: {missing}")

        self.colors = np.array(
            [classes[name] for name in CLASS_NAMES], dtype=np.uint8
        )

        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")

        if validate:
            self._validate_all()

    def _validate_all(self) -> None:
        """Iterate all records and report any errors."""
        errors = []
        for i, record in enumerate(self.records):
            for key in ("image", "label"):
                p = Path(record.get(key, ""))
                if not p.exists():
                    errors.append(f"Record {i}: {key} not found: {p}")
        if errors:
            raise FileNotFoundError(
                f"{len(errors)} missing files:\n" + "\n".join(errors[:10])
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return one (image, mask) pair.

        Returns:
            image: Float32 tensor of shape (3, H, W) in [0, 1].
            mask:  Int64 tensor of shape (H, W) with class IDs 0–4.

        Raises:
            ValueError: If a pixel color in the label doesn't match any class.
            OSError:    If an image file is corrupted or unreadable.
        """
        record = self.records[index]
        target_size = (self.size, self.size)

        try:
            with Image.open(record["image"]) as src:
                image = src.convert("RGB").resize(
                    target_size, Image.Resampling.BILINEAR
                )
                image = np.array(image, dtype=np.float32) / 255.0
        except (UnidentifiedImageError, OSError) as exc:
            raise OSError(f"Cannot read image: {record['image']}") from exc

        try:
            with Image.open(record["label"]) as src:
                label = src.convert("RGB").resize(
                    target_size, Image.Resampling.NEAREST
                )
                label = np.array(label)
        except (UnidentifiedImageError, OSError) as exc:
            raise OSError(f"Cannot read label: {record['label']}") from exc

        # Map RGB label → class ID using nearest-neighbor preserved colors.
        mask = np.full(label.shape[:2], -1, dtype=np.int64)
        for class_id, color in enumerate(self.colors):
            mask[np.all(label == color, axis=-1)] = class_id

        unknown = (mask < 0).sum()
        if unknown > 0:
            raise ValueError(
                f"Unknown label colors in {record['label']}: "
                f"{unknown} pixels did not match any class color."
            )

        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask_t = torch.from_numpy(mask)

        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)

        return image_t, mask_t
