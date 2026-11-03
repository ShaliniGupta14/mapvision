"""Segmentation-safe data augmentation transforms for MapVision.

All transforms that modify spatial layout are applied identically to
both image and mask tensors to preserve pixel-level alignment.
"""

from __future__ import annotations

import random
from typing import Tuple

import torch
import torch.nn.functional as F


def random_hflip(
    image: torch.Tensor, mask: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random horizontal flip applied jointly to image and mask."""
    if random.random() < p:
        image = image.flip(-1)
        mask = mask.flip(-1)
    return image, mask


def random_vflip(
    image: torch.Tensor, mask: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random vertical flip applied jointly to image and mask."""
    if random.random() < p:
        image = image.flip(-2)
        mask = mask.flip(-2)
    return image, mask


def random_rot90(
    image: torch.Tensor, mask: torch.Tensor, p: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random 90-degree rotation (0, 90, 180, 270) applied jointly."""
    if random.random() < p:
        k = random.randint(1, 3)
        image = torch.rot90(image, k, dims=(-2, -1))
        mask = torch.rot90(mask, k, dims=(-2, -1))
    return image, mask


def random_brightness_contrast(
    image: torch.Tensor,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
    p: float = 0.5,
) -> torch.Tensor:
    """Random brightness and contrast adjustment (image only).

    Does not affect the mask since it is a photometric transform.
    """
    if random.random() < p:
        brightness_factor = random.uniform(*brightness_range)
        contrast_factor = random.uniform(*contrast_range)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        image = (image - mean) * contrast_factor + mean * brightness_factor
        image = image.clamp(0.0, 1.0)
    return image


class TrainTransform:
    """Segmentation-safe augmentation pipeline for training.

    Applies spatial transforms to both image and mask identically,
    and photometric transforms to image only.
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_90: bool = True,
        brightness_contrast: bool = True,
    ):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_90 = rotation_90
        self.brightness_contrast = brightness_contrast

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to an image-mask pair.

        Args:
            image: Float tensor of shape (3, H, W) in [0, 1].
            mask:  Int64 tensor of shape (H, W) with class IDs 0–4.

        Returns:
            Augmented (image, mask) pair with same dtypes and shapes.
        """
        if self.horizontal_flip:
            image, mask = random_hflip(image, mask)
        if self.vertical_flip:
            image, mask = random_vflip(image, mask)
        if self.rotation_90:
            image, mask = random_rot90(image, mask)
        if self.brightness_contrast:
            image = random_brightness_contrast(image)
        return image, mask


class ValTransform:
    """Identity transform for validation and test sets.

    No augmentation is applied; images and masks pass through unchanged.
    """

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return image, mask
