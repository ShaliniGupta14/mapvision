"""U-Net semantic segmentation model for MapVision.

A lightweight encoder-decoder with skip connections designed to run
on Apple MPS. The architecture follows the original U-Net paper but
uses batch normalization and bilinear upsampling instead of transposed
convolutions, which are more stable on MPS.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two 3×3 conv layers each followed by BatchNorm and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net style encoder-decoder for semantic segmentation.

    Architecture:
        Encoder: 3 downsampling stages via MaxPool2d, each with a
                 double-conv block (feature maps: base → 2×base → 4×base).
        Bottleneck: double-conv at the lowest resolution (4×base → 8×base).
        Decoder: 3 upsampling stages via bilinear interpolation + concat
                 skip, each with a double-conv block.
        Head:   1×1 conv producing per-class logits.

    Args:
        num_classes: Number of segmentation classes (default 5).
        base:        Base channel count; doubles at each encoder level.
    """

    def __init__(self, num_classes: int = 5, base: int = 16):
        super().__init__()
        self.enc1 = double_conv(3, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.bottleneck = double_conv(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)
        self.dec3 = double_conv(base * 8 + base * 4, base * 4)
        self.dec2 = double_conv(base * 4 + base * 2, base * 2)
        self.dec1 = double_conv(base * 2 + base, base)
        self.head = nn.Conv2d(base, num_classes, 1)

    def _decode(
        self, x: torch.Tensor, skip: torch.Tensor, block: nn.Module
    ) -> torch.Tensor:
        """Upsample x to match skip's spatial dims, concatenate, and convolve."""
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return block(torch.cat([x, skip], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape (N, 3, H, W) in [0, 1].

        Returns:
            Logit tensor of shape (N, num_classes, H, W).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d = self._decode(b, e3, self.dec3)
        d = self._decode(d, e2, self.dec2)
        d = self._decode(d, e1, self.dec1)
        return self.head(d)


def model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (3, 256, 256)) -> None:
    """Print total and trainable parameter counts.

    Args:
        model:       The model to summarise.
        input_shape: (C, H, W) — used for output shape reporting.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 45)
    print(f"  Model:              {model.__class__.__name__}")
    print(f"  Input shape:        (N, {', '.join(str(d) for d in input_shape)})")
    print(f"  Total parameters:   {total:,}")
    print(f"  Trainable params:   {trainable:,}")
    print("=" * 45)


def build_model(name: str = "unet", in_channels: int = 3, num_classes: int = 5, base_channels: int = 16) -> nn.Module:
    """Factory to build a model from config strings.

    Args:
        name:         Model name ('unet').
        in_channels:  Input channels (3 for RGB).
        num_classes:  Number of output classes.
        base_channels: Base feature map width.

    Raises:
        ValueError: If the model name is not recognised.
    """
    if name == "unet":
        return UNet(num_classes=num_classes, base=base_channels)
    raise ValueError(f"Unknown model: {name!r}. Only 'unet' is supported.")
