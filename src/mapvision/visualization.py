"""Visualization utilities for MapVision.

Provides a stable color palette, mask rendering, overlay generation, and
grid plots for comparing ground-truth vs predicted segmentations.
All colors are fixed per class — no random variation between runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mapvision.dataset import CLASS_NAMES

# Fixed class color palette (RGB, 0–255) — stable across all runs
CLASS_COLORS = np.array(
    [
        [30, 30, 30],       # 0  frame       — dark grey
        [70, 130, 200],     # 1  water       — steel blue
        [220, 100, 60],     # 2  blocks      — terracotta
        [120, 190, 100],    # 3  non-built   — olive green
        [240, 240, 220],    # 4  road_network — cream
    ],
    dtype=np.uint8,
)
assert len(CLASS_COLORS) == len(CLASS_NAMES)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert a class-ID mask to an RGB image using the fixed palette.

    Args:
        mask: Int array of shape (H, W) with values in [0, num_classes).

    Returns:
        uint8 RGB array of shape (H, W, 3).
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[mask == cls_id] = color
    return rgb


def tensor_to_numpy_image(image: torch.Tensor) -> np.ndarray:
    """Convert a float image tensor (3, H, W) in [0, 1] to uint8 (H, W, 3)."""
    img = image.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a class-color mask onto an RGB image.

    Args:
        image: uint8 RGB array (H, W, 3).
        mask:  Int class-ID array (H, W).
        alpha: Mask opacity in [0, 1].

    Returns:
        uint8 blended RGB array (H, W, 3).
    """
    mask_rgb = mask_to_rgb(mask).astype(float)
    image_f = image.astype(float)
    blended = image_f * (1 - alpha) + mask_rgb * alpha
    return blended.clip(0, 255).astype(np.uint8)


def _legend_patches() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=tuple(c / 255 for c in CLASS_COLORS[i]),
            label=CLASS_NAMES[i],
        )
        for i in range(len(CLASS_NAMES))
    ]


def save_prediction_grid(
    image: torch.Tensor,
    gt_mask: Optional[torch.Tensor],
    pred_mask: torch.Tensor,
    save_path: str | Path,
    title: str = "",
) -> None:
    """Save a side-by-side grid: source image | [GT mask] | pred mask | overlay.

    Args:
        image:      Float tensor (3, H, W) in [0, 1].
        gt_mask:    Optional int64 tensor (H, W) — ground truth. May be None.
        pred_mask:  Int64 tensor (H, W) — predicted classes.
        save_path:  Output PNG file path.
        title:      Optional suptitle string.
    """
    img_np = tensor_to_numpy_image(image)
    pred_np = pred_mask.cpu().numpy()
    pred_rgb = mask_to_rgb(pred_np)
    overlay = overlay_mask(img_np, pred_np)

    panels = [img_np]
    labels = ["Input Image"]
    if gt_mask is not None:
        gt_np = gt_mask.cpu().numpy()
        panels.append(mask_to_rgb(gt_np))
        labels.append("Ground Truth")
    panels.extend([pred_rgb, overlay])
    labels.extend(["Prediction", "Overlay"])

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5))
    for ax, panel, lbl in zip(axes, panels, labels):
        ax.imshow(panel)
        ax.set_title(lbl, fontsize=11)
        ax.axis("off")

    fig.legend(
        handles=_legend_patches(),
        loc="lower center",
        ncol=len(CLASS_NAMES),
        fontsize=9,
        frameon=False,
    )
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _save_rgb(rgb: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(save_path)


def save_mask_only(mask: torch.Tensor, save_path: str | Path) -> None:
    """Save a class-ID mask as a colorized PNG.

    Args:
        mask:      Int64 tensor (H, W).
        save_path: Output file path.
    """
    _save_rgb(mask_to_rgb(mask.cpu().numpy()), save_path)


def save_image(image: torch.Tensor, save_path: str | Path) -> None:
    """Save a float image tensor (3, H, W) in [0, 1] as a PNG."""
    _save_rgb(tensor_to_numpy_image(image), save_path)


def save_overlay(
    image: torch.Tensor, mask: torch.Tensor, save_path: str | Path, alpha: float = 0.45
) -> None:
    """Save the class-color mask blended over the source image as a PNG."""
    _save_rgb(overlay_mask(tensor_to_numpy_image(image), mask.cpu().numpy(), alpha), save_path)


def plot_training_history(history: List[dict], save_path: str | Path) -> None:
    """Plot train/val loss and val mIoU/Dice per epoch to a PNG.

    Args:
        history:   Rows from a run's history.json.
        save_path: Output PNG file path.
    """
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="val")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_loss.legend()
    ax_metric.plot(epochs, [r["val_miou"] for r in history], label="val mIoU")
    ax_metric.plot(epochs, [r["val_dice"] for r in history], label="val Dice")
    ax_metric.set(title="Validation metrics", xlabel="epoch", ylim=(0, 1))
    ax_metric.legend()
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
