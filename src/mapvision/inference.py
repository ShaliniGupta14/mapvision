"""Inference pipeline for MapVision.

Loads a trained checkpoint, runs prediction on a single image, and
returns the predicted class mask and associated tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

from mapvision.dataset import CLASS_NAMES
from mapvision.model import build_model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple:
    """Load model and config from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint saved during training.
        device:          Target device.

    Returns:
        Tuple of (model, config_dict, epoch, best_metric).

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        KeyError:          If checkpoint lacks required keys.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model_state", "config"):
        if key not in ckpt:
            raise KeyError(f"Checkpoint missing required key: {key!r}")

    cfg = ckpt["config"]
    model_cfg = cfg.get("model", {})
    model = build_model(
        name=model_cfg.get("name", "unet"),
        num_classes=cfg.get("num_classes", 5) or model_cfg.get("num_classes", 5),
        base_channels=model_cfg.get("base_channels", 16),
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, cfg, ckpt.get("epoch", -1), ckpt.get("val_miou", float("nan"))


def predict_image(
    image_path: str | Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a single image file.

    Args:
        image_path: Path to a PNG/JPEG image.
        model:      Loaded model in eval mode.
        device:     Compute device.
        image_size: Input resolution for the model.

    Returns:
        (image_tensor, pred_mask) where:
          - image_tensor is float32 (3, H, W) in [0, 1]
          - pred_mask is int64 (H, W) with class IDs 0–4
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as src:
        img = src.convert("RGB").resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        img_np = np.array(img, dtype=np.float32) / 255.0

    image_tensor = torch.from_numpy(img_np.transpose(2, 0, 1).copy())
    batch = image_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(batch)

    pred_mask = logits.argmax(dim=1).squeeze(0).cpu()
    return image_tensor, pred_mask
