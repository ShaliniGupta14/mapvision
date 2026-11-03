#!/usr/bin/env python
"""Prediction script for MapVision.

Runs inference on a single image and saves the segmentation output,
including colorized mask, overlay, and a side-by-side comparison grid.

Usage:
    python scripts/predict.py \
        --image path/to/image.png \
        --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import get_device, load_config
from mapvision.inference import load_checkpoint, predict_image
from mapvision.visualization import save_mask_only, save_prediction_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MapVision inference on an image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        default="outputs/predictions",
        help="Directory to save prediction outputs",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.device.prefer_mps)

    model, ckpt_config, epoch, _ = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint (epoch {epoch})")

    print(f"Predicting: {args.image}")
    image_tensor, pred_mask = predict_image(
        args.image, model, device, cfg.data.image_size
    )

    out_dir = Path(args.output)
    stem = Path(args.image).stem

    grid_path = out_dir / f"{stem}_prediction_grid.png"
    mask_path = out_dir / f"{stem}_mask.png"

    save_prediction_grid(
        image=image_tensor,
        gt_mask=None,
        pred_mask=pred_mask,
        save_path=grid_path,
        title=f"MapVision Prediction — {Path(args.image).name}",
    )
    save_mask_only(pred_mask, mask_path)

    print(f"Grid saved  : {grid_path}")
    print(f"Mask saved  : {mask_path}")


if __name__ == "__main__":
    main()
