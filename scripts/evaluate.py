#!/usr/bin/env python
"""Evaluation script for MapVision.

Loads a trained checkpoint and evaluates it on the test split, printing
full metrics and saving results to outputs/metrics/test_metrics.json.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from mapvision.config import get_device, load_config
from mapvision.dataset import CLASS_NAMES, MapDataset
from mapvision.evaluate import evaluate
from mapvision.inference import load_checkpoint
from mapvision.transforms import ValTransform
from mapvision.utils import format_metrics, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MapVision checkpoint")
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
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which data split to evaluate on",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics",
        help="Directory to save metrics JSON",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.device.prefer_mps)

    model, ckpt_config, epoch, best_miou = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint: epoch {epoch}, best val mIoU = {best_miou:.4f}")

    manifest_map = {
        "train": cfg.data.train_manifest,
        "val": cfg.data.val_manifest,
        "test": cfg.data.test_manifest,
    }
    dataset = MapDataset(
        manifest_map[args.split],
        size=cfg.data.image_size,
        classes_path=cfg.data.classes_path,
        transform=ValTransform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    print(f"Evaluating on {args.split} split ({len(dataset)} samples)…")
    results = evaluate(model, loader, device, cfg.data.num_classes)

    print(format_metrics(results))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_metrics.json"
    save_json(results, out_path)
    print(f"\nMetrics saved to: {out_path}")


if __name__ == "__main__":
    main()
