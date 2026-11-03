#!/usr/bin/env python
"""Evaluation script for MapVision.

Loads a trained checkpoint and evaluates it on a data split (test by
default), printing full metrics and saving them to
outputs/metrics/<split>_metrics.json.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --save-examples 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader

from mapvision.config import get_device, load_config
from mapvision.dataset import MapDataset
from mapvision.evaluate import evaluate
from mapvision.inference import load_checkpoint
from mapvision.transforms import ValTransform
from mapvision.utils import format_metrics, save_json
from mapvision.visualization import save_prediction_grid


def save_examples(
    model: torch.nn.Module,
    dataset: MapDataset,
    device: torch.device,
    count: int,
    out_dir: Path,
) -> None:
    """Save input | ground truth | prediction | overlay grids for the first samples."""
    model.eval()
    for index in range(min(count, len(dataset))):
        image, mask = dataset[index]
        with torch.inference_mode():
            pred = model(image.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
        stem = Path(dataset.records[index]["image"]).stem
        path = out_dir / f"{stem}_gt_vs_pred.png"
        save_prediction_grid(image, mask, pred, path, title=stem)
        print(f"Example saved: {path}")


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
    parser.add_argument(
        "--save-examples",
        type=int,
        default=0,
        dest="save_examples",
        help="Save this many ground-truth vs prediction grids to outputs/plots",
    )
    args = parser.parse_args()

    manifest_keys = {"train": "train_manifest", "val": "val_manifest", "test": "test_manifest"}
    try:
        cfg = load_config(args.config)
        device = get_device(cfg.device.prefer_mps)
        model, _, epoch, best_miou = load_checkpoint(args.checkpoint, device)
        dataset = MapDataset(
            getattr(cfg.data, manifest_keys[args.split]),
            size=cfg.data.image_size,
            classes_path=cfg.data.classes_path,
            transform=ValTransform(),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    print(f"Loaded checkpoint: epoch {epoch}, best val mIoU = {best_miou:.4f}")
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    print(f"Evaluating on {args.split} split ({len(dataset)} samples)…")
    results = evaluate(model, loader, device, cfg.data.num_classes)
    print(format_metrics(results, title=f"{args.split.capitalize()} Results"))

    out_path = Path(args.output) / f"{args.split}_metrics.json"
    save_json(
        {
            "split": args.split,
            "num_samples": len(dataset),
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": epoch,
            **results,
        },
        out_path,
    )
    print(f"\nMetrics saved to: {out_path}")

    if args.save_examples:
        save_examples(model, dataset, device, args.save_examples, Path(cfg.paths.outputs) / "plots")


if __name__ == "__main__":
    main()
