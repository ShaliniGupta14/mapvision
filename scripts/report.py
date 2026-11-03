#!/usr/bin/env python
"""Print a resume-ready metrics summary from measured run artifacts.

Every value is read from files written by train.py / evaluate.py, or
measured here (latency). Anything not yet available prints as TBD.

Usage:
    python scripts/report.py --run runs/unet-80ep --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import get_device
from mapvision.inference import load_checkpoint, measure_latency
from mapvision.utils import load_json

TBD = "TBD — requires training/evaluation"


def _maybe_json(path: Path):
    return load_json(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise measured MapVision results")
    parser.add_argument("--run", required=True, help="Run directory with config.json and history.json")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Checkpoint for latency")
    parser.add_argument("--metrics", default="outputs/metrics/test_metrics.json", help="Test metrics JSON")
    args = parser.parse_args()

    run = Path(args.run)
    config = _maybe_json(run / "config.json")
    history = _maybe_json(run / "history.json")
    test = _maybe_json(Path(args.metrics))

    rows = {"Dataset size": TBD, "Training samples": TBD, "Validation samples": TBD,
            "Test samples": TBD, "Model parameters": TBD, "Best validation mIoU": TBD,
            "Test mIoU": TBD, "Test Dice": TBD, "Pixel accuracy": TBD,
            "Inference latency": TBD, "Training device": TBD}

    if config:
        rows["Training samples"] = str(config["train_samples"])
        rows["Validation samples"] = str(config["val_samples"])
        rows["Model parameters"] = f"{config['total_params']:,}"
        rows["Training device"] = f"{config['device']} (torch {config['torch_version']})"
    if history:
        best = max(history, key=lambda r: r["val_miou"])
        minutes = sum(r["seconds"] for r in history) / 60
        rows["Best validation mIoU"] = f"{best['val_miou']:.4f} (epoch {best['epoch']})"
        rows["Training device"] += f", {len(history)} epochs in {minutes:.1f} min"
    if test:
        rows["Test samples"] = str(test["num_samples"])
        rows["Test mIoU"] = f"{test['mean_iou']:.4f}"
        rows["Test Dice"] = f"{test['mean_dice']:.4f}"
        rows["Pixel accuracy"] = f"{test['pixel_accuracy']:.4f}"
    if config and test:
        total = config["train_samples"] + config["val_samples"] + test["num_samples"]
        rows["Dataset size"] = f"{total} annotated 1000x1000 tiles used (Paris partition)"

    if Path(args.checkpoint).exists():
        device = get_device()
        model, ckpt_cfg, _, _ = load_checkpoint(args.checkpoint, device)
        latency = measure_latency(model, device)
        rows["Inference latency"] = f"{latency:.1f} ms / 256x256 tile (median, batch 1, {device})"

    print("\nMAPVISION RESUME METRICS")
    print("------------------------")
    for name, value in rows.items():
        print(f"{name + ':':<22}{value}")


if __name__ == "__main__":
    main()
