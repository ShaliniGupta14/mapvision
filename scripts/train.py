#!/usr/bin/env python
"""Training entry-point for MapVision.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 10 --output runs/my_run
    python scripts/train.py --config configs/default.yaml --output runs/my_run \
        --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mapvision.config import load_config
from mapvision.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MapVision segmentation model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Override batch size")
    parser.add_argument("--output", help="Output directory for this run (under runs/)")
    parser.add_argument("--resume", help="Checkpoint to resume from, e.g. checkpoints/latest.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size

    train(cfg, output_dir=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
