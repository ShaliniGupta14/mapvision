#!/usr/bin/env python
"""Build the static web demo's data files from measured artifacts.

Writes into web/:
  samples/<stem>.jpg        256x256 test tiles (bilinear, like the dataset)
  samples/<stem>_gt.png     their ground-truth masks in the demo palette
  data.json                 class palette, sample list and the measured
                            test metrics / run summary of the deployed model

Run export_onnx.py first so web/model/mapvision.onnx matches the run.

Usage:
    python scripts/build_web.py --run runs/unet-80ep \
        --metrics outputs/metrics/test_metrics.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PIL import Image

from mapvision.dataset import CLASS_NAMES, MapDataset
from mapvision.utils import load_json, save_json
from mapvision.visualization import CLASS_COLORS, mask_to_rgb, tensor_to_numpy_image

DEFAULT_SAMPLES = [
    "12148_btv1b53062148cf1_0",
    "12148_btv1b53058291ff1_0",
    "12148_btv1b53062342ff1_0",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build web demo data files")
    parser.add_argument("--run", required=True, help="Run directory of the deployed model")
    parser.add_argument("--metrics", default="outputs/metrics/test_metrics.json")
    parser.add_argument("--manifest", default="splits/test.json")
    parser.add_argument("--web", default="web")
    parser.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES, help="Test tile stems")
    args = parser.parse_args()

    web = Path(args.web)
    sample_dir = web / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataset = MapDataset(args.manifest)
    by_stem = {Path(r["image"]).stem: i for i, r in enumerate(dataset.records)}
    missing = [s for s in args.samples if s not in by_stem]
    if missing:
        sys.exit(f"Error: samples not in {args.manifest}: {missing}")

    samples = []
    for stem in args.samples:
        image, mask = dataset[by_stem[stem]]
        Image.fromarray(tensor_to_numpy_image(image)).save(sample_dir / f"{stem}.jpg", quality=90)
        Image.fromarray(mask_to_rgb(mask.numpy())).save(sample_dir / f"{stem}_gt.png")
        samples.append({"id": stem, "image": f"samples/{stem}.jpg", "gt": f"samples/{stem}_gt.png"})

    config = load_json(Path(args.run) / "config.json")
    history = load_json(Path(args.run) / "history.json")
    metrics = load_json(args.metrics)
    best = max(history, key=lambda r: r["val_miou"])

    save_json(
        {
            "classes": [
                {"name": name, "color": CLASS_COLORS[i].tolist()}
                for i, name in enumerate(CLASS_NAMES)
            ],
            "image_size": config["data"]["image_size"],
            "samples": samples,
            "model": {
                "run": Path(args.run).name,
                "parameters": config["total_params"],
                "epochs_trained": len(history),
                "best_epoch": best["epoch"],
                "best_val_miou": best["val_miou"],
                "loss": config["loss"],
                "train_samples": config["train_samples"],
                "val_samples": config["val_samples"],
            },
            "test": {
                key: metrics[key]
                for key in ("num_samples", "pixel_accuracy", "mean_iou", "mean_dice",
                            "per_class_iou", "per_class_dice")
            },
        },
        web / "data.json",
    )
    print(f"Wrote {len(samples)} samples and {web / 'data.json'}")


if __name__ == "__main__":
    main()
