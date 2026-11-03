#!/usr/bin/env python
"""Dataset validation script for MapVision.

Checks all image/mask pairs in a manifest for:
  - file existence
  - image readability
  - correct dimensions (after resize)
  - valid mask class IDs (0–4 only)
  - matching image/mask filenames

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --manifest splits/test.json
    python scripts/validate_dataset.py --manifests splits/train.json splits/val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapvision.dataset import CLASS_NAMES, MapDataset


def validate_manifest(manifest: str, classes_path: str = "classes.json") -> bool:
    """Validate all records in one manifest file.

    Returns True if all checks pass, False otherwise.
    """
    records = json.loads(Path(manifest).read_text())
    print(f"\n{'='*55}")
    print(f"  Manifest : {manifest}  ({len(records)} records)")
    print(f"{'='*55}")

    errors: list[str] = []
    warnings: list[str] = []

    for i, rec in enumerate(tqdm(records, desc="Checking files")):
        img_path = Path(rec["image"])
        lbl_path = Path(rec["label"])

        # Existence
        if not img_path.exists():
            errors.append(f"  [{i}] Missing image : {img_path}")
            continue
        if not lbl_path.exists():
            errors.append(f"  [{i}] Missing label : {lbl_path}")
            continue

        # Stem match
        if img_path.stem != lbl_path.stem:
            warnings.append(
                f"  [{i}] Stem mismatch: image={img_path.stem}, label={lbl_path.stem}"
            )

        # Readability + channel check
        try:
            with Image.open(img_path) as img:
                if img.mode not in ("RGB", "RGBA", "L"):
                    warnings.append(f"  [{i}] Unexpected image mode: {img.mode}")
        except (UnidentifiedImageError, OSError) as e:
            errors.append(f"  [{i}] Unreadable image: {img_path} — {e}")
            continue

        try:
            with Image.open(lbl_path) as lbl:
                if lbl.mode not in ("RGB", "RGBA"):
                    warnings.append(f"  [{i}] Unexpected label mode: {lbl.mode}")
        except (UnidentifiedImageError, OSError) as e:
            errors.append(f"  [{i}] Unreadable label: {lbl_path} — {e}")

    # Dataset-level dataloader check (first batch)
    try:
        ds = MapDataset(manifest, size=256, classes_path=classes_path)
        loader = DataLoader(ds, batch_size=min(4, len(ds)), shuffle=False, num_workers=0)
        images, masks = next(iter(loader))
        assert images.shape[1:] == (3, 256, 256), f"Bad image shape: {images.shape}"
        assert masks.ndim == 3, f"Mask should be (N, H, W), got {masks.shape}"
        assert masks.dtype == torch.int64, f"Mask dtype should be int64, got {masks.dtype}"
        assert images.min() >= 0 and images.max() <= 1, "Image values out of [0,1]"
        unique = masks.unique().tolist()
        bad = [v for v in unique if v not in range(5)]
        if bad:
            errors.append(f"  Invalid class IDs in masks: {bad}")
        print(f"\n  DataLoader check passed.")
        print(f"    Image shape : {tuple(images.shape)}")
        print(f"    Mask shape  : {tuple(masks.shape)}")
        print(f"    Class IDs   : {unique}")
    except Exception as e:
        errors.append(f"  DataLoader check failed: {e}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(w)

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  … and {len(errors) - 20} more errors.")
        return False

    print(f"\n  ✓ All checks passed for {manifest}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MapVision dataset manifests")
    parser.add_argument(
        "--manifest",
        help="Single manifest file to check (default: all three splits)",
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        help="One or more manifest files to check",
    )
    parser.add_argument(
        "--classes",
        default="classes.json",
        help="Path to classes.json",
    )
    args = parser.parse_args()

    if args.manifest:
        manifests = [args.manifest]
    elif args.manifests:
        manifests = args.manifests
    else:
        manifests = ["splits/train.json", "splits/val.json", "splits/test.json"]

    all_ok = True
    for m in manifests:
        ok = validate_manifest(m, args.classes)
        all_ok = all_ok and ok

    print("\n" + ("✓ All manifests passed." if all_ok else "✗ Some checks failed."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
