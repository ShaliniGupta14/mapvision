#!/usr/bin/env python
"""Dataset validation script for MapVision.

Checks all image/mask pairs in a manifest for:
  - file existence and readability
  - matching image/mask filenames
  - supported image channels
  - matching, non-empty image/mask dimensions
  - valid mask class IDs (0–4 only)

then loads one batch through a DataLoader to confirm tensor shapes and dtypes.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --manifest splits/test.json
    python scripts/validate_dataset.py --manifests splits/train.json splits/val.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapvision.dataset import CLASS_NAMES, DatasetError, MapDataset, validate_record


def check_first_batch(dataset: MapDataset, size: int) -> None:
    """Load one batch and assert the shape/dtype/range contract."""
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
    images, masks = next(iter(loader))
    n = images.shape[0]
    assert images.shape == (n, 3, size, size), f"Bad image shape: {tuple(images.shape)}"
    assert masks.shape == (n, size, size), f"Bad mask shape: {tuple(masks.shape)}"
    assert images.dtype == torch.float32, f"Image dtype should be float32, got {images.dtype}"
    assert masks.dtype == torch.int64, f"Mask dtype should be int64, got {masks.dtype}"
    assert images.min() >= 0 and images.max() <= 1, "Image values out of [0, 1]"
    print(f"  Images: {images.shape} {images.dtype}")
    print(f"  Masks:  {masks.shape} {masks.dtype}")
    print(f"  Class IDs in batch: {masks.unique().tolist()}")
    print(f"  Class order: {CLASS_NAMES}")


def validate_manifest(manifest: str, classes_path: str, size: int) -> bool:
    """Validate all records in one manifest file. Returns True if all pass."""
    print(f"\n{'=' * 55}")
    try:
        dataset = MapDataset(manifest, size=size, classes_path=classes_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {manifest}: {exc}")
        return False
    print(f"  Manifest : {manifest}  ({len(dataset)} records)")
    print(f"{'=' * 55}")

    errors = []
    for i, record in enumerate(tqdm(dataset.records, desc="Checking pairs")):
        try:
            validate_record(record, dataset.colors)
        except DatasetError as exc:
            errors.append(f"  [{i}] {exc}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(e)
        if len(errors) > 20:
            print(f"  … and {len(errors) - 20} more errors.")
        return False

    try:
        check_first_batch(dataset, size)
    except (AssertionError, DatasetError, OSError) as exc:
        print(f"\n  DataLoader check failed: {exc}")
        return False

    print(f"\n  ✓ All checks passed for {manifest}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MapVision dataset manifests")
    parser.add_argument("--manifest", help="Single manifest file to check")
    parser.add_argument("--manifests", nargs="+", help="One or more manifest files to check")
    parser.add_argument("--classes", default="classes.json", help="Path to classes.json")
    parser.add_argument("--size", type=int, default=256, help="Expected tile size")
    args = parser.parse_args()

    if args.manifest:
        manifests = [args.manifest]
    elif args.manifests:
        manifests = args.manifests
    else:
        manifests = ["splits/train.json", "splits/val.json", "splits/test.json"]

    results = [validate_manifest(m, args.classes, args.size) for m in manifests]
    all_ok = all(results)
    print("\n" + ("✓ All manifests passed." if all_ok else "✗ Some checks failed."))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
