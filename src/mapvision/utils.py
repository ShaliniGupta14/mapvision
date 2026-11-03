"""General utility helpers for MapVision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize data to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n")


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file."""
    return json.loads(Path(path).read_text())


def format_metrics(results: Dict) -> str:
    """Format a metrics dict into a human-readable report string."""
    lines = [
        "=" * 45,
        "  Evaluation Results",
        "=" * 45,
        f"  Pixel Accuracy : {results['pixel_accuracy']:.4f}",
        f"  Mean IoU       : {results['mean_iou']:.4f}",
        f"  Mean Dice      : {results['mean_dice']:.4f}",
        "",
        "  Per-Class IoU",
        "  " + "-" * 30,
    ]
    for name, val in results["per_class_iou"].items():
        val_str = f"{val:.4f}" if val is not None else "N/A"
        lines.append(f"  {name:<16}: {val_str}")
    lines.append("")
    lines.append("  Per-Class Dice")
    lines.append("  " + "-" * 30)
    for name, val in results["per_class_dice"].items():
        val_str = f"{val:.4f}" if val is not None else "N/A"
        lines.append(f"  {name:<16}: {val_str}")
    lines.append("=" * 45)
    return "\n".join(lines)
