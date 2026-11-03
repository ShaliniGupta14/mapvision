"""General utility helpers for MapVision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Serialize data to a JSON file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent) + "\n")


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file."""
    return json.loads(Path(path).read_text())


def _fmt(val: Optional[float]) -> str:
    return f"{val:.4f}" if val is not None else "N/A"


def format_metrics(results: Dict, title: str = "Evaluation Results") -> str:
    """Format a metrics dict into a human-readable report string."""
    lines = [
        "=" * 56,
        f"  {title}",
        "=" * 56,
        f"  Pixel Accuracy : {results['pixel_accuracy']:.4f}",
        f"  Mean IoU       : {results['mean_iou']:.4f}",
        f"  Mean Dice      : {results['mean_dice']:.4f}",
        "",
        f"  {'Class':<14}{'IoU':>9}{'Dice':>9}{'Prec':>9}{'Recall':>9}",
        "  " + "-" * 50,
    ]
    for name in results["per_class_iou"]:
        lines.append(
            f"  {name:<14}"
            + "".join(
                f"{_fmt(results[key].get(name)):>9}"
                for key in ("per_class_iou", "per_class_dice",
                            "per_class_precision", "per_class_recall")
            )
        )

    names = list(results["per_class_iou"])
    lines += ["", "  Confusion matrix (rows = ground truth, cols = predicted)"]
    lines.append("  " + " " * 14 + "".join(f"{n[:10]:>12}" for n in names))
    for name, row in zip(names, results["confusion_matrix"]):
        lines.append(f"  {name:<14}" + "".join(f"{v:>12,}" for v in row))
    lines.append("=" * 56)
    return "\n".join(lines)
