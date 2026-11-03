#!/usr/bin/env python
"""Export a MapVision checkpoint to ONNX for in-browser inference.

The exported graph takes float32 input (N, 3, 256, 256) in [0, 1] and
returns logits (N, 5, 256, 256), matching the PyTorch model. After
export, outputs are checked against PyTorch with ONNX Runtime.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pt \
        --output web/model/mapvision.onnx

Requires the extra packages in requirements-export.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from mapvision.inference import load_checkpoint

TOLERANCE = 1e-3


def export(checkpoint: str, output: Path, image_size: int) -> None:
    """Export the checkpoint's model to ``output`` and verify it with ONNX Runtime."""
    import onnxruntime as ort

    model, _, epoch, best = load_checkpoint(checkpoint, torch.device("cpu"))
    dummy = torch.rand(1, 3, image_size, image_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    with torch.inference_mode():
        expected = model(dummy).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"image": dummy.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    agreement = float((expected.argmax(1) == actual.argmax(1)).mean())
    if max_diff > TOLERANCE:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff:.2e}")

    size_mb = output.stat().st_size / 1e6
    print(f"Exported epoch-{epoch} checkpoint (val mIoU {best:.4f}) → {output} ({size_mb:.2f} MB)")
    print(f"Max |logit diff| vs PyTorch: {max_diff:.2e}; argmax agreement: {agreement:.4%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a MapVision checkpoint to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output", default="web/model/mapvision.onnx")
    parser.add_argument("--image-size", type=int, default=256, dest="image_size")
    args = parser.parse_args()
    try:
        export(args.checkpoint, Path(args.output), args.image_size)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
