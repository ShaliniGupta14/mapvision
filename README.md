# MapVision

**Semantic segmentation of historical map imagery with a compact U-Net.**

MapVision is a semantic segmentation system that converts historical map imagery into structured pixel-level geographic layers. Given an RGB map tile, it predicts one of five classes for every pixel (frame, water, building blocks, non-built land, road network) and runs end to end on Apple Silicon (PyTorch MPS), CUDA, or CPU.

---

## Table of Contents

- [Problem](#problem)
- [Classes](#classes)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Dataset and Splits](#dataset-and-splits)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Sample Outputs](#sample-outputs)
- [Tests](#tests)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Citation and License](#citation-and-license)

---

## Problem

Historical maps record how cities looked over centuries: their street networks, building blocks, rivers and open land. That information is hard to extract automatically because of:

- paper aging, stains and fading
- scanning artifacts
- complex cartographic symbols and hatching
- overlapping structures and text labels
- widely varying color schemes between publishers and eras
- noisy, hand-drawn boundaries

MapVision learns a pixel-wise mapping from scanned map tiles to five semantic layers, so they can be analysed, compared across time, or vectorised.

---

## Classes

Class order is fixed and shared by the dataset, model, metrics and visualizations (`mapvision.dataset.CLASS_NAMES`).

| ID | Class          | Label color (`classes.json`) | Meaning                                              |
|----|----------------|------------------------------|------------------------------------------------------|
| 0  | `frame`        | black `(0, 0, 0)`            | Non-map area: margins, title boxes, decorations      |
| 1  | `water`        | blue `(0, 0, 255)`           | Rivers, canals, lakes, the sea                       |
| 2  | `blocks`       | magenta `(255, 0, 255)`      | Built-up city blocks and buildings                   |
| 3  | `non-built`    | cyan `(0, 255, 255)`         | Parks, gardens, fields, unbuilt land                 |
| 4  | `road_network` | white `(255, 255, 255)`      | Streets, roads, squares, bridges                     |

Pixel share across the 227 tiles in the train/val/test splits (measured at native resolution): blocks 27.4%, frame 26.5%, non-built 22.4%, road_network 20.8%, **water 2.8%**. Water is the clearly under-represented class; `loss.class_weights` can up-weight it.

---

## Architecture

A lightweight U-Net (`mapvision.model.UNet`, 488,069 parameters at `base_channels: 16`):

```
Input (3×256×256)
  Enc1  double-conv →  16 ch ───────────────────────────── skip ─┐
  MaxPool                                                        │
  Enc2  double-conv →  32 ch ─────────────────── skip ─┐         │
  MaxPool                                              │         │
  Enc3  double-conv →  64 ch ───────── skip ─┐         │         │
  MaxPool                                    │         │         │
  Bottleneck double-conv → 128 ch            │         │         │
  Upsample (bilinear) + concat ──────────────┘         │         │
  Dec3  double-conv →  64 ch                           │         │
  Upsample (bilinear) + concat ────────────────────────┘         │
  Dec2  double-conv →  32 ch                                     │
  Upsample (bilinear) + concat ──────────────────────────────────┘
  Dec1  double-conv →  16 ch
  1×1 conv head → 5 ch
Output logits (5×256×256)
```

- Each double-conv block is two 3×3 convolutions with BatchNorm and ReLU.
- Upsampling uses bilinear interpolation followed by convolution. This works well on MPS and avoids checkerboard artifacts.
- Channel width scales with `model.base_channels` (doubling at every level).

---

## Pipeline

```text
Historical Map
      |
      v
Preprocessing      RGB conversion, resize (bilinear for images,
      |            nearest-neighbour for masks), scale to [0, 1]
      v
256x256 Tile       float32 (3, 256, 256) image + int64 (256, 256) mask
      |
      v
U-Net              augmentations on train only: flips, 90° rotations,
      |            brightness/contrast
      v
5-Class Segmentation   logits (5, 256, 256) → argmax
      |
      v
Structured Map Layers  colorized mask, overlay, per-class metrics
```

Masks are stored as RGB PNGs and mapped to class IDs by exact color match. Any pixel that does not match a class color raises a `DatasetError`, so mislabelled or anti-aliased masks fail early and loudly.

---

## Dataset and Splits

Data comes from the [Historical City Maps Semantic Segmentation Dataset](https://zenodo.org/record/5513639) (Petitpierre, 2021). See [docs/DATASET.md](docs/DATASET.md) for the original dataset card. This project uses the **Paris** partition: 330 annotated 1000×1000 patches from BnF and BHVP maps. The World partition is not used yet.

Expected layout (not committed to git):

```
data/raw/paris/images/*.png        300 development images
data/raw/paris/labels/*.png        matching color-coded masks
data/raw/paris/eval/images/*.png   held-out evaluation images
data/raw/paris/eval/labels/*.png
```

The splits in `splits/` are explicit, committed manifests (`{"image", "label", "source_group"}`), and are preserved as-is:

| Split | Tiles | Source                                                              |
|-------|------:|---------------------------------------------------------------------|
| train | 157   | `paris/images`, grouped by source map                               |
| val   | 40    | `paris/images`, held-out source maps (seed 42)                      |
| test  | 30    | the dataset's own `paris/eval` partition                            |
| excluded | 103 | development tiles cut from a map that also appears in test        |

Tiles are grouped by source map (the filename stem minus its trailing `_<n>`), so tiles from the same map never span splits. Development tiles that share a map with the test set are excluded entirely to prevent leakage. The method is recorded in `splits/protocol.json`, and `tests/test_dataset.py` asserts that the three splits share no file and no source group. The split is roughly 70/18/13 by tile instead of the textbook 70/15/15, because the test set is the dataset's official evaluation partition.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ is required, and the project was developed on Python 3.12 with PyTorch 2.15. On Apple Silicon, MPS is used automatically. Otherwise the code falls back to CUDA, then CPU, and the selected device is printed at startup. All commands below are run from the repository root.

---

## Usage

### Validate the dataset

```bash
python scripts/validate_dataset.py                           # all three splits
python scripts/validate_dataset.py --manifest splits/test.json
```

Each record is checked for missing files, mismatched image/label filenames, unreadable files, unsupported channels, empty or mismatched dimensions, and unknown label colors. Then one batch is loaded to confirm the tensor contract (`(N,3,256,256)` float32 in [0, 1], `(N,256,256)` int64).

### Train

```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --epochs 2           # quick run
python scripts/train.py --config configs/default.yaml --output runs/exp01 \
    --resume checkpoints/latest.pt                                          # resume
```

Every epoch validates on the val split and writes `checkpoints/latest.pt`. When val mIoU improves, it also writes `checkpoints/best.pt`. Checkpoints hold model, optimizer and scheduler state, the epoch, the best val mIoU, the full config and the class names. The run directory receives `config.json`, `history.json` and `training_curves.png`.

```text
Epoch 03/80  (10.0s)
  Train Loss: …
  Val Loss:   …
  Val mIoU:   …
  Val Dice:   …
  Saved new best checkpoint.
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt                  # test split
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
python scripts/evaluate.py --checkpoint checkpoints/best.pt --save-examples 4
```

This prints pixel accuracy, mean IoU, mean Dice, a per-class IoU/Dice/precision/recall table and the 5×5 confusion matrix, and saves everything to `outputs/metrics/<split>_metrics.json`. `--save-examples N` writes input | ground truth | prediction | overlay grids to `outputs/plots/`.

Mean IoU and mean Dice are averaged over classes that occur in the ground truth or predictions. A class that never occurs is reported as `N/A`, not 0.

### Predict

```bash
python scripts/predict.py --image sample.png --checkpoint checkpoints/best.pt
```

Writes to `outputs/predictions/`:

- `<stem>_input.png`: the resized 256×256 input
- `<stem>_mask.png`: colorized predicted mask
- `<stem>_overlay.png`: mask blended over the input
- `<stem>_prediction_grid.png`: side-by-side panels with a class legend

It also prints the median model latency for a single 256×256 tile.

### Summarise a run

```bash
python scripts/report.py --run runs/<run> --checkpoint checkpoints/best.pt
```

Prints the resume metrics block below from the run's `config.json` and `history.json`, `outputs/metrics/test_metrics.json`, and a fresh latency measurement. Anything not yet measured prints as `TBD`.

---

## Results

All numbers below were measured by the scripts in this repo: `train.py`, `evaluate.py` and `report.py`. The model is the default config (`base_channels: 16`, cross-entropy, AdamW 1e-3, ReduceLROnPlateau, augmentation on), trained on an Apple M-series GPU (MPS) with PyTorch 2.15. Test is the held-out `paris/eval` split. It was never used for model selection, and the best checkpoint was chosen by val mIoU.

| Metric (test, 30 tiles) | 25 epochs | **80 epochs (default)** |
|-------------------------|----------:|------------------------:|
| Pixel accuracy          | 0.5411    | **0.6459**              |
| Mean IoU                | 0.3383    | **0.4247**              |
| Mean Dice               | 0.4614    | **0.5445**              |
| Best val mIoU (epoch)   | 0.3896 (25) | **0.4456 (54)**       |
| Training time           | 4.3 min   | 14.6 min                |

At 25 epochs val mIoU was still rising at the last epoch. At 80 epochs it peaks at epoch 54 and plateaus as the scheduler decays the learning rate.

Per-class results for the 80-epoch model on test:

| Class          | IoU    | Dice   |
|----------------|-------:|-------:|
| frame          | 0.7671 | 0.8682 |
| water          | 0.0071 | 0.0140 |
| blocks         | 0.5270 | 0.6902 |
| non-built      | 0.3076 | 0.4705 |
| road_network   | 0.5146 | 0.6795 |

```text
MAPVISION RESUME METRICS
------------------------
Dataset size:         227 annotated 1000x1000 tiles used (Paris partition)
Training samples:     157
Validation samples:   40
Test samples:         30
Model parameters:     488,069
Best validation mIoU: 0.4456 (epoch 54)
Test mIoU:            0.4247
Test Dice:            0.5445
Pixel accuracy:       0.6459
Inference latency:    7.2 ms / 256x256 tile (median, batch 1, mps)
Training device:      mps (torch 2.15.0), 80 epochs in 14.6 min
```

Latency covers the model forward pass only, not image decoding, and varies from run to run on MPS. The same architecture measured 115 ms per tile on CPU (`device.prefer_mps: false`).

**Known limitations**

- **Water is essentially not learned** (IoU 0.007). It covers 2.8% of pixels, and the confusion matrix shows it predicted mostly as blocks and non-built. Setting `loss.class_weights` or `loss.type: ce_dice` is the first thing to try. Neither has been run yet.
- **Blocks and non-built are often confused**, especially plain (unhatched) blocks, which the model reads as open land.
- Each 1000×1000 patch is downsampled to 256×256, which thins streets to a few pixels. Native-resolution crops are the main planned improvement (see [Future Work](#future-work)).

---

## Sample Outputs

Test tile from the held-out split: source image, ground truth, prediction and overlay. The model was trained for 80 epochs.

![Input, ground truth, prediction and overlay for a Paris test tile](docs/images/sample_gt_vs_pred.png)

The road network is traced well. Many plain blocks are predicted as non-built (green), the most common error on this split.

Training curves for the 80-epoch run:

![Train/val loss and val mIoU/Dice per epoch](docs/images/training_curves.png)

Generate more with `python scripts/evaluate.py --checkpoint checkpoints/best.pt --save-examples N` (written to `outputs/plots/`), or for any image with `scripts/predict.py`.

---

## Tests

```bash
pytest                 # fast suite (excludes slow tests)
pytest -m slow         # tiny-subset overfit check (~25 s on MPS)
```

| File                      | Covers                                                                 |
|---------------------------|------------------------------------------------------------------------|
| `test_dataset.py`         | shapes/dtypes/ranges, DataLoader batch contract, each validation error, split disjointness |
| `test_model.py`           | forward shape, `in_channels`, model summary, MPS/CPU device            |
| `test_metrics.py`         | hand-computed IoU, Dice, precision, recall, confusion matrix           |
| `test_losses.py`          | CE / Dice / CE+Dice, class-weight validation                          |
| `test_config.py`          | YAML defaults, overrides, unknown-key and missing-file errors          |
| `test_inference.py`       | single-image prediction, corrupt image/checkpoint errors, latency      |
| `test_visualization.py`   | fixed palette, PNG outputs, training curves                            |
| `test_smoke_train.py`     | real `train()` for one epoch: finite loss, checkpoints, reload, resume |
| `test_overfit.py` (slow)  | pipeline can memorise 4 real tiles (pixel accuracy > 0.9)              |

Tests that read real tiles need `data/raw/paris/` in place.

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Unknown keys are rejected with a clear error, so typos cannot silently fall back to defaults.

```yaml
seed: 42
data:      {image_size: 256, num_classes: 5, batch_size: 4, num_workers: 0,
            train_manifest: splits/train.json, val_manifest: splits/val.json,
            test_manifest: splits/test.json, classes_path: classes.json}
model:     {name: unet, in_channels: 3, num_classes: 5, base_channels: 16}
training:  {epochs: 80, learning_rate: 0.001, weight_decay: 0.0001,
            optimizer: adamw,                  # adam | adamw
            scheduler: reduce_lr_on_plateau,   # reduce_lr_on_plateau | cosine_annealing | none
            scheduler_patience: 5, scheduler_factor: 0.5}
loss:      {type: cross_entropy,               # cross_entropy | ce_dice
            class_weights: null}               # e.g. [1, 3, 1, 1, 1] in class order
augmentation: {enabled: true, horizontal_flip: true, vertical_flip: true,
               rotation_90: true, brightness_contrast: true}
device:    {prefer_mps: true}
paths:     {checkpoints: checkpoints, outputs: outputs, runs: runs}
```

Flips and 90° rotations are applied identically to image and mask. Brightness and contrast changes touch the image only. Validation and test data are never augmented.

---

## Reproducibility

`set_seeds` seeds Python, NumPy and PyTorch, and the training DataLoader uses a seeded generator. Runs are repeatable on a given machine and backend, but exact bit-for-bit results can differ between MPS, CUDA and CPU because their kernels are not all deterministic. Each run's `config.json` records the full config, device, torch version, sample counts and parameter count.

---

## Project Structure

```
mapvision/
├── README.md
├── requirements.txt
├── pytest.ini
├── classes.json                class → label color
├── configs/default.yaml
├── docs/DATASET.md             upstream dataset card
├── splits/                     train/val/test manifests + protocol
├── src/mapvision/
│   ├── config.py               YAML → dataclasses, seeds, device selection
│   ├── dataset.py              MapDataset, validate_record, label decoding
│   ├── transforms.py           segmentation-safe augmentations
│   ├── model.py                UNet, build_model, model_summary
│   ├── losses.py               CE, Dice, CE+Dice, class weights
│   ├── metrics.py              confusion-matrix based metrics
│   ├── train.py                training loop, checkpoints, resume
│   ├── evaluate.py             single-pass loss + metrics
│   ├── inference.py            checkpoint loading, prediction, latency
│   ├── visualization.py        palette, overlays, grids, curves
│   └── utils.py                JSON I/O, metric report formatting
├── scripts/                    train / evaluate / predict / validate_dataset / report
├── tests/
├── data/raw/                   (gitignored) dataset
├── checkpoints/                (gitignored) best.pt, latest.pt
├── runs/                       (gitignored) per-run config, history, curves
└── outputs/                    (gitignored) metrics, predictions, plots
```

---

## Future Work

- Tiled inference at native resolution. Tiles are currently resized from 1000×1000 to 256×256, which thins roads to a few pixels, so training on native-resolution crops with sliding-window inference is the most direct next step.
- Pretrained encoders (ResNet / EfficientNet backbones)
- DeepLabV3+ and SegFormer architectures
- Class-weighted or focal loss for the rare water class
- Larger training corpus: add the 305-patch World partition
- Polygon and vector extraction (GeoJSON layers from masks)
- OCR integration for street names and labels
- Uncertainty estimation (MC dropout or ensembles)

---

## Citation and License

If you use the dataset, please cite:

```bibtex
@misc{petitpierre_historical_2021,
  author       = {Petitpierre, Rémi},
  title        = {Historical City Maps Semantic Segmentation Dataset},
  year         = {2021},
  howpublished = {\url{https://zenodo.org/record/5513639}},
  doi          = {10.5281/zenodo.5513639}
}
```

Code: [MIT](LICENSE). Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en), so the trained weights and the sample tiles are for non-commercial use.
