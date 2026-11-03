# MapVision

**Semantic segmentation of historical map imagery using deep learning.**

MapVision is a production-style machine learning system that converts scanned historical maps into structured pixel-level geographic layers. Given a 256×256 RGB tile from a historical city map, the model predicts one of five semantic classes for every pixel.

---

## Table of Contents

- [Problem](#problem)
- [Classes](#classes)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Dataset](#dataset)
- [Installation](#installation)
- [Dataset Validation](#dataset-validation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Tests](#tests)
- [Results](#results)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Problem

Historical maps contain invaluable spatial information about cities across centuries — street networks, building footprints, water bodies, and land use — but are difficult to process automatically due to:

- Scanning artifacts and paper degradation
- Complex and varied cartographic styles
- Overlapping map elements and text
- Non-standard color schemes
- Noisy boundaries and imprecise annotation

MapVision addresses this with a supervised semantic segmentation model trained on 630+ annotated 1000×1000 pixel patches from the Paris and World historical map corpora, tiled to 256×256 for training.

---

## Classes

| ID | Class         | Color (annotation) | Meaning                                               |
|----|---------------|--------------------|-------------------------------------------------------|
| 0  | `frame`       | Black              | Non-mapping area: title boxes, decorations, margins   |
| 1  | `water`       | Blue               | Rivers, canals, lakes, the sea                        |
| 2  | `blocks`      | Magenta            | City blocks, buildings, walls                         |
| 3  | `non-built`   | Cyan               | Parks, gardens, wasteland, unbuilt urban land         |
| 4  | `road_network`| White              | Streets, roads, railways, bridges                     |

---

## Architecture

MapVision uses a **U-Net** encoder-decoder with skip connections.

```
Input (3×256×256)
        │
   ┌────▼────┐
   │  Enc 1  │ → 16 channels         ─── skip ───┐
   └────┬────┘                                     │
        │ MaxPool                                  │
   ┌────▼────┐                                     │
   │  Enc 2  │ → 32 channels    ─── skip ───┐      │
   └────┬────┘                               │      │
        │ MaxPool                             │      │
   ┌────▼────┐                               │      │
   │  Enc 3  │ → 64 channels  ─── skip ───┐  │      │
   └────┬────┘                             │  │      │
        │ MaxPool                          │  │      │
   ┌────▼────┐                             │  │      │
   │  Bottleneck │ → 128 channels          │  │      │
   └────┬────┘                             │  │      │
        │ Upsample + concat ───────────────┘  │      │
   ┌────▼────┐                                │      │
   │  Dec 3  │ → 64 channels                  │      │
   └────┬────┘                                │      │
        │ Upsample + concat ─────────────────┘       │
   ┌────▼────┐                                        │
   │  Dec 2  │ → 32 channels                          │
   └────┬────┘                                        │
        │ Upsample + concat ────────────────────────┘
   ┌────▼────┐
   │  Dec 1  │ → 16 channels
   └────┬────┘
        │ 1×1 conv
   ┌────▼────┐
   │  Head   │ → 5 channels (logits per class)
   └─────────┘
Output (5×256×256)
```

Each encoder/decoder block uses two 3×3 convolutions with **Batch Normalization** and **ReLU**. Upsampling uses bilinear interpolation (MPS-compatible).

---

## Pipeline

```
Historical Map
      │
      ▼
256×256 RGB Tile
      │
      ▼
  Normalize [0,1]
      │
      ▼
 Data Augmentation
 (train only)
      │
      ▼
    U-Net
      │
      ▼
 5-Class Logits
 (5×256×256)
      │
      ▼
 argmax → Class Mask
      │
      ▼
 Structured Map Layers
```

---

## Dataset

The dataset is the [Historical City Maps Semantic Segmentation Dataset](https://zenodo.org/record/5513639) by Remi Petitpierre, containing 635 annotated 1000×1000 patches from Paris and world city maps.

**Pre-built splits** (in `splits/`):

| Split  | Samples | Notes                                     |
|--------|---------|-------------------------------------------|
| Train  | 259     | Paris + BHVP images, group-stratified     |
| Val    | 41      | Held-out groups from train partition      |
| Test   | 30      | `paris/eval/` set, no overlap with train  |

Splits were constructed using group-based stratification (one map → one group) to prevent data leakage between splits.

---

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd mapvision

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package (editable)
pip install -e src/
```

> **Apple Silicon:** The code automatically uses MPS when available. No extra setup required.

---

## Dataset Validation

```bash
# Validate all three splits
python scripts/validate_dataset.py

# Validate a single manifest
python scripts/validate_dataset.py --manifest splits/train.json
```

---

## Training

```bash
# Train with default config (25 epochs)
python scripts/train.py --config configs/default.yaml

# Quick smoke run (2 epochs)
python scripts/train.py --config configs/default.yaml --epochs 2

# Custom output directory
python scripts/train.py --config configs/default.yaml --output runs/experiment_01
```

Training prints per-epoch metrics and saves checkpoints to `checkpoints/`:

```
Epoch 01/25  |  train 1.2341  |  val 1.1892  |  mIoU 0.4123  |  Dice 0.5234  |  45.2s
Epoch 02/25  |  train 0.9871  |  val 0.9441  |  mIoU 0.5012  |  Dice 0.6103  |  44.8s ← best
```

---

## Evaluation

```bash
# Evaluate best checkpoint on test set
python scripts/evaluate.py --checkpoint checkpoints/best.pt

# Evaluate on validation set
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val
```

Example output:

```
=============================================
  Evaluation Results
=============================================
  Pixel Accuracy : 0.XXXX
  Mean IoU       : 0.XXXX
  Mean Dice      : 0.XXXX

  Per-Class IoU
  ------------------------------
  frame           : 0.XXXX
  water           : 0.XXXX
  blocks          : 0.XXXX
  non-built       : 0.XXXX
  road_network    : 0.XXXX
=============================================
```

Metrics are saved to `outputs/metrics/test_metrics.json`.

---

## Inference

```bash
# Predict on a single image
python scripts/predict.py \
    --image data/raw/paris/images/BHVP_0.png \
    --checkpoint checkpoints/best.pt
```

Outputs saved to `outputs/predictions/`:
- `<stem>_prediction_grid.png` — side-by-side: input | prediction | overlay
- `<stem>_mask.png` — colorized class mask

---

## Tests

```bash
# Run the full test suite
source .venv/bin/activate
cd mapvision
pytest tests/ -v

# Run only model tests (fast, no data required)
pytest tests/test_model.py tests/test_metrics.py -v
```

Test coverage:

| File                   | What it tests                                        |
|------------------------|------------------------------------------------------|
| `test_dataset.py`      | Shapes, dtypes, value ranges, DataLoader batch format|
| `test_model.py`        | Forward pass, output shape, device compatibility     |
| `test_metrics.py`      | Exact IoU/Dice/accuracy for synthetic masks          |
| `test_smoke_train.py`  | Forward, backward, one epoch, checkpoint writing     |

---

## Results

> **Note:** Metrics below will be filled in after training completes.
> Do not treat placeholder values as measured results.

```
MAPVISION RESUME METRICS
------------------------
Dataset size:          330 (Paris partition)
Training samples:      259
Validation samples:    41
Test samples:          30 (paris/eval)
Model parameters:      TBD — requires training
Best validation mIoU:  TBD — requires training
Test mIoU:             TBD — requires evaluation
Test Dice:             TBD — requires evaluation
Pixel accuracy:        TBD — requires evaluation
Inference latency:     TBD — requires measurement
Training device:       Apple MPS (M-series) / CPU fallback
```

### Sample Outputs

*Add visualizations here after running `scripts/predict.py`.*

---

## Configuration

Edit `configs/default.yaml` to change hyperparameters:

```yaml
seed: 42

data:
  image_size: 256
  num_classes: 5
  batch_size: 4
  num_workers: 0

model:
  name: unet
  in_channels: 3
  num_classes: 5
  base_channels: 16

training:
  epochs: 25
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adamw        # adam | adamw

loss:
  type: cross_entropy     # cross_entropy | ce_dice

device:
  prefer_mps: true
```

---

## Project Structure

```
mapvision/
├── README.md
├── requirements.txt
├── .gitignore
├── classes.json              ← class color mapping
│
├── configs/
│   └── default.yaml          ← training hyperparameters
│
├── data/
│   └── raw/paris/
│       ├── images/           ← 300 training images
│       ├── labels/           ← 300 color-coded masks
│       └── eval/             ← 30 test images + masks
│
├── splits/
│   ├── train.json            ← 259 training samples
│   ├── val.json              ← 41 validation samples
│   ├── test.json             ← 30 test samples (eval)
│   └── protocol.json         ← split methodology
│
├── src/mapvision/
│   ├── __init__.py
│   ├── config.py             ← YAML loading, device selection, seeds
│   ├── dataset.py            ← MapDataset with transforms support
│   ├── transforms.py         ← Segmentation-safe augmentations
│   ├── model.py              ← UNet + build_model factory
│   ├── losses.py             ← CE, Dice, CE+Dice
│   ├── metrics.py            ← IoU, Dice, pixel acc, confusion matrix
│   ├── train.py              ← Training loop
│   ├── evaluate.py           ← Evaluation logic
│   ├── inference.py          ← Checkpoint loading + single-image predict
│   ├── visualization.py      ← Color palette, overlay, grid plots
│   └── utils.py              ← JSON I/O, metric formatting
│
├── scripts/
│   ├── train.py              ← python scripts/train.py --config ...
│   ├── evaluate.py           ← python scripts/evaluate.py --checkpoint ...
│   ├── predict.py            ← python scripts/predict.py --image ...
│   └── validate_dataset.py   ← python scripts/validate_dataset.py
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_smoke_train.py
│
├── checkpoints/
│   ├── best.pt               ← Best validation mIoU checkpoint
│   └── latest.pt             ← Most recent epoch checkpoint
│
└── outputs/
    ├── predictions/           ← Inference visualizations
    └── metrics/               ← JSON metric reports
```

---

## Future Work

- **Pretrained encoders**: ResNet / EfficientNet backbone via `torchvision`
- **DeepLabV3+ / SegFormer**: More powerful architectures
- **Class-weighted loss**: Handle class imbalance (frame dominates)
- **Focal loss**: Focus training on hard pixels
- **Larger training corpus**: World dataset (305 patches not yet included)
- **Tiled full-map inference**: Slide-window prediction on full 1000×1000 maps
- **Polygon / vector extraction**: Convert pixel masks to GeoJSON
- **OCR integration**: Detect and transcribe map annotations
- **Uncertainty estimation**: Monte Carlo dropout for confidence maps

---

## Citation

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

---

## License

Dataset: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)  
Code: MIT
