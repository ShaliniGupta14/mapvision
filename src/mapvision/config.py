"""Configuration management for MapVision.

Loads YAML configs into typed dataclasses. All hyperparameters should
flow through this module rather than being hardcoded in scripts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    image_size: int = 256
    num_classes: int = 5
    batch_size: int = 4
    num_workers: int = 0
    train_manifest: str = "splits/train.json"
    val_manifest: str = "splits/val.json"
    test_manifest: str = "splits/test.json"
    classes_path: str = "classes.json"


@dataclass
class ModelConfig:
    name: str = "unet"
    in_channels: int = 3
    num_classes: int = 5
    base_channels: int = 16


@dataclass
class TrainingConfig:
    epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "reduce_lr_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5


@dataclass
class LossConfig:
    type: str = "cross_entropy"


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_90: bool = True
    brightness_contrast: bool = True


@dataclass
class DeviceConfig:
    prefer_mps: bool = True


@dataclass
class PathsConfig:
    checkpoints: str = "checkpoints"
    outputs: str = "outputs"
    runs: str = "runs"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Optional[str] = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    defaults = {
        "seed": 42,
        "data": {
            "image_size": 256,
            "num_classes": 5,
            "batch_size": 4,
            "num_workers": 0,
            "train_manifest": "splits/train.json",
            "val_manifest": "splits/val.json",
            "test_manifest": "splits/test.json",
            "classes_path": "classes.json",
        },
        "model": {
            "name": "unet",
            "in_channels": 3,
            "num_classes": 5,
            "base_channels": 16,
        },
        "training": {
            "epochs": 25,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": "reduce_lr_on_plateau",
            "scheduler_patience": 5,
            "scheduler_factor": 0.5,
        },
        "loss": {"type": "cross_entropy"},
        "augmentation": {
            "enabled": True,
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_90": True,
            "brightness_contrast": True,
        },
        "device": {"prefer_mps": True},
        "paths": {
            "checkpoints": "checkpoints",
            "outputs": "outputs",
            "runs": "runs",
        },
    }

    if path is not None:
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        raw = _merge(defaults, loaded)
    else:
        raw = defaults

    return Config(
        seed=raw["seed"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        loss=LossConfig(**raw["loss"]),
        augmentation=AugmentationConfig(**raw["augmentation"]),
        device=DeviceConfig(**raw["device"]),
        paths=PathsConfig(**raw["paths"]),
    )


def set_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility.

    Note: exact bit-for-bit reproducibility may differ across MPS/CUDA/CPU
    backends even with the same seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_mps: bool = True) -> torch.device:
    """Select compute device with MPS → CUDA → CPU priority."""
    if prefer_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
