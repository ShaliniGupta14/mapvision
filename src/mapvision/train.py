"""Core training loop for MapVision.

Handles one full training run: data loading, model construction,
optimizer setup, epoch loop with validation, checkpointing, and
history logging.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from mapvision.config import Config, get_device, set_seeds
from mapvision.dataset import CLASS_NAMES, MapDataset
from mapvision.evaluate import evaluate
from mapvision.losses import build_loss, class_weight_tensor
from mapvision.model import UNet, build_model, model_summary
from mapvision.transforms import TrainTransform, ValTransform
from mapvision.utils import save_json


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name!r}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> Optional[object]:
    """Build LR scheduler from config."""
    sched = cfg.training.scheduler.lower()
    if sched == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg.training.scheduler_patience,
            factor=cfg.training.scheduler_factor,
        )
    elif sched == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    elif sched == "none":
        return None
    raise ValueError(f"Unknown scheduler: {sched!r}")


def train(cfg: Config, output_dir: Optional[str] = None) -> Path:
    """Run training and return the output directory path.

    Args:
        cfg:        Configuration object.
        output_dir: Where to write checkpoints and history. Defaults to
                    ``cfg.paths.runs/run_<timestamp>``.

    Returns:
        Path to the output directory containing best.pt and history.json.
    """
    set_seeds(cfg.seed)
    device = get_device(cfg.device.prefer_mps)

    if output_dir is None:
        run_id = f"run_{int(time.time())}"
        out = Path(cfg.paths.runs) / run_id
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────
    aug_cfg = cfg.augmentation
    train_transform = TrainTransform(
        horizontal_flip=aug_cfg.horizontal_flip,
        vertical_flip=aug_cfg.vertical_flip,
        rotation_90=aug_cfg.rotation_90,
        brightness_contrast=aug_cfg.brightness_contrast,
    ) if aug_cfg.enabled else ValTransform()

    train_set = MapDataset(
        cfg.data.train_manifest,
        size=cfg.data.image_size,
        classes_path=cfg.data.classes_path,
        transform=train_transform,
    )
    val_set = MapDataset(
        cfg.data.val_manifest,
        size=cfg.data.image_size,
        classes_path=cfg.data.classes_path,
        transform=ValTransform(),
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    print(f"Train samples : {len(train_set)}")
    print(f"Val samples   : {len(val_set)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        name=cfg.model.name,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
    ).to(device)
    model_summary(model)

    weights = class_weight_tensor(cfg.loss.class_weights, cfg.model.num_classes, device)
    criterion = build_loss(cfg.loss.type, weight=weights)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── Config dump ───────────────────────────────────────────────────────
    config_record: Dict = {
        "seed": cfg.seed,
        "image_size": cfg.data.image_size,
        "num_classes": cfg.data.num_classes,
        "base_channels": cfg.model.base_channels,
        "classes": CLASS_NAMES,
        "epochs": cfg.training.epochs,
        "batch_size": cfg.data.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "optimizer": cfg.training.optimizer,
        "loss": cfg.loss.type,
        "device": str(device),
        "torch_version": torch.__version__,
        "model": {"name": cfg.model.name, "num_classes": cfg.model.num_classes, "base_channels": cfg.model.base_channels},
    }
    save_json(config_record, out / "config.json")

    history = []
    best_miou = -1.0
    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining on {device} for {cfg.training.epochs} epochs\n")

    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = evaluate(
            model, val_loader, device, cfg.data.num_classes, criterion=criterion
        )
        val_loss = val_metrics["loss"]
        miou = val_metrics["mean_iou"]
        dice = val_metrics["mean_dice"]

        elapsed = time.perf_counter() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou": miou,
            "val_dice": dice,
            "val_pixel_acc": val_metrics["pixel_accuracy"],
            "val_iou": val_metrics["per_class_iou"],
            "seconds": elapsed,
        }
        history.append(row)
        save_json(history, out / "history.json")

        # ── Checkpoint ────────────────────────────────────────────────────
        def _make_ckpt() -> Dict:
            return {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_miou": miou,
                "config": config_record,
                "class_names": CLASS_NAMES,
            }

        torch.save(_make_ckpt(), ckpt_dir / "latest.pt")

        new_best = ""
        if miou > best_miou:
            best_miou = miou
            torch.save(_make_ckpt(), ckpt_dir / "best.pt")
            new_best = " ← best"

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(miou)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch:02d}/{cfg.training.epochs:02d}  |  "
            f"train {train_loss:.4f}  |  "
            f"val {val_loss:.4f}  |  "
            f"mIoU {miou:.4f}  |  "
            f"Dice {dice:.4f}  |  "
            f"{elapsed:.1f}s{new_best}",
            flush=True,
        )

    print(f"\nBest val mIoU : {best_miou:.4f}")
    print(f"Checkpoints   : {ckpt_dir}/")
    print(f"History       : {out}/history.json")
    return out
