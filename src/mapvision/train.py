"""Core training loop for MapVision.

Handles one full training run: data loading, model construction,
optimizer setup, epoch loop with validation, checkpointing, resuming,
and history logging.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from mapvision.config import Config, get_device, set_seeds
from mapvision.dataset import CLASS_NAMES, MapDataset
from mapvision.evaluate import evaluate
from mapvision.losses import build_loss, class_weight_tensor
from mapvision.model import build_model, model_summary
from mapvision.transforms import TrainTransform, ValTransform
from mapvision.utils import load_json, save_json


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_name = cfg.training.optimizer.lower()
    lr = cfg.training.learning_rate
    wd = cfg.training.weight_decay
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt_name!r}. Choose 'adam' or 'adamw'.")


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
    raise ValueError(
        f"Unknown scheduler: {sched!r}. "
        "Choose 'reduce_lr_on_plateau', 'cosine_annealing' or 'none'."
    )


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train (augmented, shuffled) and validation (plain) loaders."""
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
    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one optimisation pass and return the sample-weighted mean loss."""
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    epoch: int,
    val_miou: float,
    best_val_miou: float,
    config: Dict,
) -> None:
    """Write a resumable checkpoint with weights, optimizer/scheduler state and config."""
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "val_miou": val_miou,
            "best_val_miou": best_val_miou,
            "config": config,
            "class_names": CLASS_NAMES,
        },
        path,
    )


def _load_resume_state(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
) -> Tuple[int, float]:
    """Restore training state in place; return (start_epoch, best_val_miou)."""
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing = [k for k in ("model_state", "optimizer_state", "epoch") if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint {path} cannot be resumed; missing keys: {missing}")

    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    best = ckpt.get("best_val_miou", ckpt.get("val_miou", -1.0))
    print(f"Resumed from {path} (epoch {ckpt['epoch']}, best val mIoU {best:.4f})")
    return ckpt["epoch"] + 1, best


def train(
    cfg: Config,
    output_dir: Optional[str] = None,
    resume: Optional[str] = None,
) -> Path:
    """Run training and return the output directory path.

    Args:
        cfg:        Configuration object.
        output_dir: Where to write config.json and history.json. Defaults to
                    ``cfg.paths.runs/run_<timestamp>``.
        resume:     Optional checkpoint path to continue training from.

    Returns:
        Path to the output directory. Checkpoints (best.pt, latest.pt) are
        written to ``cfg.paths.checkpoints``.
    """
    set_seeds(cfg.seed)
    device = get_device(cfg.device.prefer_mps)

    out = Path(output_dir) if output_dir else Path(cfg.paths.runs) / f"run_{int(time.time())}"
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cfg.paths.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg)
    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val samples   : {len(val_loader.dataset)}")

    model = build_model(
        name=cfg.model.name,
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        base_channels=cfg.model.base_channels,
    ).to(device)
    param_counts = model_summary(
        model, (cfg.model.in_channels, cfg.data.image_size, cfg.data.image_size)
    )

    weights = class_weight_tensor(cfg.loss.class_weights, cfg.model.num_classes, device)
    criterion = build_loss(cfg.loss.type, weight=weights)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    config_record: Dict = asdict(cfg)
    save_json(
        {
            **config_record,
            "class_names": CLASS_NAMES,
            "device": str(device),
            "torch_version": torch.__version__,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            **param_counts,
        },
        out / "config.json",
    )

    start_epoch, best_miou = 1, -1.0
    history: List[Dict] = []
    if resume is not None:
        start_epoch, best_miou = _load_resume_state(resume, model, optimizer, scheduler)
        if (out / "history.json").exists():
            history = [r for r in load_json(out / "history.json") if r["epoch"] < start_epoch]

    epochs = cfg.training.epochs
    print(f"\nTraining on {device} for epochs {start_epoch}–{epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = evaluate(
            model, val_loader, device, cfg.data.num_classes, criterion=criterion
        )
        miou = val_metrics["mean_iou"]
        elapsed = time.perf_counter() - t0

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_miou": miou,
            "val_dice": val_metrics["mean_dice"],
            "val_pixel_acc": val_metrics["pixel_accuracy"],
            "val_iou": val_metrics["per_class_iou"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        })
        save_json(history, out / "history.json")

        is_best = miou > best_miou
        if is_best:
            best_miou = miou

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(miou)
            else:
                scheduler.step()

        ckpt_args = (model, optimizer, scheduler, epoch, miou, best_miou, config_record)
        save_checkpoint(ckpt_dir / "latest.pt", *ckpt_args)
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", *ckpt_args)

        print(f"Epoch {epoch:02d}/{epochs:02d}  ({elapsed:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val mIoU:   {miou:.4f}")
        print(f"  Val Dice:   {val_metrics['mean_dice']:.4f}")
        if is_best:
            print("  Saved new best checkpoint.")
        print(flush=True)

    print(f"Best val mIoU : {best_miou:.4f}")
    print(f"Checkpoints   : {ckpt_dir}/")
    print(f"History       : {out}/history.json")
    return out
