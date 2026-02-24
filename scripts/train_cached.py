#!/usr/bin/env python3
"""
Training script with RAM-cached ImageNet for maximum GPU utilization.

This script loads the entire ImageNet dataset into RAM at startup,
eliminating disk I/O and CPU JPEG decoding during training.

Requirements:
- ~180 GB free RAM for ImageNet at 224x224
- GPU transforms enabled (augmentations run on GPU)

Usage:
    Single GPU:  python scripts/train_cached.py --config configs/resnet18_imagenet_cached.yaml
    Multi-GPU:   torchrun --nproc_per_node=3 scripts/train_cached.py --config configs/resnet18_imagenet_cached.yaml
"""
import argparse
import csv
import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = False

from tqdm import tqdm
from typing import cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.cached_imagenet import build_cached_dataloader
from src.utils.seed import set_seed
from src.utils.distributed import DistributedManager
from src.utils.machine import resolve_machine_config

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GPUAugmentations(nn.Module):
    """GPU-based augmentations for cached uint8 data.

    Handles:
    - uint8 [0,255] -> float16/32 [0,1] conversion
    - Random horizontal flip (training only)
    - Color jitter (optional, training only)
    - Normalization
    - Random erasing (optional, training only)
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        random_flip: bool = True,
        random_erasing_p: float = 0.0,
        color_jitter: tuple = None,  # (brightness, contrast, saturation, hue)
    ):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        self.random_flip = random_flip
        self.random_erasing_p = random_erasing_p
        self.color_jitter = color_jitter  # e.g., (0.4, 0.4, 0.4, 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: uint8 tensor [B, 3, H, W] with values in [0, 255]

        Returns:
            Normalized float tensor [B, 3, H, W]
        """
        # Convert uint8 -> float and scale to [0, 1]
        x = x.to(dtype=self.mean.dtype) / 255.0

        # Random horizontal flip (training only)
        if self.training and self.random_flip:
            # Create random flip mask for batch
            flip_mask = torch.rand(x.size(0), device=x.device) > 0.5
            x[flip_mask] = x[flip_mask].flip(-1)  # Flip along width dimension

        # Color jitter (training only, before normalization)
        if self.training and self.color_jitter is not None:
            x = self._color_jitter(x)

        # Normalize
        x = (x - self.mean) / self.std

        # Random erasing (training only)
        if self.training and self.random_erasing_p > 0:
            x = self._random_erasing(x)

        return x

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply color jitter augmentation on GPU (batch-wise, same transform per image)."""
        b, c, h, w = x.shape
        brightness, contrast, saturation, hue = self.color_jitter

        # Per-image random factors
        if brightness > 0:
            b_factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * brightness
            x = x * b_factor

        if contrast > 0:
            c_factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * contrast
            gray = x.mean(dim=1, keepdim=True)
            x = c_factor * x + (1 - c_factor) * gray

        if saturation > 0:
            s_factor = 1.0 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * saturation
            gray = x.mean(dim=1, keepdim=True)
            x = s_factor * x + (1 - s_factor) * gray

        if hue > 0:
            # Simplified hue shift: rotate RGB channels slightly
            h_factor = (torch.rand(b, device=x.device) * 2 - 1) * hue * 0.5
            cos_h = torch.cos(h_factor * 3.14159).view(b, 1, 1, 1)
            sin_h = torch.sin(h_factor * 3.14159).view(b, 1, 1, 1)
            # Approximate hue rotation in RGB space
            r, g, b_ch = x[:, 0:1], x[:, 1:2], x[:, 2:3]
            gray = (r + g + b_ch) / 3
            x = torch.cat([
                gray + (r - gray) * cos_h + (g - b_ch) * sin_h * 0.5,
                gray + (g - gray) * cos_h + (b_ch - r) * sin_h * 0.5,
                gray + (b_ch - gray) * cos_h + (r - g) * sin_h * 0.5,
            ], dim=1)

        return x.clamp(0, 1)

    def _random_erasing(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random erasing augmentation."""
        batch_size, _, h, w = x.shape

        for i in range(batch_size):
            if torch.rand(1).item() > self.random_erasing_p:
                continue

            area = h * w
            target_area = torch.empty(1).uniform_(0.02, 0.33).item() * area
            aspect_ratio = torch.empty(1).uniform_(0.3, 3.3).item()

            h_erase = int(round((target_area * aspect_ratio) ** 0.5))
            w_erase = int(round((target_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                top = torch.randint(0, h - h_erase + 1, (1,)).item()
                left = torch.randint(0, w - w_erase + 1, (1,)).item()
                x[i, :, top:top+h_erase, left:left+w_erase] = torch.randn(
                    3, h_erase, w_erase, dtype=x.dtype, device=x.device
                )

        return x


def setup_logging(log_dir: str, experiment_name: str, rank: int = 0) -> logging.Logger:
    """Setup logging to file and console."""
    run_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        log_file = os.path.join(run_dir, "train.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_history_csv(history: dict, filepath: str):
    """Save training history to CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
                history["lr"][i] if i < len(history["lr"]) else ""
            ])


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_name = cfg.get("training", {}).get("optimizer", "adamw").lower()
    lr = cfg["training"]["lr"]
    weight_decay = cfg["training"].get("weight_decay", 0.01)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = cfg["training"].get("momentum", 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    """Build learning rate scheduler from config."""
    scheduler_name = cfg.get("training", {}).get("scheduler", "cosine").lower()
    epochs = cfg["training"]["epochs"]

    if scheduler_name == "cosine":
        min_lr = cfg["training"].get("min_lr", 1e-6)
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_name == "step":
        step_size = cfg["training"].get("step_size", 30)
        gamma = cfg["training"].get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "multistep":
        milestones = cfg["training"].get("milestones", [60, 80])
        gamma = cfg["training"].get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_channels_last: bool,
    epoch: int,
    dist_manager: DistributedManager,
    gpu_augmentations: GPUAugmentations,
    scaler: GradScaler | None = None,
    use_amp: bool = True,
    disable_progress_bar: bool = False
) -> dict:
    """Train for one epoch with cached data."""
    model.train()
    gpu_augmentations.train()

    total_loss = 0.0
    correct = 0
    total = 0

    train_iter = train_loader
    if dist_manager.is_main_process and not disable_progress_bar:
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

    for images, labels in train_iter:
        # Move uint8 data to GPU (very fast transfer)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply GPU augmentations (uint8->float, flip, normalize)
        images = gpu_augmentations(images)

        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            if TORCH_AMP_NEW_API:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if dist_manager.is_main_process and hasattr(train_iter, 'set_postfix'):
            train_iter.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%"
            })

    # Aggregate metrics across all processes
    if dist_manager.is_distributed:
        metrics_tensor = torch.tensor([total_loss, correct, total], device=device)
        dist_manager.all_reduce(metrics_tensor)
        total_loss = metrics_tensor[0].item()
        correct = int(metrics_tensor[1].item())
        total = int(metrics_tensor[2].item())

    return {
        "loss": total_loss / total,
        "accuracy": 100. * correct / total
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_channels_last: bool,
    dist_manager: DistributedManager,
    gpu_augmentations: GPUAugmentations,
    use_amp: bool = True,
    disable_progress_bar: bool = False
) -> dict:
    """Validate the model."""
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    gpu_augmentations.eval()  # Disables random augmentations

    total_loss = 0.0
    correct = 0
    total = 0

    val_iter = val_loader
    if dist_manager.is_main_process and not disable_progress_bar:
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)

    for images, labels in val_iter:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply GPU transforms (no augmentation in eval mode)
        images = gpu_augmentations(images)

        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)

        if use_amp:
            if TORCH_AMP_NEW_API:
                with autocast(device_type='cuda'):
                    outputs = eval_model(images)
                    loss = criterion(outputs, labels)
            else:
                with autocast():
                    outputs = eval_model(images)
                    loss = criterion(outputs, labels)
        else:
            outputs = eval_model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    eval_model.train()

    return {
        "loss": total_loss / total,
        "accuracy": 100. * correct / total
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    cfg: dict,
    filepath: str,
    scaler: GradScaler | None = None,
    history: dict | None = None,
    is_distributed: bool = False
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    model_to_save = model.module if is_distributed else model

    checkpoint = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "config": cfg,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    if history is not None:
        checkpoint["history"] = history

    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)

    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint["model"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    history = checkpoint.get("history", None)

    return start_epoch, best_acc, history


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank is not None else 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist_manager = DistributedManager(backend="nccl")
    dist_manager.setup(local_rank=args.local_rank)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Resolve machine-specific settings (paths, worker counts, wandb tags)
    project_root = Path(__file__).parent.parent
    cfg, machine_name = resolve_machine_config(cfg, project_root)

    # Setup
    experiment_name = cfg.get("experiment_name", Path(args.config).stem)
    machine_log_dir = cfg.get("_machine", {}).get("log_dir")
    base_log_dir = args.log_dir if args.log_dir else (machine_log_dir or cfg.get("logging", {}).get("log_dir", "logs"))
    run_dir = os.path.join(base_log_dir, experiment_name)

    if dist_manager.is_main_process:
        os.makedirs(run_dir, exist_ok=True)

    dist_manager.barrier()

    logger = setup_logging(base_log_dir, experiment_name, dist_manager.rank)

    if dist_manager.is_main_process:
        logger.info(f"=== RAM-Cached ImageNet Training ===")
        logger.info(f"Distributed: {dist_manager}")
        logger.info(f"Run directory: {run_dir}")

    # Initialize wandb (rank 0 only)
    wandb_cfg = cfg.get("wandb", {})
    use_wandb = (
        WANDB_AVAILABLE
        and wandb_cfg.get("enabled", False)
        and dist_manager.is_main_process
    )
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "visionCNN"),
            entity=wandb_cfg.get("entity", None),
            name=experiment_name,
            config=cfg,
            tags=wandb_cfg.get("tags", []),
            group=wandb_cfg.get("group", None),
            dir=run_dir,
        )
        wandb.config.update({"machine": machine_name}, allow_val_change=True)
        logger.info("Weights & Biases logging enabled")

    # Set seed
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed + dist_manager.rank)

    device = dist_manager.device
    if dist_manager.is_main_process:
        logger.info(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    if dist_manager.is_main_process:
        logger.info(f"Model: {cfg['model']['name']}, Parameters: {num_params:,}")

    use_channels_last = cfg.get("training", {}).get("channels_last", False)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
        if dist_manager.is_main_process:
            logger.info("Using channels-last (NHWC) memory format")

    # Compile model
    if cfg.get("training", {}).get("compile", False) and hasattr(torch, "compile"):
        if dist_manager.is_main_process:
            logger.info("Compiling model with torch.compile...")
        model = cast(nn.Module, torch.compile(model))

    # Wrap model with DDP
    sync_bn = cfg.get("training", {}).get("sync_bn", False)
    model = dist_manager.wrap_model(model, find_unused_parameters=False, sync_bn=sync_bn)

    # Build optimizer, scheduler, criterion
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("training", {}).get("label_smoothing", 0.0)
    )

    # Mixed precision
    use_amp = cfg.get("training", {}).get("use_amp", True) and device.type == "cuda"
    if use_amp:
        scaler = GradScaler('cuda') if TORCH_AMP_NEW_API else GradScaler()
    else:
        scaler = None
    if dist_manager.is_main_process:
        logger.info(f"Mixed precision training: {use_amp}")

    # Build GPU augmentations
    data_cfg = cfg.get("data", {})
    random_erasing_p = 0.25 if data_cfg.get("random_erasing", False) else 0.0
    color_jitter_cfg = data_cfg.get("color_jitter", None)
    if color_jitter_cfg:
        # Can be a list [b,c,s,h] or a single float for b,c,s with h=0
        if isinstance(color_jitter_cfg, (int, float)):
            color_jitter = (color_jitter_cfg, color_jitter_cfg, color_jitter_cfg, 0.0)
        else:
            color_jitter = tuple(color_jitter_cfg)
    else:
        color_jitter = None
    gpu_augmentations = GPUAugmentations(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        random_flip=True,
        random_erasing_p=random_erasing_p,
        color_jitter=color_jitter,
    ).to(device)

    if dist_manager.is_main_process:
        logger.info("GPU augmentations: random_flip=True, normalize=ImageNet")
        if color_jitter:
            logger.info(f"  color_jitter={color_jitter}")
        if random_erasing_p > 0:
            logger.info(f"  random_erasing_p={random_erasing_p}")

    # Build cached dataloaders (this loads data into RAM)
    if dist_manager.is_main_process:
        logger.info("Loading ImageNet into RAM (this may take a while)...")

    train_loader, val_loader, train_sampler, val_sampler = build_cached_dataloader(cfg, dist_manager)

    if dist_manager.is_main_process:
        total_train = len(train_loader.dataset)  # type: ignore
        total_val = len(val_loader.dataset)  # type: ignore
        logger.info(f"Train samples: {total_train}, Val samples: {total_val}")
        if dist_manager.is_distributed:
            effective_batch = cfg["training"]["batch_size"] * dist_manager.world_size
            logger.info(f"Effective batch size: {effective_batch} ({cfg['training']['batch_size']} x {dist_manager.world_size} GPUs)")

    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    if args.resume:
        if dist_manager.is_main_process:
            logger.info(f"Resuming from: {args.resume}")

        start_epoch, best_acc, loaded_history = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, map_location=device
        )

        if loaded_history is not None:
            history = loaded_history

        if dist_manager.is_main_process:
            logger.info(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")

    # Training loop
    epochs = cfg["training"]["epochs"]
    if dist_manager.is_main_process:
        logger.info(f"Starting training for {epochs} epochs...")

    val_frequency = cfg.get("training", {}).get("val_frequency", 1)
    save_frequency = cfg.get("training", {}).get("save_frequency", 1)
    disable_progress_bar = cfg.get("training", {}).get("disable_progress_bar", False)

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_channels_last, epoch, dist_manager, gpu_augmentations,
            scaler, use_amp, disable_progress_bar
        )

        # Validation
        should_validate = (epoch + 1) % val_frequency == 0 or (epoch + 1) == epochs
        if should_validate and dist_manager.is_main_process:
            val_metrics = validate(
                model, val_loader, criterion, device,
                epoch, use_channels_last, dist_manager, gpu_augmentations,
                use_amp, disable_progress_bar
            )
        else:
            val_metrics = {
                "loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
                "accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0
            }

        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        if dist_manager.is_main_process:
            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["lr"].append(current_lr)

            # Log to wandb
            if use_wandb:
                wandb_metrics = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/acc": train_metrics["accuracy"],
                    "lr": current_lr,
                }
                if should_validate:
                    wandb_metrics["val/loss"] = val_metrics["loss"]
                    wandb_metrics["val/acc"] = val_metrics["accuracy"]
                wandb.log(wandb_metrics)

            if should_validate:
                save_history_csv(history, os.path.join(run_dir, "history.csv"))

            val_str = f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%" if should_validate else "(validation skipped)"
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"{val_str} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            should_save = (epoch + 1) % save_frequency == 0 or (epoch + 1) == epochs

            if should_validate:
                is_best = val_metrics["accuracy"] > best_acc
                if is_best:
                    best_acc = val_metrics["accuracy"]
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, best_acc, cfg,
                        os.path.join(run_dir, "best.pt"), scaler, history,
                        is_distributed=dist_manager.is_distributed
                    )
                    logger.info(f"New best accuracy: {best_acc:.2f}%")
                    if use_wandb:
                        artifact = wandb.Artifact(
                            f"{experiment_name}-best",
                            type="model",
                            metadata={"epoch": epoch + 1, "val_acc": best_acc},
                        )
                        artifact.add_file(os.path.join(run_dir, "best.pt"))
                        wandb.log_artifact(artifact)

            if should_save:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_acc, cfg,
                    os.path.join(run_dir, "last.pt"), scaler, history,
                    is_distributed=dist_manager.is_distributed
                )

    if dist_manager.is_main_process:
        logger.info(f"Training complete! Best accuracy: {best_acc:.2f}%")
        logger.info(f"Outputs saved to: {run_dir}")

    if use_wandb:
        wandb.finish()

    dist_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with RAM-cached ImageNet")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory")
    parser.add_argument("--local_rank", type=int, default=None, help="Local rank (set by torchrun)")
    args = parser.parse_args()

    main(args)
