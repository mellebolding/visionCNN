#!/usr/bin/env python3
"""
Training script for VisionCNN models with Distributed Data Parallel (DDP) support.

Single GPU:
    python scripts/train.py --config configs/convnextv2_tiny.yaml

Multi-GPU (local, e.g., 3x A6000):
    torchrun --nproc_per_node=3 scripts/train.py --config configs/convnextv2_tiny.yaml

Multi-GPU (SLURM, e.g., 4x A100):
    See scripts/launch_snellius.sh for SLURM submission

Resume training:
    torchrun --nproc_per_node=3 scripts/train.py --config configs/convnextv2_tiny.yaml --resume logs/convnextv2_tiny_cifar10/last.pt
"""
import argparse
import csv
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml
import torch
import torch.nn as nn
#from torch.cuda.amp import GradScaler, autocast
try:
    from torch.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AMP_NEW_API = False
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from typing import cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.models.norms import adaptive_gradient_clip
from src.datasets.build import build_dataloader, build_dataloader_with_backend
from src.datasets.gpu_transforms import build_gpu_transforms
from src.utils.seed import set_seed
from src.utils.distributed import DistributedManager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(log_dir: str, experiment_name: str, rank: int = 0) -> logging.Logger:
    """Setup logging to file and console (only rank 0 logs to file)."""
    run_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    # Console handler for all ranks (but we'll control output manually)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler only for rank 0
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
        # Handle case where min_lr might be a string
        if isinstance(min_lr, str):
            min_lr = float(min_lr)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
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


def collect_gradient_norms(model):
    """Collect L2 gradient norms for all parameter groups (conv, norm, linear)."""
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.data.norm(2).item()
    return grad_norms


def save_gradient_stats(grad_log, filepath):
    """Save gradient stats to CSV."""
    if not grad_log:
        return
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # Get all layer names from first entry
        layer_names = sorted(grad_log[0]["grads"].keys())
        writer.writerow(["epoch", "step"] + layer_names)
        for entry in grad_log:
            row = [entry["epoch"], entry["step"]]
            row.extend(entry["grads"].get(n, 0.0) for n in layer_names)
            writer.writerow(row)


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_channels_last: bool,
    epoch: int,
    dist_manager: DistributedManager,
    scaler: GradScaler | None = None,
    use_amp: bool = True,
    gpu_transforms: nn.Module | None = None,
    disable_progress_bar: bool = False,
    grad_log: list | None = None,
    grad_log_freq: int = 50,
    use_agc: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Only show progress bar on main process (unless disabled for speed)
    train_iter = train_loader
    if dist_manager.is_main_process and not disable_progress_bar:
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    
    for images, labels in train_iter:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Apply GPU transforms if enabled (normalization + augmentations)
        if gpu_transforms is not None:
            images = gpu_transforms(images)

        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)

        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            if 'TORCH_AMP_NEW_API' in globals() and TORCH_AMP_NEW_API:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if use_agc:
                scaler.unscale_(optimizer)
                adaptive_gradient_clip(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if use_agc:
                adaptive_gradient_clip(model)
            optimizer.step()
        
        # Log gradient norms periodically
        if grad_log is not None and dist_manager.is_main_process:
            step = total // images.size(0)
            if step % grad_log_freq == 0:
                grad_norms = collect_gradient_norms(model)
                grad_log.append({"epoch": epoch + 1, "step": step, "grads": grad_norms})

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
        metrics_tensor = torch.tensor(
            [total_loss, correct, total],
            device=device
        )
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
    use_amp: bool = True,
    gpu_transforms: nn.Module | None = None,
    disable_progress_bar: bool = False
) -> dict:
    """Validate the model."""
    # Use unwrapped model to avoid DDP buffer sync issues when validating on single rank
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Only show progress bar on main process (unless disabled for speed)
    val_iter = val_loader
    if dist_manager.is_main_process and not disable_progress_bar:
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    
    for images, labels in val_iter:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # Apply GPU transforms if enabled (normalization only for validation)
        if gpu_transforms is not None:
            images = gpu_transforms(images)

        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        if use_amp:
            if 'TORCH_AMP_NEW_API' in globals() and TORCH_AMP_NEW_API:
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

    # Restore model to training mode
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
    
    # Get the actual model (unwrap DDP if necessary)
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


def load_checkpoint(
    filepath: str, 
    model: nn.Module, 
    optimizer=None, 
    scheduler=None, 
    scaler=None,
    map_location="cpu"
):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location=map_location, weights_only=False)
    
    # Handle DDP wrapper
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
    import socket
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Apply --set overrides (e.g. --set model.name=vgg_small model.norm_layer=groupnorm)
    for override in getattr(args, 'set', []):
        if '=' not in override:
            raise ValueError(f"Invalid --set format: '{override}'. Expected KEY=VALUE")
        key, value = override.split('=', 1)
        keys = key.split('.')
        # Auto-convert numeric values
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
        # Set nested key
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    # Auto-detect host-specific settings
    host = socket.gethostname().lower()

    if cfg.get("data", {}).get("dataset", "").lower() in ("imagenet", "imagenet100"):
        imagenet_root = None

        if "guppy" in host:
            imagenet_root = "/export/scratch1/home/melle/datasets/imagenet"
        elif "snellius" in host:
            imagenet_root = "/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder"

        # Replace if we detected a known host AND root is placeholder or empty
        if imagenet_root is not None:
            current_root = cfg["data"].get("root", "")
            if current_root in (None, "<IMAGENET_ROOT>", "", "None"):
                cfg["data"]["root"] = imagenet_root
                print(f"[INFO] Auto-detected ImageNet root: {imagenet_root}")
            else:
                print(f"[INFO] Using ImageNet root from config: {current_root}")

    # Auto-enable GPU transforms on guppy (limited CPU cores)
    if "guppy" in host:
        if "gpu_transforms" not in cfg.get("data", {}):
            cfg.setdefault("data", {})["gpu_transforms"] = True
            print(f"[INFO] Auto-enabled GPU transforms (guppy has limited CPU cores)")
    
    # Setup - everything goes in logs/{experiment_name}/
    experiment_name = cfg.get("experiment_name", Path(args.config).stem)
    # Allow command line override for log_dir (useful for different clusters)
    base_log_dir = args.log_dir if args.log_dir else cfg.get("logging", {}).get("log_dir", "logs")
    run_dir = os.path.join(base_log_dir, experiment_name)
    
    # Only main process creates directories
    if dist_manager.is_main_process:
        os.makedirs(run_dir, exist_ok=True)
    
    # Synchronize before logging setup
    dist_manager.barrier()
    
    logger = setup_logging(base_log_dir, experiment_name, dist_manager.rank)
    
    if dist_manager.is_main_process:
        logger.info(f"Distributed: {dist_manager}")
        logger.info(f"Config: {cfg}")
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
            dir=run_dir,
        )
        logger.info("Weights & Biases logging enabled")

    # Set seed (with rank offset for diversity)
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed + dist_manager.rank)
    if dist_manager.is_main_process:
        logger.info(f"Set random seed: {seed} (rank offset applied)")
    
    # Device is managed by dist_manager
    device = dist_manager.device
    if dist_manager.is_main_process:
        logger.info(f"Using device: {device}")
        if dist_manager.is_distributed:
            logger.info(f"World size: {dist_manager.world_size}")
    
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
    
    # Compile model for faster training (PyTorch 2.0+)
    # Note: torch.compile should be done before DDP wrapping
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
    #scaler = GradScaler() if use_amp else None
    if use_amp:
        if TORCH_AMP_NEW_API:
            scaler = GradScaler('cuda')
        else:
            scaler = GradScaler()
    else:
        scaler = None
    if dist_manager.is_main_process:
        logger.info(f"Mixed precision training: {use_amp}")
    
    # Build dataloaders with distributed samplers
    # Use backend-aware loader (supports pytorch, dali, cached)
    backend = cfg.get("data", {}).get("backend", "pytorch").lower()
    train_loader, val_loader, train_sampler, val_sampler = build_dataloader_with_backend(cfg, dist_manager)

    # Check if using DALI (for special handling)
    use_dali = backend in ("dali", "dali_cached") and hasattr(train_loader, 'reset')
    if use_dali and dist_manager.is_main_process:
        logger.info("Using DALI GPU-accelerated data loading")

    # Build GPU transforms if enabled (but skip if using DALI - it handles transforms internally)
    use_gpu_transforms = cfg.get("data", {}).get("gpu_transforms", False)
    if use_gpu_transforms and not use_dali:
        gpu_train_transforms = build_gpu_transforms(cfg, is_train=True).to(device)
        gpu_val_transforms = build_gpu_transforms(cfg, is_train=False).to(device)
        if dist_manager.is_main_process:
            logger.info("Using GPU-accelerated transforms to reduce CPU bottleneck")
    else:
        gpu_train_transforms = None
        gpu_val_transforms = None
        if use_gpu_transforms and use_dali and dist_manager.is_main_process:
            logger.info("Skipping gpu_transforms (DALI handles augmentations internally)")

    train_fraction = cfg.get("data", {}).get("train_fraction", 1.0)
    fraction_str = f" ({train_fraction*100:.0f}% of full dataset)" if train_fraction < 1.0 else ""

    if dist_manager.is_main_process:
        total_train = len(train_loader.dataset)  # type: ignore
        total_val = len(val_loader.dataset)  # type: ignore
        logger.info(f"Train samples: {total_train}{fraction_str}, Val samples: {total_val}")
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
            args.resume, model, optimizer, scheduler, scaler,
            map_location=device
        )
        
        if loaded_history is not None:
            history = loaded_history
        
        if dist_manager.is_main_process:
            logger.info(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")
    
    # Training loop
    epochs = cfg["training"]["epochs"]
    if dist_manager.is_main_process:
        logger.info(f"Starting training for {epochs} epochs...")
    
    # Speed optimization settings
    val_frequency = cfg.get("training", {}).get("val_frequency", 1)
    save_frequency = cfg.get("training", {}).get("save_frequency", 1)
    disable_progress_bar = cfg.get("training", {}).get("disable_progress_bar", False)

    # Gradient flow logging
    log_gradients = cfg.get("training", {}).get("log_gradients", False)
    grad_log_freq = cfg.get("training", {}).get("grad_log_freq", 50)
    grad_log = [] if log_gradients and dist_manager.is_main_process else None
    if log_gradients and dist_manager.is_main_process:
        logger.info(f"Gradient flow logging enabled (every {grad_log_freq} steps)")

    # Adaptive Gradient Clipping (auto-enabled for nonorm_ws)
    norm_layer_name = cfg.get("model", {}).get("norm_layer", "batchnorm").lower()
    use_agc = cfg.get("training", {}).get("use_agc", norm_layer_name == "nonorm_ws")
    if use_agc and dist_manager.is_main_process:
        logger.info("Adaptive Gradient Clipping (AGC) enabled")

    for epoch in range(start_epoch, epochs):
        start_time = time.time()

        # Set epoch for distributed sampler (important for shuffling!)
        # DALI handles this internally with shuffle_after_epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        elif use_dali and epoch > start_epoch:
            # DALI auto-resets, but we call reset explicitly for clarity
            train_loader.reset()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_channels_last, epoch, dist_manager, scaler, use_amp,
            gpu_transforms=gpu_train_transforms,
            disable_progress_bar=disable_progress_bar,
            grad_log=grad_log,
            grad_log_freq=grad_log_freq,
            use_agc=use_agc,
        )

        # Validation - only run every N epochs + last epoch
        should_validate = (epoch + 1) % val_frequency == 0 or (epoch + 1) == epochs
        if should_validate and dist_manager.is_main_process:
            val_metrics = validate(
                model, val_loader, criterion, device,
                epoch, use_channels_last, dist_manager, use_amp,
                gpu_transforms=gpu_val_transforms,
                disable_progress_bar=disable_progress_bar
            )
        else:
            # Skip validation - reuse previous metrics for logging
            val_metrics = {"loss": history["val_loss"][-1] if history["val_loss"] else 0.0,
                          "accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0}

        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Only main process logs and saves
        if dist_manager.is_main_process:
            # Track history
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

            # Save history to CSV only when we validate (reduces I/O)
            if should_validate:
                save_history_csv(history, os.path.join(run_dir, "history.csv"))

            # Logging
            val_str = f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%" if should_validate else "(validation skipped)"
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                f"{val_str} | "
                f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
            )

            # Save checkpoints periodically + always save best
            should_save = (epoch + 1) % save_frequency == 0 or (epoch + 1) == epochs

            # Check for best accuracy only when we validate
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

            # Save last checkpoint periodically (not every epoch)
            if should_save:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_acc, cfg,
                    os.path.join(run_dir, "last.pt"), scaler, history,
                    is_distributed=dist_manager.is_distributed
                )
    
    if dist_manager.is_main_process:
        logger.info(f"Training complete! Best accuracy: {best_acc:.2f}%")
        logger.info(f"Outputs saved to: {run_dir}")

        # Save gradient flow log
        if grad_log:
            grad_path = os.path.join(run_dir, "gradient_stats.csv")
            save_gradient_stats(grad_log, grad_path)
            logger.info(f"Gradient stats saved to: {grad_path}")

    if use_wandb:
        wandb.finish()

    # Cleanup distributed
    dist_manager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vision model (supports DDP)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory (default: from config or 'logs')")
    parser.add_argument("--local_rank", type=int, default=None, help="Local rank (set by torchrun)")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set model.name=vgg_small model.norm_layer=groupnorm")
    args = parser.parse_args()
    
    main(args)
