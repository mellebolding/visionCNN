#!/usr/bin/env python3
"""
Training script for VisionCNN models.

Usage:
    python scripts/train.py --config configs/convnextv2_tiny.yaml
    python scripts/train.py --config configs/convnextv2_tiny.yaml --resume logs/convnextv2_tiny_cifar10/last.pt
"""
import argparse
import csv
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
from typing import cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.build import build_dataloader
from src.utils.seed import set_seed


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    run_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    log_file = os.path.join(run_dir, "train.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


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


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: GradScaler | None = None,
    use_amp: bool = True
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            with autocast('cuda'):
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
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.2f}%"
        })
    
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
    use_amp: bool = True
) -> dict:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
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
    history: dict | None = None
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
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


def load_checkpoint(filepath: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_acc = checkpoint.get("best_acc", 0.0)
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    
    return start_epoch, best_acc


def main(args):
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    # Setup - everything goes in logs/{experiment_name}/
    experiment_name = cfg.get("experiment_name", Path(args.config).stem)
    base_log_dir = cfg.get("logging", {}).get("log_dir", "logs")
    run_dir = os.path.join(base_log_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    
    logger = setup_logging(base_log_dir, experiment_name)
    logger.info(f"Config: {cfg}")
    logger.info(f"Run directory: {run_dir}")
    
    # Set seed
    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"Set random seed: {seed}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build model
    model = build_model(cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {cfg['model']['name']}, Parameters: {num_params:,}")
    
    # Compile model for faster training (PyTorch 2.0+)
    if cfg.get("training", {}).get("compile", False) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = cast(nn.Module, torch.compile(model))
    
    # Build optimizer, scheduler, criterion
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("training", {}).get("label_smoothing", 0.0)
    )
    
    # Mixed precision
    use_amp = cfg.get("training", {}).get("use_amp", True) and device.type == "cuda"
    scaler = GradScaler('cuda') if use_amp else None
    logger.info(f"Mixed precision training: {use_amp}")
    
    # Build dataloaders
    train_loader, val_loader = build_dataloader(cfg)
    train_fraction = cfg.get("data", {}).get("train_fraction", 1.0)
    fraction_str = f" ({train_fraction*100:.0f}% of full dataset)" if train_fraction < 1.0 else ""
    logger.info(f"Train samples: {len(train_loader.dataset)}{fraction_str}, Val samples: {len(val_loader.dataset)}")  # type: ignore
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        history = checkpoint.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.2f}%")
    
    # Training loop
    epochs = cfg["training"]["epochs"]
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler, use_amp
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, use_amp)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
        
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Track history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["lr"].append(current_lr)
        
        # Save history to CSV
        save_history_csv(history, os.path.join(run_dir, "history.csv"))
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )
        
        # Save checkpoints
        is_best = val_metrics["accuracy"] > best_acc
        if is_best:
            best_acc = val_metrics["accuracy"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc, cfg,
                os.path.join(run_dir, "best.pt"), scaler, history
            )
            logger.info(f"New best accuracy: {best_acc:.2f}%")
        
        # Always save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_acc, cfg,
            os.path.join(run_dir, "last.pt"), scaler, history
        )
    
    logger.info(f"Training complete! Best accuracy: {best_acc:.2f}%")
    logger.info(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a vision model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    main(args)
