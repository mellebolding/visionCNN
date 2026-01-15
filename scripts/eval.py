#!/usr/bin/env python3
"""
Evaluation script for VisionCNN models.

Usage:
    python scripts/eval.py --checkpoint models/checkpoints/best.pt
    python scripts/eval.py --checkpoint models/checkpoints/best.pt --config configs/custom.yaml
"""
import argparse
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import yaml
import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.build import build_dataloader


def load_checkpoint(filepath: str, model: nn.Module = None, device: torch.device = None):
    """Load model checkpoint and return config."""
    checkpoint = torch.load(filepath, map_location=device or "cpu")
    cfg = checkpoint.get("config", {})
    
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    
    return cfg, checkpoint


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = True,
    num_classes: int = 10,
) -> dict:
    """Evaluate model and return comprehensive metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        if use_amp:
            with autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        total_loss += loss.item() * images.size(0)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_labels, all_probs, num_classes)
    metrics["loss"] = total_loss / len(all_labels)
    
    return metrics


def calculate_metrics(preds: np.ndarray, labels: np.ndarray, probs: np.ndarray, num_classes: int) -> dict:
    """Calculate comprehensive classification metrics."""
    # Overall accuracy
    accuracy = 100.0 * (preds == labels).sum() / len(labels)
    
    # Per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)
    
    for pred, label in zip(preds, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
            class_tp[label] += 1
        else:
            class_fn[label] += 1
            class_fp[pred] += 1
    
    # Per-class accuracy
    per_class_acc = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}
    
    for c in range(num_classes):
        if class_total[c] > 0:
            per_class_acc[c] = 100.0 * class_correct[c] / class_total[c]
        else:
            per_class_acc[c] = 0.0
        
        # Precision
        if class_tp[c] + class_fp[c] > 0:
            per_class_precision[c] = class_tp[c] / (class_tp[c] + class_fp[c])
        else:
            per_class_precision[c] = 0.0
        
        # Recall
        if class_tp[c] + class_fn[c] > 0:
            per_class_recall[c] = class_tp[c] / (class_tp[c] + class_fn[c])
        else:
            per_class_recall[c] = 0.0
        
        # F1
        if per_class_precision[c] + per_class_recall[c] > 0:
            per_class_f1[c] = 2 * per_class_precision[c] * per_class_recall[c] / (
                per_class_precision[c] + per_class_recall[c]
            )
        else:
            per_class_f1[c] = 0.0
    
    # Macro-averaged metrics
    macro_precision = np.mean(list(per_class_precision.values()))
    macro_recall = np.mean(list(per_class_recall.values()))
    macro_f1 = np.mean(list(per_class_f1.values()))
    
    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for pred, label in zip(preds, labels):
        confusion_matrix[label, pred] += 1
    
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision * 100,
        "macro_recall": macro_recall * 100,
        "macro_f1": macro_f1 * 100,
        "per_class_accuracy": per_class_acc,
        "per_class_precision": {k: v * 100 for k, v in per_class_precision.items()},
        "per_class_recall": {k: v * 100 for k, v in per_class_recall.items()},
        "per_class_f1": {k: v * 100 for k, v in per_class_f1.items()},
        "confusion_matrix": confusion_matrix.tolist(),
        "total_samples": len(labels),
    }


def print_results(metrics: dict, class_names: list = None):
    """Pretty print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"   Loss:            {metrics['loss']:.4f}")
    print(f"   Macro Precision: {metrics['macro_precision']:.2f}%")
    print(f"   Macro Recall:    {metrics['macro_recall']:.2f}%")
    print(f"   Macro F1:        {metrics['macro_f1']:.2f}%")
    print(f"   Total Samples:   {metrics['total_samples']}")
    
    print(f"\n📈 Per-Class Accuracy:")
    num_classes = len(metrics['per_class_accuracy'])
    for c in range(num_classes):
        name = class_names[c] if class_names else f"Class {c}"
        acc = metrics['per_class_accuracy'][c]
        prec = metrics['per_class_precision'][c]
        rec = metrics['per_class_recall'][c]
        f1 = metrics['per_class_f1'][c]
        print(f"   {name:>12}: Acc={acc:5.1f}%, Prec={prec:5.1f}%, Rec={rec:5.1f}%, F1={f1:5.1f}%")
    
    print("\n" + "=" * 60)


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    cfg, checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    # Override config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    
    # Build model
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}, Parameters: {num_params:,}")
    print(f"Checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Build dataloader (validation set)
    _, val_loader = build_dataloader(cfg)
    print(f"Evaluating on {len(val_loader.dataset)} samples")
    
    # Get class names if available
    dataset_name = cfg.get("data", {}).get("dataset", "cifar10").lower()
    class_names_map = {
        "cifar10": ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "cifar100": None,  # Too many classes
        "svhn": [str(i) for i in range(10)],
    }
    class_names = class_names_map.get(dataset_name)
    
    # Evaluate
    use_amp = cfg.get("training", {}).get("use_amp", True) and device.type == "cuda"
    num_classes = cfg["model"]["num_classes"]
    
    metrics = evaluate(model, val_loader, device, use_amp=use_amp, num_classes=num_classes)
    
    # Print results
    print_results(metrics, class_names)
    
    # Save results if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            # Convert numpy arrays for JSON serialization
            metrics_json = {k: v for k, v in metrics.items()}
            json.dump(metrics_json, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Override config file")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()
    
    main(args)