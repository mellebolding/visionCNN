#!/usr/bin/env python3
"""
Unified analysis script for VisionCNN experiments.

Usage:
    # Quick summary table of all runs
    python scripts/analyze.py summary
    
    # Plot training curves for one or more runs
    python scripts/analyze.py curves --runs logs/simple_cnn_cifar10 logs/convnextv2_tiny_cifar10
    
    # Full evaluation with per-class metrics
    python scripts/analyze.py evaluate --run logs/simple_cnn_cifar10
    
    # Compare runs side-by-side (table + curves)
    python scripts/analyze.py compare --runs logs/simple_cnn_cifar10 logs/convnextv2_tiny_cifar10
"""
import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

import matplotlib
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.build import build_dataloader


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_history_from_csv(csv_path: str) -> dict | None:
    """Load training history from CSV file."""
    try:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history["train_loss"].append(float(row["train_loss"]))
                history["train_acc"].append(float(row["train_acc"]))
                history["val_loss"].append(float(row["val_loss"]))
                history["val_acc"].append(float(row["val_acc"]))
                if row.get("lr"):
                    history["lr"].append(float(row["lr"]))
        return history
    except Exception as e:
        print(f"Error loading CSV {csv_path}: {e}")
        return None


def load_run_info(run_dir: str) -> dict | None:
    """Load all information about a run."""
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Run directory not found: {run_dir}")
        return None
    
    info = {
        "run_dir": str(run_path),
        "name": run_path.name,
        "history": None,
        "config": {},
        "best_acc": 0.0,
        "epochs": 0,
        "model_name": "unknown",
        "params": 0,
        "train_fraction": 1.0,
    }
    
    # Load history from CSV
    csv_path = run_path / "history.csv"
    if csv_path.exists():
        info["history"] = load_history_from_csv(str(csv_path))
        if info["history"]:
            info["epochs"] = len(info["history"]["train_loss"])
            info["best_acc"] = max(info["history"]["val_acc"]) if info["history"]["val_acc"] else 0.0
    
    # Load config and additional info from checkpoint
    for ckpt_name in ["best.pt", "last.pt"]:
        ckpt_path = run_path / ckpt_name
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                info["config"] = ckpt.get("config", {})
                info["model_name"] = info["config"].get("model", {}).get("name", "unknown")
                info["train_fraction"] = info["config"].get("data", {}).get("train_fraction", 1.0)
                info["best_acc"] = ckpt.get("best_acc", info["best_acc"])
                info["epochs"] = max(info["epochs"], ckpt.get("epoch", 0) + 1)
                # Count parameters from state dict
                info["params"] = sum(
                    p.numel() for p in ckpt.get("model", {}).values() 
                    if isinstance(p, torch.Tensor)
                )
                break
            except Exception as e:
                print(f"Error loading checkpoint {ckpt_path}: {e}")
    
    return info


def find_all_runs(logs_dir: str = "logs") -> list[dict]:
    """Find all experiment runs."""
    runs = []
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        return runs
    
    for run_dir in sorted(logs_path.iterdir()):
        if run_dir.is_dir():
            # Check if it has checkpoints or history
            if (run_dir / "best.pt").exists() or (run_dir / "history.csv").exists():
                info = load_run_info(str(run_dir))
                if info:
                    runs.append(info)
    
    return runs


# =============================================================================
# Summary Command
# =============================================================================

def cmd_summary(args):
    """Show summary table of all runs."""
    runs = find_all_runs(args.logs_dir)
    
    if not runs:
        print(f"No runs found in {args.logs_dir}")
        return
    
    # Sort by best accuracy
    runs.sort(key=lambda x: x["best_acc"], reverse=True)
    
    # Print table
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"{'Rank':<5} {'Experiment':<30} {'Model':<18} {'Data':<8} {'Acc (%)':<10} {'Params (M)':<12} {'Epochs':<8}")
    print("-" * 100)
    
    for idx, run in enumerate(runs, 1):
        data_str = f"{run['train_fraction']*100:.0f}%" if run['train_fraction'] < 1.0 else "100%"
        print(f"{idx:<5} {run['name']:<30} {run['model_name']:<18} {data_str:<8} "
              f"{run['best_acc']:<10.2f} {run['params']/1e6:<12.2f} {run['epochs']:<8}")
    
    print("=" * 100)
    
    # Show commands to analyze further
    if runs:
        print(f"\nTo plot curves:   python scripts/analyze.py curves --run {runs[0]['run_dir']}")
        print(f"To evaluate:      python scripts/analyze.py evaluate --run {runs[0]['run_dir']}")


# =============================================================================
# Curves Command
# =============================================================================

def cmd_curves(args):
    """Plot training curves."""
    run_dirs = args.runs if args.runs else [args.run] if args.run else []
    
    if not run_dirs:
        print("Error: Provide --run or --runs")
        return
    
    # Load run info
    runs = []
    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        if info and info["history"]:
            runs.append(info)
    
    if not runs:
        print("No valid runs with history found")
        return
    
    # Use provided names or experiment names
    names = args.names if args.names else [r["name"] for r in runs]
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for run, name in zip(runs, names):
        h = run["history"]
        print(f"\n{name} ({run['name']})")
        print(f"  Epochs: {run['epochs']}, Best val acc: {run['best_acc']:.2f}%")
        print(f"  Final: train_loss={h['train_loss'][-1]:.4f}, val_loss={h['val_loss'][-1]:.4f}, "
              f"train_acc={h['train_acc'][-1]:.2f}%, val_acc={h['val_acc'][-1]:.2f}%")
    print("=" * 80)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # pyright: ignore
    # Colors: SimpleCNN variations (blues), ConvNeXtV2 variations (reds)
    colors = [
        "#1a3a6e",  # dark blue (SimpleCNN 100%)
        "#4a7ab0",  # medium blue (SimpleCNN 50%)
        "#8bbbd9",  # light blue (SimpleCNN 10%)
        "#8b1a1a",  # dark red (ConvNeXtV2 100%)
        "#c45a5a",  # medium red (ConvNeXtV2 50%)
        "#e89a9a",  # light red (ConvNeXtV2 10%)
    ]
    
    for i, (run, name) in enumerate(zip(runs, names)):
        color = colors[i % len(colors)]
        h = run["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        
        axes[0, 0].plot(epochs, h["train_loss"], label=name, color=color, linewidth=2)  # pyright: ignore
        axes[0, 1].plot(epochs, h["val_loss"], label=name, color=color, linewidth=2)  # pyright: ignore
        axes[1, 0].plot(epochs, h["train_acc"], label=name, color=color, linewidth=2)  # pyright: ignore
        axes[1, 1].plot(epochs, h["val_acc"], label=name, color=color, linewidth=2)  # pyright: ignore
    
    for ax, title, ylabel in [  # pyright: ignore
        (axes[0, 0], "Training Loss", "Loss"),  # pyright: ignore
        (axes[0, 1], "Validation Loss", "Loss"),  # pyright: ignore
        (axes[1, 0], "Training Accuracy", "Accuracy (%)"),  # pyright: ignore
        (axes[1, 1], "Validation Accuracy", "Accuracy (%)"),  # pyright: ignore
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    output = args.output or "results/training_curves.png"
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else "results", exist_ok=True)
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output}")


# =============================================================================
# Evaluate Command
# =============================================================================

@torch.no_grad()
def cmd_evaluate(args):
    """Full evaluation with per-class metrics."""
    run_dir = args.run
    if not run_dir:
        print("Error: Provide --run")
        return
    
    info = load_run_info(run_dir)
    if not info:
        return
    
    # Load checkpoint
    ckpt_path = Path(run_dir) / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(run_dir) / "last.pt"
    
    if not ckpt_path.exists():
        print(f"No checkpoint found in {run_dir}")
        return
    
    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg['model']['name']}, Parameters: {num_params:,}")
    
    # Build dataloader
    _, val_loader = build_dataloader(cfg)
    
    # Get class names
    dataset_name = cfg.get("data", {}).get("dataset", "cifar10").lower()
    class_names = {
        "cifar10": ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "svhn": [str(i) for i in range(10)],
    }.get(dataset_name, [str(i) for i in range(cfg["model"]["num_classes"])])
    
    num_classes = cfg["model"]["num_classes"]
    
    # Evaluate
    print(f"\nEvaluating on {len(val_loader.dataset)} samples...")  # type: ignore
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    avg_loss = total_loss / len(all_labels)
    
    # Per-class metrics
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    for pred, label in zip(all_preds, all_labels):
        per_class[label]["total"] += 1
        if pred == label:
            per_class[label]["tp"] += 1
        else:
            per_class[label]["fn"] += 1
            per_class[pred]["fp"] += 1
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\n📊 Overall: Accuracy = {accuracy:.2f}%, Loss = {avg_loss:.4f}")
    
    print(f"\n{'Class':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Count':<8}")
    print("-" * 80)
    
    for i in range(num_classes):
        c = per_class[i]
        acc = 100.0 * c["tp"] / c["total"] if c["total"] > 0 else 0
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name:<12} {acc:<12.2f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f} {c['total']:<8}")
    
    print("=" * 80)
    
    # Save results
    if args.output:
        import json
        results = {
            "run": run_dir,
            "accuracy": accuracy,
            "loss": avg_loss,
            "per_class": {class_names[i]: {
                "accuracy": 100.0 * per_class[i]["tp"] / per_class[i]["total"] if per_class[i]["total"] > 0 else 0,
                "count": per_class[i]["total"]
            } for i in range(num_classes)}
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


# =============================================================================
# Compare Command
# =============================================================================

def cmd_compare(args):
    """Compare multiple runs with table and curves."""
    run_dirs = args.runs
    if not run_dirs or len(run_dirs) < 2:
        print("Error: Provide at least 2 runs with --runs")
        return
    
    # Load runs
    runs = []
    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        if info:
            runs.append(info)
    
    if len(runs) < 2:
        print("Need at least 2 valid runs to compare")
        return
    
    names = args.names if args.names else [r["name"] for r in runs]
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(f"{'Model':<25} {'Best Acc (%)':<15} {'Final Acc (%)':<15} {'Params (M)':<12} {'Epochs':<8}")
    print("-" * 90)
    
    for run, name in zip(runs, names):
        final_acc = run["history"]["val_acc"][-1] if run["history"] else 0
        print(f"{name:<25} {run['best_acc']:<15.2f} {final_acc:<15.2f} "
              f"{run['params']/1e6:<12.2f} {run['epochs']:<8}")
    
    print("=" * 90)
    
    # Show winner
    best_run = max(runs, key=lambda x: x["best_acc"])
    best_name = names[runs.index(best_run)]
    print(f"\nHighest acc.: {best_name} with {best_run['best_acc']:.2f}% accuracy")
    
    # Plot curves if history available
    runs_with_history = [(r, n) for r, n in zip(runs, names) if r["history"]]
    if runs_with_history:
        args.runs = [r["run_dir"] for r, _ in runs_with_history]
        args.names = [n for _, n in runs_with_history]
        args.run = None
        args.output = args.output or "results/training_comparison.png"
        cmd_curves(args)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze VisionCNN experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze.py summary
  python scripts/analyze.py curves --run logs/simple_cnn_cifar10
  python scripts/analyze.py curves --runs logs/simple_cnn_cifar10 logs/convnextv2_tiny_cifar10
  python scripts/analyze.py evaluate --run logs/simple_cnn_cifar10
  python scripts/analyze.py compare --runs logs/simple_cnn_cifar10 logs/convnextv2_tiny_cifar10
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Summary command
    p_summary = subparsers.add_parser("summary", help="Show summary table of all runs")
    p_summary.add_argument("--logs-dir", type=str, default="logs", help="Logs directory")
    
    # Curves command
    p_curves = subparsers.add_parser("curves", help="Plot training curves")
    p_curves.add_argument("--run", type=str, help="Single run directory")
    p_curves.add_argument("--runs", type=str, nargs="+", help="Multiple run directories")
    p_curves.add_argument("--names", type=str, nargs="+", help="Names for legend")
    p_curves.add_argument("--output", type=str, help="Output file path")
    
    # Evaluate command
    p_eval = subparsers.add_parser("evaluate", help="Full evaluation with per-class metrics")
    p_eval.add_argument("--run", type=str, required=True, help="Run directory")
    p_eval.add_argument("--output", type=str, help="Save results to JSON")
    
    # Compare command
    p_compare = subparsers.add_parser("compare", help="Compare multiple runs")
    p_compare.add_argument("--runs", type=str, nargs="+", required=True, help="Run directories to compare")
    p_compare.add_argument("--names", type=str, nargs="+", help="Names for the runs")
    p_compare.add_argument("--output", type=str, help="Output file for plot")
    
    args = parser.parse_args()
    
    if args.command == "summary":
        cmd_summary(args)
    elif args.command == "curves":
        cmd_curves(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
