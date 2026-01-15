#!/usr/bin/env python3
"""
Compare multiple trained models side-by-side.

Usage:
    # Compare all experiments in models/checkpoints/
    python scripts/compare_models.py
    
    # Compare specific experiments
    python scripts/compare_models.py --experiments simple_cnn_cifar10 convnextv2_tiny_cifar10
    
    # Export to CSV
    python scripts/compare_models.py --output results_comparison.csv
"""
import argparse
import os
import sys
from pathlib import Path
from tabulate import tabulate
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_checkpoint_info(checkpoint_path: str) -> dict:
    """Extract information from a checkpoint."""
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Count parameters
        num_params = sum(p.numel() for p in ckpt["model"].values() if isinstance(p, torch.Tensor))
        
        return {
            "experiment": Path(checkpoint_path).parent.name,
            "model": ckpt.get("config", {}).get("model", {}).get("name", "unknown"),
            "dataset": ckpt.get("config", {}).get("data", {}).get("dataset", "unknown"),
            "epoch": ckpt.get("epoch", 0),
            "best_acc": ckpt.get("best_acc", 0.0),
            "params": num_params,
            "optimizer": ckpt.get("config", {}).get("training", {}).get("optimizer", "unknown"),
            "lr": ckpt.get("config", {}).get("training", {}).get("lr", 0.0),
            "batch_size": ckpt.get("config", {}).get("training", {}).get("batch_size", 0),
            "checkpoint": checkpoint_path,
        }
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def find_all_experiments(checkpoint_dir: str) -> list:
    """Find all experiment directories with checkpoints."""
    experiments = []
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return experiments
    
    for exp_dir in checkpoint_path.iterdir():
        if exp_dir.is_dir():
            best_ckpt = exp_dir / "best.pt"
            if best_ckpt.exists():
                experiments.append(str(best_ckpt))
    
    return experiments


def main(args):
    checkpoint_dir = args.checkpoint_dir
    
    # Find experiments
    if args.experiments:
        # Specific experiments provided
        checkpoints = [
            os.path.join(checkpoint_dir, exp, "best.pt")
            for exp in args.experiments
        ]
    else:
        # Find all experiments
        checkpoints = find_all_experiments(checkpoint_dir)
    
    if not checkpoints:
        print(f"No experiments found in {checkpoint_dir}")
        print("Train a model first:")
        print("  python scripts/train.py --config configs/simple_cnn.yaml")
        return
    
    # Load checkpoint information
    results = []
    for ckpt_path in checkpoints:
        if os.path.exists(ckpt_path):
            info = load_checkpoint_info(ckpt_path)
            if info:
                results.append(info)
        else:
            print(f"Warning: Checkpoint not found: {ckpt_path}")
    
    if not results:
        print("No valid checkpoints found")
        return
    
    # Sort by accuracy (descending)
    results.sort(key=lambda x: x["best_acc"], reverse=True)
    
    # Prepare table
    headers = ["Rank", "Experiment", "Model", "Accuracy (%)", "Params (M)", "Epochs", "Optimizer", "LR", "Batch Size"]
    table_data = []
    
    for idx, r in enumerate(results, 1):
        table_data.append([
            idx,
            r["experiment"],
            r["model"],
            f"{r['best_acc']:.2f}",
            f"{r['params'] / 1e6:.2f}",
            r["epoch"] + 1,
            r["optimizer"],
            r["lr"],
            r["batch_size"],
        ])
    
    # Print table
    print("\n" + "="*120)
    print("MODEL COMPARISON")
    print("="*120)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*120)
    
    # Show evaluation commands
    print("\nTo evaluate a specific model:")
    for r in results:
        print(f"  python scripts/eval.py --checkpoint {r['checkpoint']}")
    
    # Save to CSV if requested
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained models")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/checkpoints",
        help="Directory containing experiment checkpoints"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Specific experiments to compare (by name)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save CSV results"
    )
    args = parser.parse_args()
    
    main(args)
