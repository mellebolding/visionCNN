#!/usr/bin/env python3
"""Analyze results from the normalization comparison experiment.

Reads history.csv from each experiment's log directory and generates
comparison plots. Uses only stdlib + matplotlib (no pandas dependency).

Usage:
    python scripts/analyze_norm_experiment.py --log_dir logs
    python scripts/analyze_norm_experiment.py --log_dir /projects/prjs0771/melle/projects/visionCNN/logs
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ["vgg_small", "vgg_medium", "resnet_small", "resnet_medium", "convnext_small", "convnext_medium"]
NORMS = ["batchnorm", "layernorm", "groupnorm"]
NORM_COLORS = {"batchnorm": "#1f77b4", "layernorm": "#ff7f0e", "groupnorm": "#2ca02c"}
NORM_LABELS = {"batchnorm": "BatchNorm", "layernorm": "LayerNorm", "groupnorm": "GroupNorm"}
ARCH_LABELS = {
    "vgg_small": "VGG-11", "vgg_medium": "VGG-16",
    "resnet_small": "ResNet-18", "resnet_medium": "ResNet-34",
    "convnext_small": "ConvNeXt-S", "convnext_medium": "ConvNeXt-M",
}


def load_csv(path):
    """Load a history.csv into a dict of lists."""
    data = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["epoch"].append(int(row["epoch"]))
            data["train_loss"].append(float(row["train_loss"]))
            data["train_acc"].append(float(row["train_acc"]))
            data["val_loss"].append(float(row["val_loss"]))
            data["val_acc"].append(float(row["val_acc"]))
            data["lr"].append(float(row["lr"]) if row.get("lr") else 0.0)
    return data


def load_results(log_dir):
    """Load history.csv from all experiments."""
    results = {}
    for model in MODELS:
        for norm in NORMS:
            exp_name = f"{model}_{norm}"
            csv_path = os.path.join(log_dir, exp_name, "history.csv")
            if os.path.exists(csv_path):
                results[exp_name] = load_csv(csv_path)
            else:
                print(f"  Warning: missing {csv_path}")
    return results


def plot_accuracy_curves(results, output_dir):
    """Plot val accuracy vs epoch, grouped by architecture."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for norm in NORMS:
            exp_name = f"{model}_{norm}"
            if exp_name in results:
                d = results[exp_name]
                ax.plot(d["epoch"], d["val_acc"],
                        color=NORM_COLORS[norm], label=NORM_LABELS[norm], linewidth=1.5)
        ax.set_title(ARCH_LABELS[model], fontsize=12)
        ax.set_xlabel("Epoch")
        if idx % 3 == 0:
            ax.set_ylabel("Val Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Validation Accuracy by Architecture and Normalization", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_curves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved accuracy_curves.png")


def plot_loss_curves(results, output_dir):
    """Plot train loss vs epoch, grouped by architecture."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for norm in NORMS:
            exp_name = f"{model}_{norm}"
            if exp_name in results:
                d = results[exp_name]
                ax.plot(d["epoch"], d["train_loss"],
                        color=NORM_COLORS[norm], label=NORM_LABELS[norm], linewidth=1.5)
        ax.set_title(ARCH_LABELS[model], fontsize=12)
        ax.set_xlabel("Epoch")
        if idx % 3 == 0:
            ax.set_ylabel("Train Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Training Loss by Architecture and Normalization", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved loss_curves.png")


def plot_final_accuracy_bar(results, output_dir):
    """Bar chart of final val accuracy for each model/norm combination."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = []
    bar_width = 0.25

    for i, model in enumerate(MODELS):
        base_x = i * (len(NORMS) + 1) * bar_width
        for j, norm in enumerate(NORMS):
            exp_name = f"{model}_{norm}"
            x = base_x + j * bar_width
            if exp_name in results:
                final_acc = results[exp_name]["val_acc"][-1]
                ax.bar(x, final_acc, bar_width * 0.85, color=NORM_COLORS[norm],
                       label=NORM_LABELS[norm] if i == 0 else "")
                ax.text(x, final_acc + 0.3, f"{final_acc:.1f}", ha='center', fontsize=7)
        x_labels.append(base_x + bar_width)

    ax.set_xticks(x_labels)
    ax.set_xticklabels([ARCH_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("Final Val Accuracy (%)")
    ax.set_title("Final Validation Accuracy: Normalization Comparison")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "final_accuracy_bar.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved final_accuracy_bar.png")


def plot_overfitting_bar(results, output_dir):
    """Bar chart of train-val accuracy gap (overfitting indicator)."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x_labels = []
    bar_width = 0.25

    for i, model in enumerate(MODELS):
        base_x = i * (len(NORMS) + 1) * bar_width
        for j, norm in enumerate(NORMS):
            exp_name = f"{model}_{norm}"
            x = base_x + j * bar_width
            if exp_name in results:
                gap = results[exp_name]["train_acc"][-1] - results[exp_name]["val_acc"][-1]
                ax.bar(x, gap, bar_width * 0.85, color=NORM_COLORS[norm],
                       label=NORM_LABELS[norm] if i == 0 else "")
                ax.text(x, gap + 0.15, f"{gap:.1f}", ha='center', fontsize=7)
        x_labels.append(base_x + bar_width)

    ax.set_xticks(x_labels)
    ax.set_xticklabels([ARCH_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("Train - Val Accuracy Gap (%)")
    ax.set_title("Overfitting Gap: Train Acc - Val Acc (lower = less overfitting)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "overfitting_gap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved overfitting_gap.png")


def print_summary_table(results):
    """Print a summary table of final accuracies."""
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'BatchNorm':>12} {'LayerNorm':>12} {'GroupNorm':>12}")
    print("-" * 70)
    for model in MODELS:
        row = f"{ARCH_LABELS[model]:<15}"
        for norm in NORMS:
            exp_name = f"{model}_{norm}"
            if exp_name in results:
                acc = results[exp_name]["val_acc"][-1]
                row += f" {acc:>11.2f}%"
            else:
                row += f" {'N/A':>12}"
        print(row)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze normalization experiment results")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--output_dir", type=str, default="results/norm_experiment", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    results = load_results(args.log_dir)
    print(f"Loaded {len(results)}/{len(MODELS) * len(NORMS)} experiments.\n")

    if not results:
        print("No results found. Check --log_dir path.")
        sys.exit(1)

    print("Generating plots...")
    plot_accuracy_curves(results, args.output_dir)
    plot_loss_curves(results, args.output_dir)
    plot_final_accuracy_bar(results, args.output_dir)
    plot_overfitting_bar(results, args.output_dir)

    print_summary_table(results)
    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
