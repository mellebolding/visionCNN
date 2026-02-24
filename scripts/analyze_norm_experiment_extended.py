#!/usr/bin/env python3
"""Analyze results from the extended normalization experiment.

Generates plots for:
  1. Batch size sensitivity (val accuracy vs batch size, per model/norm)
  2. Learning rate sensitivity (val accuracy vs LR, per model/norm)
  3. Training dynamics (loss curves for NoNorm vs norms)
  4. Gradient flow analysis (gradient norms per layer over training)
  5. Feature analysis summary (effective rank, Fisher ratio from extract_features.py output)

Usage:
    python scripts/analyze_norm_experiment_extended.py --log_dir /path/to/logs
    python scripts/analyze_norm_experiment_extended.py --log_dir /path/to/logs --features_dir results/features
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["vgg_medium", "resnet_medium", "convnext_medium"]
NORMS = ["batchnorm", "layernorm", "groupnorm", "rmsnorm", "rmsnorm_bias", "nonorm", "nonorm_ws"]
BATCH_SIZES = [16, 32, 128, 512]
LEARNING_RATES = [0.0003, 0.001, 0.003]

NORM_COLORS = {
    "batchnorm": "#1f77b4", "layernorm": "#ff7f0e",
    "groupnorm": "#2ca02c", "rmsnorm": "#e377c2",
    "rmsnorm_bias": "#8c564b", "nonorm": "#d62728",
    "nonorm_ws": "#9467bd",
}
NORM_LABELS = {
    "batchnorm": "BatchNorm", "layernorm": "LayerNorm",
    "groupnorm": "GroupNorm", "rmsnorm": "RMSNorm",
    "rmsnorm_bias": "RMSNorm+bias", "nonorm": "NoNorm",
    "nonorm_ws": "NoNorm+WS+AGC",
}
ARCH_LABELS = {
    "vgg_medium": "VGG-16", "resnet_medium": "ResNet-34",
    "convnext_medium": "ConvNeXtV2-M",
}
NORM_LINESTYLES = {
    "batchnorm": "-", "layernorm": "--",
    "groupnorm": "-.", "rmsnorm": (0, (5, 2)),
    "rmsnorm_bias": (0, (5, 1)), "nonorm": ":",
    "nonorm_ws": (0, (3, 1, 1, 1)),
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


def load_gradient_csv(path):
    """Load gradient_stats.csv into a dict of {layer_name: [(step, norm), ...]}."""
    grad_data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            epoch = int(row["epoch"])
            for key, val in row.items():
                if key not in ("step", "epoch"):
                    grad_data[key].append((step, float(val)))
    return grad_data


def exp_name(model, norm, bs, lr):
    return f"{model}_{norm}_bs{bs}_lr{lr}"


def load_all_results(log_dir):
    """Load history.csv from all extended experiments."""
    results = {}
    for model in MODELS:
        for norm in NORMS:
            # Batch size sweep (lr=0.001)
            for bs in BATCH_SIZES:
                name = exp_name(model, norm, bs, "0.001")
                csv_path = os.path.join(log_dir, name, "history.csv")
                if os.path.exists(csv_path):
                    results[name] = load_csv(csv_path)
            # LR sweep (bs=128)
            for lr in LEARNING_RATES:
                name = exp_name(model, norm, 128, lr)
                csv_path = os.path.join(log_dir, name, "history.csv")
                if os.path.exists(csv_path):
                    results[name] = load_csv(csv_path)
    # Also try loading Phase 1 results (different naming: model_norm without bs/lr)
    for model in MODELS:
        for norm in ["batchnorm", "layernorm", "groupnorm"]:
            old_name = f"{model}_{norm}"
            csv_path = os.path.join(log_dir, old_name, "history.csv")
            if os.path.exists(csv_path) and old_name not in results:
                results[old_name] = load_csv(csv_path)
    return results


def get_final_acc(results, name):
    """Get final val accuracy for an experiment, trying extended and Phase 1 names."""
    if name in results:
        return results[name]["val_acc"][-1]
    # Try Phase 1 naming for bs=128, lr=0.001
    parts = name.split("_")
    # e.g. vgg_medium_batchnorm_bs128_lr0.001 -> vgg_medium_batchnorm
    if "bs128" in name and "lr0.001" in name:
        old_name = "_".join(parts[:3])  # model_size_norm
        if old_name in results:
            return results[old_name]["val_acc"][-1]
    return None


def plot_batch_size_sensitivity(results, output_dir):
    """Plot final val accuracy vs batch size for each model, colored by norm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for norm in NORMS:
            bs_vals = []
            acc_vals = []
            for bs in BATCH_SIZES:
                name = exp_name(model, norm, bs, "0.001")
                acc = get_final_acc(results, name)
                if acc is not None:
                    bs_vals.append(bs)
                    acc_vals.append(acc)
            if bs_vals:
                ax.plot(bs_vals, acc_vals, 'o-', color=NORM_COLORS[norm],
                        label=NORM_LABELS[norm], linewidth=2, markersize=6)

        ax.set_title(ARCH_LABELS[model], fontsize=13)
        ax.set_xlabel("Batch Size")
        ax.set_xscale('log', base=2)
        ax.set_xticks(BATCH_SIZES)
        ax.set_xticklabels([str(b) for b in BATCH_SIZES])
        if idx == 0:
            ax.set_ylabel("Final Val Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Batch Size Sensitivity (lr=0.001)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "batch_size_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved batch_size_sensitivity.png")


def plot_lr_sensitivity(results, output_dir):
    """Plot final val accuracy vs learning rate for each model, colored by norm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        for norm in NORMS:
            lr_vals = []
            acc_vals = []
            for lr in LEARNING_RATES:
                name = exp_name(model, norm, 128, lr)
                acc = get_final_acc(results, name)
                if acc is not None:
                    lr_vals.append(lr)
                    acc_vals.append(acc)
            if lr_vals:
                ax.plot(lr_vals, acc_vals, 'o-', color=NORM_COLORS[norm],
                        label=NORM_LABELS[norm], linewidth=2, markersize=6)

        ax.set_title(ARCH_LABELS[model], fontsize=13)
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        if idx == 0:
            ax.set_ylabel("Final Val Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Learning Rate Sensitivity (bs=128)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lr_sensitivity.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved lr_sensitivity.png")


def plot_nonorm_training_dynamics(results, output_dir):
    """Plot training loss curves comparing NoNorm vs normed models at bs=128, lr=0.001."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for idx, model in enumerate(MODELS):
        # Top row: train loss
        ax_loss = axes[0, idx]
        # Bottom row: val accuracy
        ax_acc = axes[1, idx]

        for norm in NORMS:
            name = exp_name(model, norm, 128, "0.001")
            if name not in results:
                # Try Phase 1 naming
                old_name = f"{model}_{norm}"
                if old_name in results:
                    name = old_name
                else:
                    continue
            d = results[name]
            ax_loss.plot(d["epoch"], d["train_loss"], color=NORM_COLORS[norm],
                         linestyle=NORM_LINESTYLES[norm], label=NORM_LABELS[norm], linewidth=1.5)
            ax_acc.plot(d["epoch"], d["val_acc"], color=NORM_COLORS[norm],
                        linestyle=NORM_LINESTYLES[norm], label=NORM_LABELS[norm], linewidth=1.5)

        ax_loss.set_title(ARCH_LABELS[model], fontsize=12)
        ax_loss.set_xlabel("Epoch")
        ax_acc.set_xlabel("Epoch")
        if idx == 0:
            ax_loss.set_ylabel("Train Loss")
            ax_acc.set_ylabel("Val Accuracy (%)")
        ax_loss.grid(True, alpha=0.3)
        ax_acc.grid(True, alpha=0.3)
        ax_loss.legend(fontsize=8)
        ax_acc.legend(fontsize=8)

    fig.suptitle("Training Dynamics: NoNorm vs Normalization (bs=128, lr=0.001)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nonorm_dynamics.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved nonorm_dynamics.png")


def find_gradient_csv(log_dir, model, norm):
    """Find a gradient_stats.csv for a given model+norm, trying multiple naming conventions."""
    # Try extended naming first (bs128, lr0.001)
    candidates = [
        exp_name(model, norm, 128, "0.001"),
        f"{model}_{norm}",  # Phase 1 naming
        exp_name(model, norm, 128, "0.0003"),
        exp_name(model, norm, 128, "0.003"),
        exp_name(model, norm, 32, "0.001"),
    ]
    for name in candidates:
        path = os.path.join(log_dir, name, "gradient_stats.csv")
        if os.path.exists(path):
            return path, name
    return None, None


def plot_gradient_flow(log_dir, output_dir):
    """Plot gradient norm distributions per layer for each model/norm."""
    fig, axes = plt.subplots(len(MODELS), len(NORMS), figsize=(20, 12))

    has_data = False
    for i, model in enumerate(MODELS):
        for j, norm in enumerate(NORMS):
            ax = axes[i, j]
            grad_path, source_name = find_gradient_csv(log_dir, model, norm)

            if grad_path is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f"{ARCH_LABELS[model]} + {NORM_LABELS[norm]}", fontsize=9)
                continue

            has_data = True
            grad_data = load_gradient_csv(grad_path)

            # Plot gradient norms for each layer over training steps
            layer_names = sorted(grad_data.keys())
            # Sample layers evenly to avoid clutter (max 10 layers)
            if len(layer_names) > 10:
                indices = np.linspace(0, len(layer_names) - 1, 10, dtype=int)
                layer_names = [layer_names[k] for k in indices]

            for layer in layer_names:
                steps, norms = zip(*grad_data[layer])
                short_name = layer.split('.')[-2] + '.' + layer.split('.')[-1] if '.' in layer else layer
                ax.semilogy(steps, norms, linewidth=0.8, alpha=0.7, label=short_name)

            ax.set_title(f"{ARCH_LABELS[model]} + {NORM_LABELS[norm]}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Gradient L2 Norm")
            if i == len(MODELS) - 1:
                ax.set_xlabel("Training Step")
            ax.grid(True, alpha=0.2)
            # Only show legend for first subplot to avoid clutter
            if i == 0 and j == 0:
                ax.legend(fontsize=5, loc='upper right', ncol=2)

    if has_data:
        fig.suptitle("Gradient Flow Analysis: Per-Layer Gradient Norms Over Training", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradient_flow.png"), dpi=150, bbox_inches='tight')
        print("  Saved gradient_flow.png")
    else:
        print("  Skipping gradient_flow.png (no gradient data found)")
    plt.close(fig)


def plot_gradient_summary(log_dir, output_dir):
    """Bar chart of mean gradient norms (early vs late training) per model/norm."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    has_data = False
    for idx, model in enumerate(MODELS):
        ax = axes[idx]
        early_means = []
        late_means = []
        labels = []

        for norm in NORMS:
            grad_path, _ = find_gradient_csv(log_dir, model, norm)
            if grad_path is None:
                continue

            has_data = True
            grad_data = load_gradient_csv(grad_path)

            # Aggregate all layer norms
            all_steps = []
            all_norms = []
            for layer, data_points in grad_data.items():
                for step, norm_val in data_points:
                    all_steps.append(step)
                    all_norms.append(norm_val)

            if not all_steps:
                continue

            all_steps = np.array(all_steps)
            all_norms = np.array(all_norms)
            mid = np.median(all_steps)

            early_mask = all_steps < mid
            late_mask = all_steps >= mid

            early_mean = all_norms[early_mask].mean() if early_mask.any() else 0
            late_mean = all_norms[late_mask].mean() if late_mask.any() else 0

            labels.append(NORM_LABELS[norm])
            early_means.append(early_mean)
            late_means.append(late_mean)

        if labels:
            x = np.arange(len(labels))
            w = 0.35
            ax.bar(x - w/2, early_means, w, label='Early training', alpha=0.8)
            ax.bar(x + w/2, late_means, w, label='Late training', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Mean Gradient L2 Norm")
            ax.set_title(ARCH_LABELS[model], fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.set_yscale('log')

    if has_data:
        fig.suptitle("Gradient Magnitude: Early vs Late Training", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "gradient_summary.png"), dpi=150, bbox_inches='tight')
        print("  Saved gradient_summary.png")
    else:
        print("  Skipping gradient_summary.png (no gradient data found)")
    plt.close(fig)


def plot_feature_analysis(features_dir, output_dir):
    """Plot feature analysis metrics (effective rank, Fisher ratio) from extract_features.py output."""
    if not os.path.exists(features_dir):
        print(f"  Skipping feature analysis (directory not found: {features_dir})")
        return

    metrics = {}
    for model in MODELS:
        for norm in NORMS:
            name = exp_name(model, norm, 128, "0.001")
            csv_path = os.path.join(features_dir, name, "feature_metrics.csv")
            if not os.path.exists(csv_path):
                # Try Phase 1 naming
                old_name = f"{model}_{norm}"
                csv_path = os.path.join(features_dir, old_name, "feature_metrics.csv")
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metrics[name] = {k: float(v) for k, v in row.items()}

    if not metrics:
        print("  Skipping feature analysis (no feature_metrics.csv files found)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Effective rank
    ax = axes[0]
    for norm in NORMS:
        vals = []
        model_labels = []
        for model in MODELS:
            name = exp_name(model, norm, 128, "0.001")
            if name in metrics:
                vals.append(metrics[name]["effective_rank"])
                model_labels.append(ARCH_LABELS[model])
        if vals:
            ax.bar([m + f"\n{NORM_LABELS[norm]}" for m in model_labels], vals,
                   color=NORM_COLORS[norm], alpha=0.8, label=NORM_LABELS[norm])
    ax.set_ylabel("Effective Rank")
    ax.set_title("Feature Space Effective Rank")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Fisher ratio
    ax = axes[1]
    for idx, model in enumerate(MODELS):
        vals = []
        norm_labels_local = []
        for norm in NORMS:
            name = exp_name(model, norm, 128, "0.001")
            if name in metrics:
                vals.append(metrics[name]["fisher_ratio"])
                norm_labels_local.append(NORM_LABELS[norm])
        if vals:
            x = np.arange(len(norm_labels_local)) + idx * (len(NORMS) + 1)
            ax.bar(x, vals, color=[NORM_COLORS[n] for n in NORMS if
                   exp_name(model, n, 128, "0.001") in metrics], alpha=0.8)
    ax.set_ylabel("Fisher Discriminant Ratio")
    ax.set_title("Class Separability (Fisher Ratio)")
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: Feature norms
    ax = axes[2]
    for idx, model in enumerate(MODELS):
        for j, norm in enumerate(NORMS):
            name = exp_name(model, norm, 128, "0.001")
            if name in metrics:
                x = idx * (len(NORMS) + 1) + j
                mean = metrics[name]["feat_norm_mean"]
                std = metrics[name]["feat_norm_std"]
                ax.bar(x, mean, yerr=std, color=NORM_COLORS[norm], alpha=0.8,
                       capsize=3, label=NORM_LABELS[norm] if idx == 0 else "")
    ax.set_ylabel("Feature L2 Norm")
    ax.set_title("Feature Norms (mean +/- std)")
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Feature Representation Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "feature_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved feature_analysis.png")


def plot_heatmap(results, output_dir):
    """Heatmap of final val accuracy: rows=model+norm, cols=batch_size (at lr=0.001)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    row_labels = []
    data_matrix = []

    for model in MODELS:
        for norm in NORMS:
            row_labels.append(f"{ARCH_LABELS[model]} + {NORM_LABELS[norm]}")
            row = []
            for bs in BATCH_SIZES:
                name = exp_name(model, norm, bs, "0.001")
                acc = get_final_acc(results, name)
                row.append(acc if acc is not None else np.nan)
            data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=np.nanmin(data_matrix) - 2, vmax=np.nanmax(data_matrix) + 2)

    ax.set_xticks(range(len(BATCH_SIZES)))
    ax.set_xticklabels([str(b) for b in BATCH_SIZES])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Batch Size")
    ax.set_title("Final Val Accuracy Heatmap (lr=0.001)")

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(BATCH_SIZES)):
            val = data_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8,
                        color='white' if val < np.nanmedian(data_matrix) else 'black')

    fig.colorbar(im, ax=ax, label="Val Accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "accuracy_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved accuracy_heatmap.png")


def print_summary_table(results):
    """Print comprehensive summary tables."""
    # Table 1: Batch size sweep
    print("\n" + "=" * 90)
    print("BATCH SIZE SENSITIVITY (lr=0.001)")
    print("=" * 90)
    header = f"{'Model + Norm':<35}"
    for bs in BATCH_SIZES:
        header += f"{'bs=' + str(bs):>12}"
    print(header)
    print("-" * 90)

    for model in MODELS:
        for norm in NORMS:
            row = f"{ARCH_LABELS[model] + ' + ' + NORM_LABELS[norm]:<35}"
            for bs in BATCH_SIZES:
                name = exp_name(model, norm, bs, "0.001")
                acc = get_final_acc(results, name)
                if acc is not None:
                    row += f"{acc:>11.2f}%"
                else:
                    row += f"{'N/A':>12}"
            print(row)
        print()

    # Table 2: LR sweep
    print("=" * 80)
    print("LEARNING RATE SENSITIVITY (bs=128)")
    print("=" * 80)
    header = f"{'Model + Norm':<35}"
    for lr in LEARNING_RATES:
        header += f"{'lr=' + str(lr):>14}"
    print(header)
    print("-" * 80)

    for model in MODELS:
        for norm in NORMS:
            row = f"{ARCH_LABELS[model] + ' + ' + NORM_LABELS[norm]:<35}"
            for lr in LEARNING_RATES:
                name = exp_name(model, norm, 128, lr)
                acc = get_final_acc(results, name)
                if acc is not None:
                    row += f"{acc:>13.2f}%"
                else:
                    row += f"{'N/A':>14}"
            print(row)
        print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Analyze extended normalization experiment results")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory with experiment folders")
    parser.add_argument("--output_dir", type=str, default="results/norm_experiment_extended",
                        help="Output directory for plots")
    parser.add_argument("--features_dir", type=str, default="results/features",
                        help="Directory with feature analysis output from extract_features.py")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading results...")
    results = load_all_results(args.log_dir)
    print(f"Loaded {len(results)} experiments.\n")

    if not results:
        print("No results found. Check --log_dir path.")
        sys.exit(1)

    print("Generating plots...")
    plot_batch_size_sensitivity(results, args.output_dir)
    plot_lr_sensitivity(results, args.output_dir)
    plot_nonorm_training_dynamics(results, args.output_dir)
    plot_heatmap(results, args.output_dir)
    plot_gradient_flow(args.log_dir, args.output_dir)
    plot_gradient_summary(args.log_dir, args.output_dir)
    plot_feature_analysis(args.features_dir, args.output_dir)

    print_summary_table(results)
    print(f"\nPlots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
