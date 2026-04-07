#!/usr/bin/env python3
"""Analysis of the 3-architecture ecovalid augmentation experiment.

Covers 15 single-seed runs: ResNet-50, ConvNeXt-S, ViT-S/16 × 5 norms,
50 epochs, ecovalid augmentation, ImageNet-Ecoset136 dataset. Produces:

  1. Summary table (best val accuracy per arch × norm)
  2. Training curves (val accuracy + loss, one subplot per architecture)
  3. Accuracy heatmap (archs × norms)
  4. Summary bar chart (norms side-by-side per architecture)
  5. Overfitting analysis (train vs val accuracy)
  6. OOD analysis (if ood_results.json or train.log OOD entries are available)

Usage:
    python scripts/analyze_ecovalid_3arch.py
    python scripts/analyze_ecovalid_3arch.py --log_dir local_logs --output_dir results/imagenet_ecoset_ecovalid_3arch
"""
import argparse
import csv
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORMS = ["batchnorm", "layernorm", "groupnorm", "rmsnorm", "derf"]
NORM_LABELS = {
    "batchnorm": "BatchNorm",
    "layernorm": "LayerNorm",
    "groupnorm": "GroupNorm",
    "rmsnorm":   "RMSNorm",
    "derf":      "Derf",
}
NORM_COLORS = {
    "batchnorm": "#1f77b4",
    "layernorm": "#ff7f0e",
    "groupnorm": "#2ca02c",
    "rmsnorm":   "#d62728",
    "derf":      "#9467bd",
}
NORM_STYLES = {
    "batchnorm": "-",
    "layernorm": "--",
    "groupnorm": "-.",
    "rmsnorm":   ":",
    "derf":      (0, (3, 1, 1, 1)),
}

ARCHS = ["resnet50", "convnext_small", "vit_small"]
ARCH_LABELS = {
    "resnet50":       "ResNet-50",
    "convnext_small": "ConvNeXt-S",
    "vit_small":      "ViT-S/16",
}

CHANCE_ACC = 100.0 / 136  # ~0.74%

OOD_KEYS = [
    ("ood/imagenet_r/acc",      "ImageNet-R",  "#e377c2"),
    ("ood/imagenet_a/acc",      "ImageNet-A",  "#bcbd22"),
    ("ood/imagenet_sketch/acc", "Sketch",      "#17becf"),
]
# 4-corruption proxy keys from train.log (written during training)
OOD_LOG_KEYS = {
    "imagenet_r":     "ImageNet-R",
    "imagenet_a":     "ImageNet-A",
    "imagenet_sketch": "Sketch",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_history(path):
    """Load history.csv into a dict of float lists."""
    data = {k: [] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    if not os.path.exists(path):
        return None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data["epoch"].append(int(row["epoch"]))
                data["train_loss"].append(float(row["train_loss"]))
                data["train_acc"].append(float(row["train_acc"]))
                data["val_loss"].append(float(row["val_loss"]))
                data["val_acc"].append(float(row["val_acc"]))
                data["lr"].append(float(row.get("lr", 0) or 0))
            except (ValueError, KeyError):
                continue
    return data if data["epoch"] else None


def load_all(log_dir):
    """Load history for all 15 runs.

    Returns dict[(arch, norm)] -> history, plus a count of loaded runs.
    """
    data = {}
    for arch in ARCHS:
        for norm in NORMS:
            exp = f"{arch}_{norm}_imagenet_ecoset_ecovalid"
            path = os.path.join(log_dir, exp, "history.csv")
            h = load_history(path)
            if h:
                data[(arch, norm)] = h
    return data


def load_ood(log_dir):
    """Load OOD metrics per run.

    Priority: ood_results.json (full eval) > train.log (4-corruption proxy).
    Returns dict[(arch, norm)] -> dict[metric_key -> float]
    """
    ood_pat = re.compile(r"OOD accuracies: (.+)$")
    val_kv  = re.compile(r"([\w_]+): ([\d.]+)%")

    results = {}
    for arch in ARCHS:
        for norm in NORMS:
            exp = f"{arch}_{norm}_imagenet_ecoset_ecovalid"
            run_dir = os.path.join(log_dir, exp)

            # 1. Try ood_results.json
            json_path = os.path.join(run_dir, "ood_results.json")
            if os.path.exists(json_path):
                with open(json_path) as f:
                    results[(arch, norm)] = json.load(f)
                continue

            # 2. Fall back to last OOD line in train.log
            log_path = os.path.join(run_dir, "train.log")
            if not os.path.exists(log_path):
                continue
            last_ood = None
            with open(log_path) as f:
                for line in f:
                    om = ood_pat.search(line)
                    if om:
                        last_ood = {k: float(v) for k, v in val_kv.findall(om.group(1))}
            if last_ood:
                # Re-map log keys to standard ood_results.json style
                mapped = {}
                for k, v in last_ood.items():
                    mapped[f"ood/{k}/acc"] = v
                results[(arch, norm)] = mapped

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_val(history):
    if history is None or not history["val_acc"]:
        return None
    return max(history["val_acc"])


def overfit_gap(history):
    """train_acc − val_acc at the epoch of best val_acc."""
    if history is None or not history["val_acc"]:
        return None
    best_idx = int(np.argmax(history["val_acc"]))
    return history["train_acc"][best_idx] - history["val_acc"][best_idx]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "legend.fontsize":  9,
        "lines.linewidth":  1.8,
    })


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def print_summary_table(data):
    n_loaded = len(data)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY — EcoValid Augmentation, 50 epochs  ({n_loaded}/15 runs loaded)")
    print(f"{'=' * 70}")
    header = f"{'Norm':<12}" + "".join(f"{ARCH_LABELS[a]:>16}" for a in ARCHS)
    print(header)
    print("-" * (12 + 16 * len(ARCHS)))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        for arch in ARCHS:
            h = data.get((arch, norm))
            acc = best_val(h)
            if acc is None:
                row += f"{'—':>16}"
            elif acc < 2.0:
                row += f"{'FAILED':>16}"
            else:
                row += f"{acc:>15.2f}%"
        print(row)
    print()

    print("Overfitting gap (train_acc − val_acc at best-val epoch):")
    print(header)
    print("-" * (12 + 16 * len(ARCHS)))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        for arch in ARCHS:
            h = data.get((arch, norm))
            gap = overfit_gap(h)
            if gap is None:
                row += f"{'—':>16}"
            elif best_val(h) is not None and best_val(h) < 2.0:
                row += f"{'FAILED':>16}"
            else:
                row += f"{gap:>15.1f}%"
        print(row)
    print()


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(data, output_dir):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Training Dynamics — EcoValid Augmentation (ImageNet-Ecoset136, 50 epochs)",
        fontsize=14, fontweight="bold",
    )

    for col, arch in enumerate(ARCHS):
        ax_val  = axes[0, col]
        ax_loss = axes[1, col]

        for norm in NORMS:
            h = data.get((arch, norm))
            if h is None:
                continue
            c  = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            lbl = NORM_LABELS[norm]
            ax_val.plot(h["epoch"],  h["val_acc"],   color=c, linestyle=ls, label=lbl)
            ax_loss.plot(h["epoch"], h["val_loss"],   color=c, linestyle=ls, label=lbl, alpha=0.8)
            ax_loss.plot(h["epoch"], h["train_loss"], color=c, linestyle=ls, alpha=0.35)

        ax_val.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax_val.set_title(ARCH_LABELS[arch])
        if col == 0:
            ax_val.set_ylabel("Val Accuracy (%)")
        ax_val.set_xlabel("Epoch")
        ax_val.legend(loc="upper left", framealpha=0.7)

        ax_loss.set_title(f"{ARCH_LABELS[arch]} — Loss")
        if col == 0:
            ax_loss.set_ylabel("Loss (val solid, train faint)")
        ax_loss.set_xlabel("Epoch")

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Accuracy heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(data, output_dir):
    set_style()
    mat = np.full((len(NORMS), len(ARCHS)), np.nan)
    for i, norm in enumerate(NORMS):
        for j, arch in enumerate(ARCHS):
            acc = best_val(data.get((arch, norm)))
            if acc is not None and acc > 2.0:
                mat[i, j] = acc

    fig, ax = plt.subplots(figsize=(8, 5))
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=85, aspect="auto")
    plt.colorbar(im, ax=ax, label="Best Val Accuracy (%)")
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=10)
    ax.set_yticks(range(len(NORMS)))
    ax.set_yticklabels([NORM_LABELS[n] for n in NORMS])
    ax.set_title("Best Validation Accuracy — EcoValid Augmentation", fontweight="bold")

    for i in range(len(NORMS)):
        for j in range(len(ARCHS)):
            val = mat[i, j]
            txt = f"{val:.1f}%" if not np.isnan(val) else "—"
            color = "black" if (not np.isnan(val) and val > 35) else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color=color)

    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary_bars(data, output_dir):
    set_style()
    n_norms = len(NORMS)
    x = np.arange(len(ARCHS))
    width = 0.15
    offsets = np.linspace(-(n_norms - 1) / 2, (n_norms - 1) / 2, n_norms) * width

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, norm in enumerate(NORMS):
        vals = []
        for arch in ARCHS:
            acc = best_val(data.get((arch, norm)))
            vals.append(acc if (acc and acc > 2.0) else 0.0)
        bars = ax.bar(x + offsets[i], vals, width,
                      label=NORM_LABELS[norm],
                      color=NORM_COLORS[norm], alpha=0.85,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Best Val Accuracy by Architecture and Norm — EcoValid Augmentation",
                 fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(0, 85)
    ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, "summary_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 5. Overfitting
# ---------------------------------------------------------------------------

def plot_overfitting(data, output_dir):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Overfitting: Train vs Val Accuracy — EcoValid Augmentation",
                 fontsize=13, fontweight="bold")

    for col, arch in enumerate(ARCHS):
        ax = axes[col]
        for norm in NORMS:
            h = data.get((arch, norm))
            if h is None or (best_val(h) or 0) < 2.0:
                continue
            c  = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(h["epoch"], h["train_acc"], color=c, linestyle=ls,
                    linewidth=1.5, label=f"{NORM_LABELS[norm]} (train)")
            ax.plot(h["epoch"], h["val_acc"],   color=c, linestyle=ls,
                    linewidth=1.0, alpha=0.55, label=f"{NORM_LABELS[norm]} (val)")

        ax.set_title(ARCH_LABELS[arch])
        ax.set_xlabel("Epoch")
        if col == 0:
            ax.set_ylabel("Accuracy (%)")
        handles = [
            plt.Line2D([0], [0], color=NORM_COLORS[n], linestyle=NORM_STYLES[n],
                       label=NORM_LABELS[n])
            for n in NORMS
            if data.get((arch, n)) and (best_val(data.get((arch, n))) or 0) > 2.0
        ]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "overfitting.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 6. OOD analysis
# ---------------------------------------------------------------------------

def plot_ood_analysis(data, ood, output_dir):
    """OOD accuracy plots — two complementary views.

    1. ood_analysis.png  — one subplot per OOD dataset, grouped by norm,
       3 arch bars per group, shared y-axis so ViT's low numbers are
       visually proportional to ResNet/ConvNeXt.
    2. ood_heatmap.png   — heatmap: rows=norms, cols=archs×datasets,
       makes the pattern of which arch/norm generalises scannable at a glance.
    """
    available_ood_keys = []
    for ood_key, label, color in OOD_KEYS:
        if any(ood.get((arch, norm), {}).get(ood_key) is not None
               for arch in ARCHS for norm in NORMS):
            available_ood_keys.append((ood_key, label, color))

    if not available_ood_keys:
        print("  Skipping OOD plot (no OOD data found)")
        return

    n_ood = len(available_ood_keys)
    n_archs = len(ARCHS)
    arch_colors = ["#4878cf", "#6acc65", "#d65f5f"]   # blue / green / red

    # ------------------------------------------------------------------
    # Plot 1: grouped bars — norm on x-axis, arch as sub-bars, shared y
    # ------------------------------------------------------------------
    set_style()
    fig, axes = plt.subplots(1, n_ood, figsize=(5 * n_ood + 1, 5), sharey=True)
    if n_ood == 1:
        axes = [axes]
    fig.suptitle("OOD Accuracy — EcoValid Augmentation\n"
                 "(shared y-axis across all OOD datasets)",
                 fontsize=12, fontweight="bold")

    # Global y-max across all dataset × arch × norm
    global_max = 0.0
    for ood_key, _, _ in available_ood_keys:
        for arch in ARCHS:
            for norm in NORMS:
                v = ood.get((arch, norm), {}).get(ood_key, 0.0) or 0.0
                global_max = max(global_max, v)
    ylim_top = global_max * 1.18 + 1.0

    width = 0.18
    x = np.arange(len(NORMS))

    for col, (ood_key, ood_label, _) in enumerate(available_ood_keys):
        ax = axes[col]
        for ai, (arch, acolor) in enumerate(zip(ARCHS, arch_colors)):
            vals = [ood.get((arch, norm), {}).get(ood_key, 0.0) or 0.0
                    for norm in NORMS]
            offset = (ai - n_archs / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width=width, color=acolor,
                          alpha=0.85, label=ARCH_LABELS[arch],
                          edgecolor="white", linewidth=0.4)
            for bar, v in zip(bars, vals):
                if v > 0.8:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.4,
                            f"{v:.1f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)

        ax.set_title(ood_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([NORM_LABELS[n] for n in NORMS],
                           rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, ylim_top)
        ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1,
                   alpha=0.6, label="Chance" if col == 0 else "")
        if col == 0:
            ax.set_ylabel("OOD Accuracy (%)")
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = os.path.join(output_dir, "ood_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # ------------------------------------------------------------------
    # Plot 2: heatmap — norms × (arch, dataset), colour = OOD acc
    # ------------------------------------------------------------------
    set_style()
    col_labels = [f"{ARCH_LABELS[a]}\n{lbl}"
                  for _, lbl, _ in available_ood_keys
                  for a in ARCHS]
    mat = np.zeros((len(NORMS), len(col_labels)))
    for ci, (ood_key, _, _) in enumerate(available_ood_keys):
        for ai, arch in enumerate(ARCHS):
            mat_col = ci * n_archs + ai
            for ri, norm in enumerate(NORMS):
                mat[ri, mat_col] = ood.get((arch, norm), {}).get(ood_key, 0.0) or 0.0

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.4), 4))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="OOD Accuracy (%)")

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(len(NORMS)))
    ax.set_yticklabels([NORM_LABELS[n] for n in NORMS], fontsize=9)

    # Dividers between OOD datasets
    for ci in range(1, n_ood):
        ax.axvline(ci * n_archs - 0.5, color="white", linewidth=2)

    for ri in range(len(NORMS)):
        for ci in range(len(col_labels)):
            v = mat[ri, ci]
            color = "white" if v > mat.max() * 0.6 else "black"
            ax.text(ci, ri, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_title("OOD Accuracy Heatmap — EcoValid Augmentation\n"
                 "(grouped by OOD dataset, then architecture)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(output_dir, "ood_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_ood_table(ood):
    """Print OOD summary table per arch × norm."""
    # Collect which keys are available
    all_keys = set()
    for v in ood.values():
        all_keys |= set(v.keys())
    relevant = [(k, lbl, _) for k, lbl, _ in OOD_KEYS if k in all_keys]
    if not relevant:
        print("  No OOD data available.")
        return

    print(f"\n{'=' * 70}")
    print("OOD ACCURACY SUMMARY — EcoValid Augmentation")
    print(f"{'=' * 70}")
    for arch in ARCHS:
        print(f"\n{ARCH_LABELS[arch]}:")
        header = f"  {'Norm':<12}" + "".join(f"{lbl:>14}" for _, lbl, _ in relevant)
        print(header)
        print("  " + "-" * (12 + 14 * len(relevant)))
        for norm in NORMS:
            row = f"  {NORM_LABELS[norm]:<12}"
            d = ood.get((arch, norm), {})
            for ood_key, _, _ in relevant:
                v = d.get(ood_key)
                row += f"{v:>13.1f}%" if v is not None else f"{'—':>14}"
            print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze 3-arch ecovalid augmentation experiment"
    )
    parser.add_argument("--log_dir",    default="logs")
    parser.add_argument("--output_dir", default="results/imagenet_ecoset_ecovalid_3arch")
    args = parser.parse_args()

    log_dir    = args.log_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading runs from: {log_dir}")
    data = load_all(log_dir)
    n_loaded = len(data)
    print(f"Loaded {n_loaded}/15 runs\n")

    for arch in ARCHS:
        for norm in NORMS:
            acc = best_val(data.get((arch, norm)))
            status = f"{acc:.2f}%" if (acc and acc > 2.0) else ("FAILED" if acc is not None else "missing")
            print(f"  {arch}/{norm}: {status}")

    print_summary_table(data)

    print("Generating figures...")
    plot_training_curves(data, output_dir)
    plot_heatmap(data, output_dir)
    plot_summary_bars(data, output_dir)
    plot_overfitting(data, output_dir)

    print("\nLoading OOD data...")
    ood = load_ood(log_dir)
    n_ood = len(ood)
    print(f"  Found OOD data for {n_ood}/15 runs")
    print_ood_table(ood)
    plot_ood_analysis(data, ood, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
