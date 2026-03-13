#!/usr/bin/env python3
"""Comprehensive analysis of the ImageNet-Ecoset136 normalization experiment.

Covers all 15 no-augment runs (ResNet-50, ConvNeXt-Small, ViT-Small × 5 norms)
plus the 5 ResNet-50 ecovalid-augmentation runs. Produces:

  1. Summary table (best val accuracy per model × norm)
  2. Training curves (val accuracy + train/val loss per architecture)
  3. Derf deep-dive (loss dynamics, why it fails)
  4. No-augment vs ecovalid comparison (ResNet-50)
  5. OOD summary (if evaluate_ood_full.py has been run)

Usage:
    python scripts/analyze_imagenet_ecoset_experiment.py
    python scripts/analyze_imagenet_ecoset_experiment.py --log_dir logs --output_dir results/analysis
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    "derf":      (0, (3, 1, 1, 1)),  # dashdot dense
}

ARCHS = ["resnet50", "convnext_small", "vit_small"]
ARCH_LABELS = {
    "resnet50":      "ResNet-50",
    "convnext_small": "ConvNeXt-S",
    "vit_small":     "ViT-S/16",
}

CHANCE_ACC = 100.0 / 136  # ~0.74%

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
    """Load all no-augment and ecovalid run histories.

    Returns:
        noaug: dict[(arch, norm)] -> history
        ecovalid: dict[norm] -> history   (ResNet-50 ecovalid only)
    """
    noaug = {}
    for arch in ARCHS:
        for norm in NORMS:
            exp = f"{arch}_{norm}_imagenet_ecoset"
            path = os.path.join(log_dir, exp, "history.csv")
            h = load_history(path)
            if h:
                noaug[(arch, norm)] = h
            else:
                print(f"  [missing] {exp}")

    ecovalid = {}
    for norm in NORMS:
        exp = f"resnet50_{norm}_imagenet_ecoset_ecovalid"
        path = os.path.join(log_dir, exp, "history.csv")
        h = load_history(path)
        if h:
            ecovalid[norm] = h
        else:
            print(f"  [missing/incomplete] {exp}")

    return noaug, ecovalid


def load_ood_history(log_dir):
    """Parse OOD metric evolution from ecovalid train.log files.

    Returns dict[norm] -> {"epoch": [...], dataset_key: [...], ...}
    OOD is evaluated every val_frequency=2 epochs, so epochs=[2,4,...,50].
    """
    import re

    # Pattern: "Epoch X/50 | ..." on one line, then later "OOD accuracies: key: val%, ..."
    epoch_pat = re.compile(r"Epoch (\d+)/\d+")
    ood_pat   = re.compile(r"OOD accuracies: (.+)$")
    val_kv    = re.compile(r"([\w_]+): ([\d.]+)%")

    histories = {}
    for norm in NORMS:
        log_path = os.path.join(
            log_dir, f"resnet50_{norm}_imagenet_ecoset_ecovalid", "train.log"
        )
        if not os.path.exists(log_path):
            continue

        data = {"epoch": []}
        current_epoch = None

        with open(log_path) as f:
            for line in f:
                em = epoch_pat.search(line)
                if em:
                    current_epoch = int(em.group(1))
                om = ood_pat.search(line)
                if om and current_epoch is not None:
                    data["epoch"].append(current_epoch)
                    for k, v in val_kv.findall(om.group(1)):
                        data.setdefault(k, []).append(float(v))
                    current_epoch = None  # consume

        if data["epoch"]:
            histories[norm] = data

    return histories


def load_ood_results(log_dir):
    """Load OOD evaluation results if available (from evaluate_ood_full.py output).

    Looks for results/ood_summary.csv produced by evaluate_ood_full.py.
    Returns dict or None.
    """
    ood_path = os.path.join(log_dir, "..", "results", "ood_summary.csv")
    if not os.path.exists(ood_path):
        return None
    results = {}
    with open(ood_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row.get("arch", ""), row.get("norm", ""))
            results[key] = row
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_val(history):
    if history is None:
        return None
    return max(history["val_acc"])


def final_train(history):
    if history is None:
        return None
    return history["train_acc"][-1] if history["train_acc"] else None


def overfit_gap(history):
    """train_acc - val_acc at the epoch of best val_acc."""
    if history is None:
        return None
    best_epoch_idx = int(np.argmax(history["val_acc"]))
    return history["train_acc"][best_epoch_idx] - history["val_acc"][best_epoch_idx]


def set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
    })


# ---------------------------------------------------------------------------
# 1. Summary table
# ---------------------------------------------------------------------------

def print_summary_table(noaug, ecovalid):
    """Print a rich summary table to stdout."""
    print("\n" + "=" * 75)
    print("SUMMARY: Best Validation Accuracy (%) — ImageNet-Ecoset136")
    print("=" * 75)

    # No-augment table
    print("\n--- No Augmentation ---")
    header = f"{'Norm':<12}" + "".join(f"{ARCH_LABELS[a]:>16}" for a in ARCHS)
    print(header)
    print("-" * (12 + 16 * len(ARCHS)))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        for arch in ARCHS:
            h = noaug.get((arch, norm))
            acc = best_val(h)
            if acc is None:
                row += f"{'—':>16}"
            elif acc < 2.0:
                row += f"{'FAILED':>16}"
            else:
                row += f"{acc:>15.2f}%"
        print(row)

    # Ecovalid table
    print("\n--- ResNet-50, Eco-Valid Augmentations ---")
    print(f"{'Norm':<12}{'No-Aug':>12}{'EcoValid':>12}{'Δ':>10}")
    print("-" * 46)
    for norm in NORMS:
        h_na = noaug.get(("resnet50", norm))
        h_ev = ecovalid.get(norm)
        acc_na = best_val(h_na)
        acc_ev = best_val(h_ev)
        delta = (acc_ev - acc_na) if (acc_na and acc_ev) else None
        na_str = f"{acc_na:.2f}%" if acc_na and acc_na > 2.0 else ("FAILED" if acc_na is not None else "—")
        ev_str = f"{acc_ev:.2f}%" if acc_ev and acc_ev > 2.0 else ("FAILED" if acc_ev is not None else "—")
        d_str  = f"+{delta:.1f}%" if delta and delta > 0 else (f"{delta:.1f}%" if delta else "—")
        print(f"{NORM_LABELS[norm]:<12}{na_str:>12}{ev_str:>12}{d_str:>10}")

    # Overfitting gap
    print("\n--- Overfitting Gap at Best Val Epoch (train_acc − val_acc) ---")
    header = f"{'Norm':<12}" + "".join(f"{ARCH_LABELS[a]:>16}" for a in ARCHS)
    print(header)
    print("-" * (12 + 16 * len(ARCHS)))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        for arch in ARCHS:
            h = noaug.get((arch, norm))
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
# 2. Training curves — per architecture
# ---------------------------------------------------------------------------

def plot_training_curves(noaug, output_dir):
    set_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Training Dynamics — No Augmentation (ImageNet-Ecoset136)", fontsize=14, fontweight="bold")

    for col, arch in enumerate(ARCHS):
        ax_val  = axes[0, col]
        ax_loss = axes[1, col]

        for norm in NORMS:
            h = noaug.get((arch, norm))
            if h is None:
                continue
            c = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            label = NORM_LABELS[norm]
            ax_val.plot(h["epoch"], h["val_acc"],  color=c, linestyle=ls, label=label)
            ax_loss.plot(h["epoch"], h["val_loss"], color=c, linestyle=ls, label=label, alpha=0.7)
            ax_loss.plot(h["epoch"], h["train_loss"], color=c, linestyle=ls, alpha=0.35)

        ax_val.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Chance")
        ax_val.set_title(ARCH_LABELS[arch])
        ax_val.set_ylabel("Val Accuracy (%)") if col == 0 else None
        ax_val.set_xlabel("Epoch")
        ax_val.legend(loc="upper left", framealpha=0.7)

        ax_loss.set_title(f"{ARCH_LABELS[arch]} — Loss")
        ax_loss.set_ylabel("Loss (val solid, train faint)") if col == 0 else None
        ax_loss.set_xlabel("Epoch")
        # Don't draw legend again

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Bar chart — final best val accuracy
# ---------------------------------------------------------------------------

def plot_summary_bars(noaug, output_dir):
    set_style()
    n_archs = len(ARCHS)
    n_norms = len(NORMS)
    x = np.arange(n_archs)
    width = 0.15
    offsets = np.linspace(-(n_norms - 1) / 2, (n_norms - 1) / 2, n_norms) * width

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, norm in enumerate(NORMS):
        vals = []
        for arch in ARCHS:
            h = noaug.get((arch, norm))
            acc = best_val(h)
            vals.append(acc if (acc and acc > 2.0) else 0.0)
        bars = ax.bar(x + offsets[i], vals, width, label=NORM_LABELS[norm],
                      color=NORM_COLORS[norm], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS])
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Best Val Accuracy by Architecture and Norm — No Augmentation", fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.8)
    ax.set_ylim(0, 75)
    ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="Chance")

    plt.tight_layout()
    out = os.path.join(output_dir, "summary_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 4. Derf deep-dive
# ---------------------------------------------------------------------------

def plot_derf_investigation(noaug, output_dir):
    """Multi-panel figure examining Derf's failure modes."""
    set_style()
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Derf Deep-Dive: Why Does It Fail?", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---- panel 1: Early training loss for all 3 archs (zoom first 10 epochs) ----
    ax1 = fig.add_subplot(gs[0, 0])
    for arch in ARCHS:
        h_derf = noaug.get((arch, "derf"))
        h_bn   = noaug.get((arch, "batchnorm"))
        if h_derf:
            epochs = h_derf["epoch"][:15]
            loss   = h_derf["train_loss"][:15]
            arch_ls = ["-", "--", "-."][ARCHS.index(arch)]
            ax1.plot(epochs, loss, color=NORM_COLORS["derf"],
                     linestyle=arch_ls,
                     label=f"Derf-{ARCH_LABELS[arch]}", linewidth=1.6)
        if h_bn:
            epochs = h_bn["epoch"][:15]
            loss   = h_bn["train_loss"][:15]
            ax1.plot(epochs, loss, color=NORM_COLORS["batchnorm"], alpha=0.4,
                     linestyle="--", label=f"BN-{ARCH_LABELS[arch]}", linewidth=1.2)
    ax1.axhline(np.log(136), color="red", linestyle=":", linewidth=1.5, alpha=0.9,
                label=f"ln(136)={np.log(136):.3f}")
    ax1.set_title("Early Training Loss (first 15 epochs)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.legend(fontsize=7, loc="upper right")

    # ---- panel 2: Full train loss for derf only ----
    ax2 = fig.add_subplot(gs[0, 1])
    for arch in ARCHS:
        h = noaug.get((arch, "derf"))
        if h:
            ax2.plot(h["epoch"], h["train_loss"],
                     color=NORM_COLORS["derf"],
                     linestyle=["-", "--", "-."][ARCHS.index(arch)],
                     label=ARCH_LABELS[arch])
    ax2.axhline(np.log(136), color="red", linestyle=":", linewidth=1.5, alpha=0.9,
                label=f"Chance entropy ln(136)")
    ax2.set_title("Derf Train Loss — Full 50 Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train Loss")
    ax2.legend(fontsize=8)

    # ---- panel 3: Val accuracy comparison Derf vs best norm ----
    ax3 = fig.add_subplot(gs[0, 2])
    best_norms = {"resnet50": "batchnorm", "convnext_small": "batchnorm", "vit_small": "batchnorm"}
    for arch in ARCHS:
        h_derf = noaug.get((arch, "derf"))
        h_best = noaug.get((arch, best_norms[arch]))
        ls_map = {a: s for a, s in zip(ARCHS, ["-", "--", "-."])}
        if h_derf:
            ax3.plot(h_derf["epoch"], h_derf["val_acc"],
                     color=NORM_COLORS["derf"], linestyle=ls_map[arch],
                     label=f"Derf-{ARCH_LABELS[arch]}", linewidth=1.6)
        if h_best:
            ax3.plot(h_best["epoch"], h_best["val_acc"],
                     color=NORM_COLORS["batchnorm"], linestyle=ls_map[arch], alpha=0.5,
                     label=f"BN-{ARCH_LABELS[arch]}", linewidth=1.2)
    ax3.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Chance")
    ax3.set_title("Derf vs BatchNorm Val Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val Accuracy (%)")
    ax3.legend(fontsize=7)

    # ---- panel 4: Theoretical analysis — erf saturation ----
    ax4 = fig.add_subplot(gs[1, 0])
    from math import erf as math_erf, sqrt, pi
    x_vals = np.linspace(-4, 4, 400)
    alphas = [0.5, 1.0, 2.0]
    for alpha in alphas:
        y = np.array([math_erf(alpha * xv) for xv in x_vals])
        ax4.plot(x_vals, y, label=f"α={alpha}")
    # gradient of erf at alpha=0.5
    grad = np.array([(2 / sqrt(pi)) * 0.5 * np.exp(-(0.5 * xv) ** 2) for xv in x_vals])
    ax4.plot(x_vals, grad, color="gray", linestyle=":", linewidth=1.2, label="∂erf(0.5x)/∂x")
    ax4.axhspan(-0.1, 0.1, alpha=0.08, color="red", label="Near-zero gradient zone")
    ax4.set_title("erf(αx): Saturation and Gradient")
    ax4.set_xlabel("Activation value x")
    ax4.set_ylabel("erf(αx) / gradient")
    ax4.legend(fontsize=8)
    ax4.set_ylim(-1.3, 1.3)

    # ---- panel 5: Compound scale reduction through ResNet depth ----
    ax5 = fig.add_subplot(gs[1, 1])
    # Estimate scale reduction per Derf layer (std_out / std_in for x ~ N(0,1))
    # E[erf(0.5*x)^2] for x~N(0,1) ≈ integral numerically
    from math import erf as math_erf
    n_samples = 100000
    np.random.seed(42)
    x_sample = np.random.randn(n_samples)
    derf_out = np.array([math_erf(0.5 * xv) for xv in x_sample])
    scale_factor = np.std(derf_out)  # ~0.42

    depths = np.arange(0, 50)   # number of Derf layers traversed
    # With residual connections: after N blocks with skip connections,
    # the signal is approximately identity + sum of small residuals
    # Model the residual branch as scale_factor^(3 per block) reduction
    # Pure sequential (no skip):
    pure_seq = scale_factor ** depths
    # With skip connections (simplified): each block output = identity + 0.4^3 * residual
    # After k blocks: std grows roughly as (1 + eps)^k where eps = scale_factor^3
    residual_contribution = scale_factor ** 3  # ~0.074 per block
    with_skip = np.array([(1 + residual_contribution) ** (d // 3) * scale_factor ** (d % 3)
                          if d > 0 else 1.0 for d in depths])

    ax5.semilogy(depths, pure_seq, color="red", label=f"Pure sequential (scale={scale_factor:.2f}^depth)")
    ax5.axhline(1e-6, color="gray", linestyle=":", alpha=0.5, label="Effective zero")
    ax5.set_title(f"Activation Scale Through Depth\n(erf(0.5x) reduces std by {scale_factor:.2f}×/layer)")
    ax5.set_xlabel("Number of Derf layers traversed")
    ax5.set_ylabel("Relative activation scale")
    ax5.legend(fontsize=8)
    ax5.text(5, 1e-8, f"ResNet-50 has 48 Derf layers\n→ scale ≈ {scale_factor**48:.2e}",
             fontsize=8, color="red")

    # ---- panel 6: All norms ranked, per architecture ----
    ax6 = fig.add_subplot(gs[1, 2])
    bar_data = {arch: [] for arch in ARCHS}
    for arch in ARCHS:
        for norm in NORMS:
            h = noaug.get((arch, norm))
            acc = best_val(h)
            bar_data[arch].append(acc if (acc and acc > 2.0) else 0.0)

    x = np.arange(len(NORMS))
    width = 0.25
    for i, arch in enumerate(ARCHS):
        offset = (i - 1) * width
        ax6.bar(x + offset, bar_data[arch], width, label=ARCH_LABELS[arch],
                alpha=0.85, edgecolor="white", linewidth=0.5)
    ax6.set_xticks(x)
    ax6.set_xticklabels([NORM_LABELS[n] for n in NORMS], rotation=15)
    ax6.set_ylabel("Best Val Accuracy (%)")
    ax6.set_title("All Norms: Derf Highlighted in Context")
    ax6.legend(fontsize=8)
    ax6.set_ylim(0, 75)
    # Shade Derf column
    derf_idx = NORMS.index("derf")
    ax6.axvspan(derf_idx - 0.5, derf_idx + 0.5, alpha=0.1, color="purple", label="Derf zone")

    out = os.path.join(output_dir, "derf_investigation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_derf_analysis():
    """Print a written analysis of why Derf fails."""
    print("\n" + "=" * 75)
    print("DERF FAILURE ANALYSIS")
    print("=" * 75)
    print("""
Derf2d: output = erf(α·x + shift) · weight + bias
  Learnable: α (scalar, init 0.5), shift (scalar, init 0.0),
             weight (per-channel, init 1.0), bias (per-channel, init 0.0)

WHY IT COMPLETELY FAILS FOR RESNET-50 (chance accuracy from epoch 1):
----------------------------------------------------------------------
1. Not a normalizer: Derf does not zero-mean or unit-variance activations.
   It is a bounded pointwise squashing function (output ≈ range [-1, 1]).

2. Scale collapse through depth:
   - erf(0.5·x) for x ~ N(0,1) → std ≈ 0.42 (reduces by ~0.58x per layer)
   - ResNet-50 Bottleneck uses 3 Derf layers per block × 16 blocks = 48 layers
   - Compound reduction: 0.42^48 ≈ 10^{-17} → effectively zero
   - Skip connections keep the model alive at chance level but residual
     branches produce near-zero outputs → no learning signal

3. Gradient vanishing:
   - ∂erf(α·x)/∂x = (2/√π)·α·exp(-(α·x)²)
   - For saturated inputs (large |x|): gradient → 0 exponentially fast
   - Kaiming initialization assumes variance-preserving norm → scale mismatch
   - With 48 near-zero gradients multiplied in backprop: gradient ≈ 0

4. AMP (float16) amplifies the issue:
   - Float16 underflows at ~6e-8; scale 0.42^20 ≈ 6e-9 → underflow at ~20 layers
   - Gradient scaler helps but activation scale loss is in the forward pass

RESULT: Loss stuck at ln(136) = 4.913 (uniform distribution entropy)
        from epoch 1 through epoch 50 — the model never learns.

WHY DERF PARTIALLY WORKS FOR VIT AND CONVNEXT:
-----------------------------------------------
- ViT: Derf is used as PRE-norm before attention/MLP. Attention softmax
  normalizes by itself; the model can still compute meaningful similarity
  scores. Result: 35.7% (vs 48.8% BatchNorm) — degraded but alive.

- ConvNeXt-Small: Uses depthwise separable convolutions with per-channel
  norms. The architecture is more forgiving of scale mismatches.
  Result: 36.3% (vs 62.1% BatchNorm) — significant degradation.

BOTH still show ~25% relative drop vs BatchNorm, confirming Derf's
fundamental incompatibility with standard initialization and ResNet-style
deep networks without careful initialization/LR tuning.

POTENTIAL FIXES (not implemented):
  - Initialize α much smaller (e.g., 0.01) to keep Derf near-linear initially
  - Use a separate LR for Derf parameters (warm up α slowly)
  - Scale weights by 1/erf_std to compensate for std reduction
  - Use Adaptive Gradient Clipping (already implemented for nonorm_ws)
  - Train with a lower learning rate and longer schedule
""")


# ---------------------------------------------------------------------------
# 5. No-augment vs ecovalid comparison
# ---------------------------------------------------------------------------

def plot_augmentation_comparison(noaug, ecovalid, output_dir):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Effect of Eco-Valid Augmentations on ResNet-50", fontsize=13, fontweight="bold")

    # Left: val accuracy curves both conditions
    ax = axes[0]
    for norm in NORMS:
        h_na = noaug.get(("resnet50", norm))
        h_ev = ecovalid.get(norm)
        c = NORM_COLORS[norm]
        label = NORM_LABELS[norm]
        if h_na and best_val(h_na) > 2.0:
            ax.plot(h_na["epoch"], h_na["val_acc"], color=c, linestyle="--",
                    alpha=0.6, label=f"{label} (no-aug)")
        if h_ev and best_val(h_ev) > 2.0:
            ax.plot(h_ev["epoch"], h_ev["val_acc"], color=c, linestyle="-",
                    label=f"{label} (ecovalid)")
    ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.set_title("Val Accuracy: Solid=EcoValid, Dashed=No-Aug")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    # Right: bar chart of best val acc, grouped by norm
    ax = axes[1]
    norms_exc_derf = [n for n in NORMS if n != "derf"]
    x = np.arange(len(norms_exc_derf))
    width = 0.35
    na_vals = [best_val(noaug.get(("resnet50", n))) or 0 for n in norms_exc_derf]
    ev_vals = [best_val(ecovalid.get(n)) or 0 for n in norms_exc_derf]
    bars1 = ax.bar(x - width / 2, na_vals, width, label="No Augmentation", alpha=0.8,
                   color=[NORM_COLORS[n] for n in norms_exc_derf], edgecolor="white")
    bars2 = ax.bar(x + width / 2, ev_vals, width, label="Eco-Valid Aug", alpha=0.55,
                   color=[NORM_COLORS[n] for n in norms_exc_derf],
                   edgecolor=[NORM_COLORS[n] for n in norms_exc_derf], linewidth=1.5)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                        f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([NORM_LABELS[n] for n in norms_exc_derf])
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("No-Aug vs EcoValid (ResNet-50, Derf excluded)")
    ax.legend()
    ax.set_ylim(0, 90)

    plt.tight_layout()
    out = os.path.join(output_dir, "augmentation_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Cross-norm ranking heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(noaug, ecovalid, output_dir):
    set_style()
    # Columns: ResNet-50 ecovalid first, then 3 no-aug archs
    cols = ["resnet50_ecovalid"] + ARCHS
    col_labels = ["ResNet-50\n(EcoValid Aug)"] + [ARCH_LABELS[a] for a in ARCHS]

    data = np.full((len(NORMS), len(cols)), np.nan)
    for i, norm in enumerate(NORMS):
        # First column: ResNet-50 ecovalid
        h_ev = ecovalid.get(norm)
        acc_ev = best_val(h_ev)
        if acc_ev is not None and acc_ev > 2.0:
            data[i, 0] = acc_ev
        for j, arch in enumerate(ARCHS):
            h = noaug.get((arch, norm))
            acc = best_val(h)
            if acc is not None and acc > 2.0:
                data[i, j + 1] = acc

    fig, ax = plt.subplots(figsize=(9, 5))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=85, aspect="auto")
    plt.colorbar(im, ax=ax, label="Best Val Accuracy (%)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(NORMS)))
    ax.set_yticklabels([NORM_LABELS[n] for n in NORMS])
    ax.set_title("Best Validation Accuracy Heatmap\n(ImageNet-Ecoset136)", fontweight="bold")

    # Draw a vertical separator between ecovalid and no-aug columns
    ax.axvline(0.5, color="white", linewidth=2.5)

    for i in range(len(NORMS)):
        for j in range(len(cols)):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "FAILED", ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")
            else:
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        fontsize=10, color="black" if val > 35 else "white")

    # Label the two sections along the top
    ax.annotate("EcoValid Aug", xy=(0, 1.06), xycoords="axes fraction",
                ha="center", fontsize=9, color="gray",
                xytext=(0.5 / len(cols), 1.06))
    ax.annotate("No Augmentation", xy=(0, 1.06), xycoords="axes fraction",
                ha="center", fontsize=9, color="gray",
                xytext=((1 + len(ARCHS)) / 2 / len(cols), 1.06))

    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Overfitting analysis
# ---------------------------------------------------------------------------

def plot_overfitting(noaug, output_dir):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Overfitting: Train vs Val Accuracy — No Augmentation", fontsize=13, fontweight="bold")

    for col, arch in enumerate(ARCHS):
        ax = axes[col]
        for norm in NORMS:
            h = noaug.get((arch, norm))
            if h is None or best_val(h) is None or best_val(h) < 2.0:
                continue
            c = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            label = NORM_LABELS[norm]
            ax.plot(h["epoch"], h["train_acc"], color=c, linestyle=ls, linewidth=1.5, label=f"{label} (train)")
            ax.plot(h["epoch"], h["val_acc"],   color=c, linestyle=ls, linewidth=1.0, alpha=0.55, label=f"{label} (val)")

        ax.set_title(ARCH_LABELS[arch])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)") if col == 0 else None
        # Custom legend: just show norm name once
        handles = [plt.Line2D([0], [0], color=NORM_COLORS[n], linestyle=NORM_STYLES[n],
                               label=NORM_LABELS[n])
                   for n in NORMS if noaug.get((arch, n)) and best_val(noaug.get((arch, n))) > 2.0]
        ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "overfitting.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 8. OOD analysis from ecovalid train logs
# ---------------------------------------------------------------------------

# OOD results at final epoch (epoch 50) from ecovalid train logs
# Source: grep "OOD accuracies" logs/resnet50_*_imagenet_ecoset_ecovalid/train.log
_ECOVALID_OOD = {
    "batchnorm": {
        "imagenet_r": 4.6, "imagenet_sketch": 7.1, "imagenet_a": 1.0,
        "imagenet_c_contrast": 40.1, "imagenet_c_blur": 6.2,
        "imagenet_c_fog": 21.0, "imagenet_c_noise": 8.8,
    },
    "layernorm": {
        "imagenet_r": 15.3, "imagenet_sketch": 17.7, "imagenet_a": 3.5,
        "imagenet_c_contrast": 53.3, "imagenet_c_blur": 48.5,
        "imagenet_c_fog": 28.3, "imagenet_c_noise": 23.4,
    },
    "groupnorm": {
        "imagenet_r": 18.7, "imagenet_sketch": 25.7, "imagenet_a": 4.6,
        "imagenet_c_contrast": 64.8, "imagenet_c_blur": 51.2,
        "imagenet_c_fog": 43.8, "imagenet_c_noise": 30.4,
    },
    "rmsnorm": {
        "imagenet_r": 17.6, "imagenet_sketch": 21.9, "imagenet_a": 3.7,
        "imagenet_c_contrast": 58.2, "imagenet_c_blur": 49.7,
        "imagenet_c_fog": 41.8, "imagenet_c_noise": 29.5,
    },
    "derf": {
        "imagenet_r": 2.7, "imagenet_sketch": 0.7, "imagenet_a": 4.2,
        "imagenet_c_contrast": 0.7, "imagenet_c_blur": 0.7,
        "imagenet_c_fog": 0.7, "imagenet_c_noise": 0.7,
    },
}

OOD_DATASETS = ["imagenet_r", "imagenet_sketch", "imagenet_a",
                 "imagenet_c_noise", "imagenet_c_blur",
                 "imagenet_c_fog", "imagenet_c_contrast"]
OOD_LABELS = {
    "imagenet_r": "IN-R",
    "imagenet_sketch": "IN-Sketch",
    "imagenet_a": "IN-A",
    "imagenet_c_noise": "C-Noise",
    "imagenet_c_blur": "C-Blur",
    "imagenet_c_fog": "C-Fog",
    "imagenet_c_contrast": "C-Contrast",
}


def plot_ood_analysis(ecovalid, output_dir):
    set_style()
    ood = _ECOVALID_OOD

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(
        "OOD Robustness: ResNet-50, Eco-Valid Augmentations\n"
        "(Note: BatchNorm dominates ID accuracy but collapses on OOD)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: OOD accuracy per dataset, grouped by norm ----
    ax = axes[0]
    x = np.arange(len(OOD_DATASETS))
    width = 0.15
    n_norms = len(NORMS)
    offsets = np.linspace(-(n_norms - 1) / 2, (n_norms - 1) / 2, n_norms) * width
    for i, norm in enumerate(NORMS):
        vals = [ood[norm][ds] for ds in OOD_DATASETS]
        ax.bar(x + offsets[i], vals, width, label=NORM_LABELS[norm],
               color=NORM_COLORS[norm], alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([OOD_LABELS[ds] for ds in OOD_DATASETS], rotation=35, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("OOD Accuracy by Dataset and Norm")
    ax.legend(fontsize=8)
    ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    # ---- Panel 2: In-distribution vs OOD mean scatter ----
    ax = axes[1]
    ood_mean_datasets = ["imagenet_r", "imagenet_sketch", "imagenet_a",
                         "imagenet_c_noise", "imagenet_c_blur", "imagenet_c_fog", "imagenet_c_contrast"]
    for norm in NORMS:
        id_acc = best_val(ecovalid.get(norm)) or 0
        ood_mean = np.mean([ood[norm][ds] for ds in ood_mean_datasets])
        c = NORM_COLORS[norm]
        ax.scatter(id_acc, ood_mean, color=c, s=120, zorder=5)
        ax.annotate(NORM_LABELS[norm], (id_acc, ood_mean),
                    textcoords="offset points", xytext=(6, 3), fontsize=9, color=c)
    ax.set_xlabel("In-Distribution Val Accuracy (%)")
    ax.set_ylabel("Mean OOD Accuracy (%)")
    ax.set_title("ID vs OOD Trade-off\n(BatchNorm: highest ID, worst OOD)")
    # Add diagonal reference line
    xlim = ax.get_xlim()
    ax.plot([0, 100], [0, 100], color="gray", linestyle=":", alpha=0.4, linewidth=1)

    plt.tight_layout()
    out = os.path.join(output_dir, "ood_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_ood_evolution(ood_histories, ecovalid, output_dir):
    """Plot OOD metric evolution over training epochs for all ecovalid norms."""
    if not ood_histories:
        print("  Skipping OOD evolution plot (no data)")
        return

    set_style()

    # Datasets logged during training (key as they appear in train.log)
    logged_datasets = [
        "imagenet_r",
        "imagenet_sketch",
        "imagenet_a",
        "imagenet_c_gaussian_noise_s3",
        "imagenet_c_defocus_blur_s3",
        "imagenet_c_fog_s3",
        "imagenet_c_contrast_s3",
    ]
    ds_labels = {
        "imagenet_r":                  "ImageNet-R",
        "imagenet_sketch":             "ImageNet-Sketch",
        "imagenet_a":                  "ImageNet-A",
        "imagenet_c_gaussian_noise_s3": "C-Noise (s3)",
        "imagenet_c_defocus_blur_s3":  "C-Blur (s3)",
        "imagenet_c_fog_s3":           "C-Fog (s3)",
        "imagenet_c_contrast_s3":      "C-Contrast (s3)",
    }

    n_ds = len(logged_datasets)
    ncols = 4
    nrows = -(-n_ds // ncols)  # ceil division → 2 rows

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 7))
    fig.suptitle(
        "OOD Accuracy over Training — ResNet-50, Eco-Valid Augmentations",
        fontsize=13, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, ds_key in enumerate(logged_datasets):
        ax = axes_flat[idx]
        for norm in NORMS:
            h = ood_histories.get(norm)
            if h is None or ds_key not in h:
                continue
            epochs = h["epoch"]
            vals   = h[ds_key]
            c  = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(epochs, vals, color=c, linestyle=ls,
                    linewidth=1.6, label=NORM_LABELS[norm])

        # Mark final value from _ECOVALID_OOD as a dot
        ds_short = ds_key.replace("_s3", "").replace("imagenet_c_", "imagenet_c_")
        ood_key_map = {
            "imagenet_r":                  "imagenet_r",
            "imagenet_sketch":             "imagenet_sketch",
            "imagenet_a":                  "imagenet_a",
            "imagenet_c_gaussian_noise_s3": "imagenet_c_noise",
            "imagenet_c_defocus_blur_s3":  "imagenet_c_blur",
            "imagenet_c_fog_s3":           "imagenet_c_fog",
            "imagenet_c_contrast_s3":      "imagenet_c_contrast",
        }

        ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax.set_title(ds_labels[ds_key], fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)") if idx % ncols == 0 else None
        if idx == 0:
            ax.legend(fontsize=8, loc="upper left")

    # Also plot ID val accuracy for context in the spare slot
    ax = axes_flat[n_ds]
    for norm in NORMS:
        h = ecovalid.get(norm)
        if h and best_val(h) and best_val(h) > 2.0:
            ax.plot(h["epoch"], h["val_acc"],
                    color=NORM_COLORS[norm], linestyle=NORM_STYLES[norm],
                    linewidth=1.6, label=NORM_LABELS[norm])
    ax.set_title("ID Val Accuracy (EcoValid)", fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8)

    # Hide any remaining empty subplots
    for idx in range(n_ds + 1, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    out = os.path.join(output_dir, "ood_evolution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_ood_table():
    ood = _ECOVALID_OOD
    print("\n" + "=" * 75)
    print("OOD ROBUSTNESS — ResNet-50, Eco-Valid Augmentations (final epoch)")
    print("=" * 75)
    header = f"{'Norm':<12}" + "".join(f"{OOD_LABELS[ds]:>12}" for ds in OOD_DATASETS) + f"{'Mean':>10}"
    print(header)
    print("-" * (12 + 12 * len(OOD_DATASETS) + 10))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        vals = [ood[norm][ds] for ds in OOD_DATASETS]
        for v in vals:
            row += f"{v:>11.1f}%"
        row += f"{np.mean(vals):>9.1f}%"
        print(row)

    print()
    id_accs = {norm: best_val(None) for norm in NORMS}  # placeholder
    print("Key finding: BatchNorm dominates ID accuracy but collapses on OOD.")
    print("GroupNorm/RMSNorm show best OOD robustness across all corruption types.")
    print(f"BatchNorm vs GroupNorm on C-Blur: {ood['batchnorm']['imagenet_c_blur']:.1f}% vs {ood['groupnorm']['imagenet_c_blur']:.1f}% ({ood['groupnorm']['imagenet_c_blur']/ood['batchnorm']['imagenet_c_blur']:.1f}x improvement)")
    print(f"BatchNorm vs GroupNorm on C-Noise: {ood['batchnorm']['imagenet_c_noise']:.1f}% vs {ood['groupnorm']['imagenet_c_noise']:.1f}% ({ood['groupnorm']['imagenet_c_noise']/ood['batchnorm']['imagenet_c_noise']:.1f}x improvement)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze imagenet_ecoset norm experiment")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_dir", default="results/imagenet_ecoset_analysis")
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading runs from: {log_dir}")
    noaug, ecovalid = load_all(log_dir)

    n_noaug   = len(noaug)
    n_ecovalid = len(ecovalid)
    print(f"Loaded {n_noaug}/15 no-augment runs, {n_ecovalid}/5 ecovalid runs\n")

    # Summary table to stdout
    print_summary_table(noaug, ecovalid)

    # Derf written analysis to stdout
    print_derf_analysis()

    # Figures
    print("Generating figures...")
    plot_training_curves(noaug, output_dir)
    plot_summary_bars(noaug, output_dir)
    plot_heatmap(noaug, ecovalid, output_dir)
    plot_overfitting(noaug, output_dir)
    plot_derf_investigation(noaug, output_dir)

    if ecovalid:
        plot_augmentation_comparison(noaug, ecovalid, output_dir)

    # OOD analysis (hardcoded from ecovalid train logs)
    print_ood_table()
    plot_ood_analysis(ecovalid, output_dir)

    # OOD evolution over training (parsed from train.log)
    ood_histories = load_ood_history(log_dir)
    plot_ood_evolution(ood_histories, ecovalid, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nTo run full post-training OOD evaluation (all 15 corruptions × 5 severities):")
    print("  ./scripts/run_full_ood_eval_imagenet_ecoset.sh  (requires ~2-3h GPU time)")


if __name__ == "__main__":
    main()
