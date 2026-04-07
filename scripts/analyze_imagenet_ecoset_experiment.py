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

NORMS = ["batchnorm", "layernorm", "groupnorm", "rmsnorm", "derf_stable", "localnorm"]
NORM_LABELS = {
    "batchnorm":    "BatchNorm",
    "layernorm":    "LayerNorm",
    "groupnorm":    "GroupNorm",
    "rmsnorm":      "RMSNorm",
    "derf_stable":  "Derf",
    "localnorm":    "LocalNorm",
}
NORM_COLORS = {
    "batchnorm":    "#1f77b4",
    "layernorm":    "#ff7f0e",
    "groupnorm":    "#2ca02c",
    "rmsnorm":      "#d62728",
    "derf_stable":  "#9467bd",
    "localnorm":    "#8c564b",
}
NORM_STYLES = {
    "batchnorm":    "-",
    "layernorm":    "--",
    "groupnorm":    "-.",
    "rmsnorm":      ":",
    "derf_stable":  (0, (3, 1, 1, 1)),  # dashdot dense
    "localnorm":    (0, (5, 2)),         # long dash
}

SEEDS = [42, 43, 44, 45, 46]

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


def average_histories(histories):
    """Average a list of history dicts element-wise."""
    if not histories:
        return None
    min_len = min(len(h["epoch"]) for h in histories)
    keys = [k for k in histories[0] if k != "epoch"]
    avg = {"epoch": histories[0]["epoch"][:min_len]}
    for k in keys:
        # Per-key min length in case some seeds logged fewer OOD entries
        k_len = min(len(h.get(k, [])) for h in histories)
        use_len = min(min_len, k_len)
        avg[k] = [
            float(np.mean([h[k][i] for h in histories if k in h and len(h[k]) > i]))
            for i in range(use_len)
        ]
    return avg


def mean_std_histories(histories):
    """Return (mean_history, std_history) from a list of seed histories."""
    if not histories:
        return None, None
    min_len = min(len(h["epoch"]) for h in histories)
    keys = [k for k in histories[0] if k != "epoch"]
    mean_h = {"epoch": histories[0]["epoch"][:min_len]}
    std_h  = {"epoch": histories[0]["epoch"][:min_len]}
    for k in keys:
        k_len = min(len(h.get(k, [])) for h in histories)
        use_len = min(min_len, k_len)
        for i in range(use_len):
            vals = [h[k][i] for h in histories if k in h and len(h[k]) > i]
            mean_h.setdefault(k, []).append(float(np.mean(vals)))
            std_h.setdefault(k, []).append(float(np.std(vals)))
    return mean_h, std_h


def load_all(log_dir):
    """Load all no-augment and ecovalid run histories.

    Returns:
        noaug: dict[(arch, norm)] -> history   (empty if no single-run logs exist)
        ecovalid: dict[norm] -> history         (seed-averaged ResNet-50 ecovalid)
        ecovalid_std: dict[norm] -> std_history  (per-epoch std across seeds)
    """
    noaug = {}
    for arch in ARCHS:
        for norm in NORMS:
            exp = f"{arch}_{norm}_imagenet_ecoset"
            path = os.path.join(log_dir, exp, "history.csv")
            h = load_history(path)
            if h:
                noaug[(arch, norm)] = h

    ecovalid = {}
    ecovalid_std = {}
    for norm in NORMS:
        # Try seed-averaged runs first
        seed_histories = []
        for seed in SEEDS:
            exp = f"resnet50_{norm}_imagenet_ecoset_ecovalid_seed{seed}"
            path = os.path.join(log_dir, exp, "history.csv")
            h = load_history(path)
            if h:
                seed_histories.append(h)
        if seed_histories:
            mean_h, std_h = mean_std_histories(seed_histories)
            ecovalid[norm] = mean_h
            ecovalid_std[norm] = std_h
            print(f"  Loaded {len(seed_histories)} seeds for ecovalid/{norm}")
        else:
            # Fall back to single non-seeded run
            exp = f"resnet50_{norm}_imagenet_ecoset_ecovalid"
            path = os.path.join(log_dir, exp, "history.csv")
            h = load_history(path)
            if h:
                ecovalid[norm] = h
            else:
                print(f"  [missing/incomplete] ecovalid/{norm}")

    return noaug, ecovalid, ecovalid_std


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

    def _parse_log(log_path):
        # Collect all (epoch, ood_dict) pairs, then keep only the LAST
        # occurrence of each epoch (handles crash + resume logs).
        entries = []
        current_epoch = None
        with open(log_path) as f:
            for line in f:
                em = epoch_pat.search(line)
                if em:
                    current_epoch = int(em.group(1))
                om = ood_pat.search(line)
                if om and current_epoch is not None:
                    kvs = {k: float(v) for k, v in val_kv.findall(om.group(1))}
                    entries.append((current_epoch, kvs))
                    current_epoch = None

        if not entries:
            return None

        # Keep only the last entry for each epoch (from the final run)
        last_by_epoch = {}
        for ep, kvs in entries:
            last_by_epoch[ep] = kvs

        data = {"epoch": []}
        for ep in sorted(last_by_epoch):
            data["epoch"].append(ep)
            for k, v in last_by_epoch[ep].items():
                data.setdefault(k, []).append(v)

        return data if data["epoch"] else None

    histories = {}
    histories_std = {}
    for norm in NORMS:
        seed_datas = []
        for seed in SEEDS:
            log_path = os.path.join(
                log_dir, f"resnet50_{norm}_imagenet_ecoset_ecovalid_seed{seed}", "train.log"
            )
            if os.path.exists(log_path):
                d = _parse_log(log_path)
                if d:
                    seed_datas.append(d)
        if seed_datas:
            mean_h, std_h = mean_std_histories(seed_datas)
            histories[norm] = mean_h
            histories_std[norm] = std_h
        else:
            # Fall back to single run
            log_path = os.path.join(
                log_dir, f"resnet50_{norm}_imagenet_ecoset_ecovalid", "train.log"
            )
            if os.path.exists(log_path):
                d = _parse_log(log_path)
                if d:
                    histories[norm] = d

    return histories, histories_std


def load_ood_final(log_dir):
    """Load final-epoch OOD metrics from train.log files, seed-averaged.

    Returns dict[norm] -> dict[metric -> (mean, std)]
    Metrics: imagenet_r, imagenet_a, imagenet_sketch,
             imagenet_c_gaussian_noise_s3, imagenet_c_defocus_blur_s3,
             imagenet_c_fog_s3, imagenet_c_contrast_s3
    """
    import re
    ood_pat = re.compile(r"OOD accuracies: (.+)$")
    val_kv  = re.compile(r"([\w_]+): ([\d.]+)%")

    def _parse_final(log_path):
        last_ood = None
        with open(log_path) as f:
            for line in f:
                om = ood_pat.search(line)
                if om:
                    last_ood = {k: float(v) for k, v in val_kv.findall(om.group(1))}
        return last_ood

    results = {}
    for norm in NORMS:
        seed_vals = []
        for seed in SEEDS:
            log_path = os.path.join(
                log_dir, f"resnet50_{norm}_imagenet_ecoset_ecovalid_seed{seed}", "train.log"
            )
            if os.path.exists(log_path):
                d = _parse_final(log_path)
                if d:
                    seed_vals.append(d)
        if seed_vals:
            all_keys = set().union(*[set(d.keys()) for d in seed_vals])
            results[norm] = {
                k: (
                    float(np.mean([d[k] for d in seed_vals if k in d])),
                    float(np.std( [d[k] for d in seed_vals if k in d]))
                )
                for k in all_keys
            }
    return results


def load_imagenet_c_full(log_dir):
    """Load full imagenet-c mCE from ood_results.json (if available), seed-averaged.

    Returns dict[norm] -> dict[metric -> (mean, std)], or {} if not yet run.
    """
    import json
    results = {}
    for norm in NORMS:
        seed_vals = []
        for seed in SEEDS:
            json_path = os.path.join(
                log_dir, f"resnet50_{norm}_imagenet_ecoset_ecovalid_seed{seed}", "ood_results.json"
            )
            if os.path.exists(json_path):
                with open(json_path) as f:
                    seed_vals.append(json.load(f))
        if seed_vals:
            all_keys = set().union(*[set(d.keys()) for d in seed_vals])
            results[norm] = {
                k: (
                    float(np.mean([d[k] for d in seed_vals if k in d])),
                    float(np.std( [d[k] for d in seed_vals if k in d]))
                )
                for k in all_keys
            }
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
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
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

def plot_training_curves(ecovalid, ecovalid_std, output_dir):
    """Training curves for ecovalid 5-seed runs with mean ± std bands."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Training Dynamics — ResNet-50, Eco-Valid (5 seeds, mean ± 1 std)",
        fontsize=13, fontweight="bold",
    )

    panels = [
        ("val_acc", "Val Accuracy (%)", "Validation Accuracy"),
        ("train_loss", "Loss", "Training Loss"),
        ("val_loss", "Loss", "Validation Loss"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for norm in NORMS:
            h = ecovalid.get(norm)
            s = ecovalid_std.get(norm)
            if h is None or key not in h:
                continue
            epochs = np.array(h["epoch"][:len(h[key])])
            mean = np.array(h[key])
            c = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(epochs, mean, color=c, linestyle=ls, linewidth=1.8,
                    label=NORM_LABELS[norm])
            if s and key in s:
                std = np.array(s[key][:len(mean)])
                ax.fill_between(epochs, mean - std, mean + std,
                                color=c, alpha=0.15)

        if key == "val_acc":
            ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 3. Bar chart — final best val accuracy
# ---------------------------------------------------------------------------

def plot_summary_bars(ecovalid, ecovalid_std, output_dir):
    """Bar chart of best val accuracy per norm with std error bars (ecovalid 5-seed)."""
    set_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    norms_valid, means, stds = [], [], []
    for norm in NORMS:
        h = ecovalid.get(norm)
        s = ecovalid_std.get(norm)
        if h is None:
            continue
        acc = best_val(h)
        if acc is None or acc < 2.0:
            continue
        best_idx = int(np.argmax(h["val_acc"]))
        std_val = s["val_acc"][best_idx] if s and "val_acc" in s and best_idx < len(s["val_acc"]) else 0.0
        norms_valid.append(norm)
        means.append(acc)
        stds.append(std_val)

    x = np.arange(len(norms_valid))
    bars = ax.bar(x, means, color=[NORM_COLORS[n] for n in norms_valid],
                  alpha=0.85, edgecolor="white", linewidth=0.5,
                  yerr=stds, capsize=5, error_kw={"linewidth": 1.2})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.8,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([NORM_LABELS[n] for n in norms_valid], fontsize=11)
    ax.set_ylabel("Best Validation Accuracy (%)")
    ax.set_title("Best Val Accuracy — ResNet-50, Eco-Valid (5 seeds, mean ± std)", fontweight="bold")
    ax.set_ylim(0, max(means) + 10)
    ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, "summary_bars.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")




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
    norms_exc_derf = [n for n in NORMS if "derf" not in n]
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
# 5b. Ecovalid training progression with confidence bands
# ---------------------------------------------------------------------------

def plot_ecovalid_progression(ecovalid, ecovalid_std, output_dir):
    """Training progression for ecovalid runs with mean ± std bands."""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Training Progression — ResNet-50, Eco-Valid (5 seeds, mean ± 1 std)",
        fontsize=13, fontweight="bold",
    )

    panels = [
        ("val_acc", "Val Accuracy (%)", "Validation Accuracy"),
        ("train_loss", "Loss", "Training Loss"),
        ("val_loss", "Loss", "Validation Loss"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for norm in NORMS:
            h = ecovalid.get(norm)
            s = ecovalid_std.get(norm)
            if h is None or key not in h:
                continue
            if key == "val_acc" and best_val(h) is not None and best_val(h) < 2.0:
                continue
            epochs = np.array(h["epoch"][:len(h[key])])
            mean = np.array(h[key])
            c = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(epochs, mean, color=c, linestyle=ls, linewidth=1.8,
                    label=NORM_LABELS[norm])
            if s and key in s:
                std = np.array(s[key][:len(mean)])
                ax.fill_between(epochs, mean - std, mean + std,
                                color=c, alpha=0.15)

        if key == "val_acc":
            ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "ecovalid_progression.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 5c. ID vs OOD trajectory during training
# ---------------------------------------------------------------------------

def _smooth(arr, window=3):
    """Simple moving average for smoothing noisy trajectories."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    # Pad edges to preserve length
    padded = np.concatenate([arr[:1]] * (window // 2) + [arr] + [arr[-1:]] * (window // 2))
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def plot_id_ood_trajectory(ecovalid, ood_histories, ood_histories_std, output_dir):
    """Parametric trajectory: x=ID val acc, y=OOD acc, over training epochs.

    Shows how OOD robustness tracks (or diverges from) ID accuracy.
    Only uses C-corruption data (available at all epochs) to avoid
    discontinuities when standard OOD datasets appear partway through.
    """
    if not ood_histories:
        print("  Skipping ID-OOD trajectory (no OOD history data)")
        return

    set_style()

    c_keys = [
        "imagenet_c_gaussian_noise_s3", "imagenet_c_defocus_blur_s3",
        "imagenet_c_fog_s3", "imagenet_c_contrast_s3",
    ]
    # Standard OOD only available from mid-training; separate panel
    std_ood_keys = ["imagenet_r", "imagenet_a", "imagenet_sketch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "ID vs OOD Trajectory During Training — ResNet-50, Eco-Valid (5 seeds)",
        fontsize=13, fontweight="bold",
    )

    for ax, (ds_keys, ds_label, smooth_w) in zip(axes, [
        (c_keys, "Mean ImageNet-C (sev=3)", 5),
        (std_ood_keys, "Mean Standard OOD (R + A + Sketch)", 3),
    ]):
        for norm in NORMS:
            h_ood = ood_histories.get(norm)
            h_id  = ecovalid.get(norm)
            if h_ood is None or h_id is None:
                continue
            if best_val(h_id) is not None and best_val(h_id) < 2.0:
                continue

            ood_epochs = h_ood["epoch"]
            avail = [k for k in ds_keys if k in h_ood]
            if not avail:
                continue
            min_ood_len = min(len(h_ood[k]) for k in avail)
            ood_mean_raw = np.array([
                np.mean([h_ood[k][i] for k in avail])
                for i in range(min_ood_len)
            ])
            ood_ep = ood_epochs[:min_ood_len]

            # Skip very early epochs where both ID and OOD are near chance
            start_idx = 0
            id_epochs = np.array(h_id["epoch"])
            id_vals   = np.array(h_id["val_acc"])
            id_at_ood_raw = np.array([
                id_vals[np.argmin(np.abs(id_epochs - e))]
                for e in ood_ep
            ])
            # Only start when ID > 5% (past initial noise)
            for si in range(len(id_at_ood_raw)):
                if id_at_ood_raw[si] > 5.0:
                    start_idx = si
                    break

            id_at_ood = id_at_ood_raw[start_idx:]
            ood_mean = ood_mean_raw[start_idx:]
            if len(id_at_ood) < 3:
                continue

            # Smooth to reduce epoch-to-epoch noise
            ood_smooth = _smooth(ood_mean, smooth_w)
            id_smooth  = _smooth(id_at_ood, smooth_w)

            c = NORM_COLORS[norm]
            ax.plot(id_smooth, ood_smooth, color=c, linewidth=2.0,
                    label=NORM_LABELS[norm], zorder=3)
            # Start marker (circle) and end marker (star)
            ax.scatter(id_smooth[0], ood_smooth[0], color=c, s=50,
                       marker="o", zorder=5, edgecolors="white", linewidth=0.8)
            ax.scatter(id_smooth[-1], ood_smooth[-1], color=c, s=120,
                       marker="*", zorder=5, edgecolors="white", linewidth=0.8)
            # Label final point with norm name
            ax.annotate(NORM_LABELS[norm],
                        (id_smooth[-1], ood_smooth[-1]),
                        textcoords="offset points", xytext=(6, 2),
                        fontsize=8, color=c, fontweight="bold")

        # Diagonal reference
        ax.plot([0, 100], [0, 100], color="gray", linestyle=":", alpha=0.3, linewidth=1)
        ax.set_xlabel("ID Val Accuracy (%)")
        ax.set_ylabel(f"{ds_label} (%)")
        ax.set_title(f"{ds_label}\n(o = start, * = end of training)")
        ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = os.path.join(output_dir, "id_ood_trajectory.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 5d. Relative robustness over training
# ---------------------------------------------------------------------------

def plot_robustness_ratio(ecovalid, ood_histories, ood_histories_std, output_dir):
    """Plot OOD/ID ratio over training epochs — shows if robustness keeps pace.

    Starts from epoch 5 (after initial warm-up noise) and uses the
    robustness gap (ID - OOD) in percentage points, which is more
    interpretable than the raw ratio.
    """
    if not ood_histories:
        print("  Skipping robustness ratio (no OOD history data)")
        return

    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Robustness Gap Over Training — (ID Acc - OOD Acc) in pp",
        fontsize=13, fontweight="bold",
    )

    c_keys = [
        "imagenet_c_gaussian_noise_s3", "imagenet_c_defocus_blur_s3",
        "imagenet_c_fog_s3", "imagenet_c_contrast_s3",
    ]
    std_ood_keys = ["imagenet_r", "imagenet_a", "imagenet_sketch"]
    MIN_EPOCH = 4  # skip early chaos

    for ax, (ds_keys, ds_label) in zip(axes, [
        (c_keys, "ID - Mean C-Corruption"),
        (std_ood_keys, "ID - Mean Standard OOD"),
    ]):
        for norm in NORMS:
            h_ood = ood_histories.get(norm)
            h_id  = ecovalid.get(norm)
            if h_ood is None or h_id is None:
                continue
            if best_val(h_id) is not None and best_val(h_id) < 2.0:
                continue

            ood_epochs = h_ood["epoch"]
            avail = [k for k in ds_keys if k in h_ood]
            if not avail:
                continue
            min_ood_len = min(len(h_ood[k]) for k in avail)
            ood_mean = np.array([
                np.mean([h_ood[k][i] for k in avail])
                for i in range(min_ood_len)
            ])
            ood_ep = np.array(ood_epochs[:min_ood_len])

            id_epochs = np.array(h_id["epoch"])
            id_vals   = np.array(h_id["val_acc"])
            id_at_ood = np.array([
                id_vals[np.argmin(np.abs(id_epochs - e))]
                for e in ood_ep
            ])

            # Filter to epochs >= MIN_EPOCH
            mask = ood_ep >= MIN_EPOCH
            if mask.sum() < 2:
                continue
            gap = id_at_ood[mask] - ood_mean[mask]
            gap_smooth = _smooth(gap, 3)

            c  = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(ood_ep[mask], gap_smooth, color=c, linestyle=ls,
                    linewidth=1.8, label=NORM_LABELS[norm])

            # Std band
            s_ood = ood_histories_std.get(norm)
            if s_ood:
                avail_s = [k for k in ds_keys if k in s_ood]
                if avail_s:
                    min_s_len = min(len(s_ood[k]) for k in avail_s)
                    use_len = min(min_ood_len, min_s_len)
                    ood_std = np.array([
                        np.mean([s_ood[k][i] for k in avail_s])
                        for i in range(use_len)
                    ])
                    # Align with mask
                    std_masked = ood_std[:len(ood_ep)][mask][:len(gap_smooth)]
                    ax.fill_between(ood_ep[mask][:len(std_masked)],
                                    gap_smooth[:len(std_masked)] - std_masked,
                                    gap_smooth[:len(std_masked)] + std_masked,
                                    color=c, alpha=0.1)

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Gap (pp): higher = more drop from ID to OOD")
        ax.set_title(f"{ds_label}\n(lower gap = more robust)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "robustness_ratio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 6. Cross-norm ranking heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(ecovalid, ecovalid_std, output_dir):
    """Horizontal bar chart of ecovalid best val accuracy per norm."""
    set_style()

    norms_sorted = []
    for norm in NORMS:
        h = ecovalid.get(norm)
        acc = best_val(h)
        if acc is not None and acc > 2.0:
            norms_sorted.append((norm, acc))
    norms_sorted.sort(key=lambda x: x[1])  # ascending for horizontal bars

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(norms_sorted))
    vals = [v for _, v in norms_sorted]
    norm_keys = [n for n, _ in norms_sorted]
    colors = [NORM_COLORS[n] for n in norm_keys]

    # Get std at best epoch
    err = []
    for n in norm_keys:
        s = ecovalid_std.get(n)
        h = ecovalid.get(n)
        best_idx = int(np.argmax(h["val_acc"]))
        std_val = s["val_acc"][best_idx] if s and "val_acc" in s and best_idx < len(s["val_acc"]) else 0.0
        err.append(std_val)

    bars = ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5,
                   xerr=err, capsize=4, error_kw={"linewidth": 1.0})
    ax.set_yticks(y)
    ax.set_yticklabels([NORM_LABELS[n] for n in norm_keys], fontsize=11)
    ax.set_xlabel("Best Validation Accuracy (%)")
    ax.set_title("Best Val Accuracy — ResNet-50, Eco-Valid (5 seeds)", fontweight="bold")

    for bar, v, e in zip(bars, vals, err):
        ax.text(v + e + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(0, max(vals) + 10)
    ax.axvline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Overfitting analysis
# ---------------------------------------------------------------------------

def plot_overfitting(ecovalid, ecovalid_std, output_dir):
    """Overfitting analysis: train vs val accuracy for ecovalid 5-seed runs."""
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        "Overfitting: Train vs Val Accuracy — ResNet-50, Eco-Valid (5 seeds)",
        fontsize=13, fontweight="bold",
    )

    for norm in NORMS:
        h = ecovalid.get(norm)
        s = ecovalid_std.get(norm)
        if h is None or best_val(h) is None or best_val(h) < 2.0:
            continue
        c = NORM_COLORS[norm]
        ls = NORM_STYLES[norm]
        epochs = np.array(h["epoch"][:len(h["train_acc"])])

        # Train accuracy (bold)
        train_mean = np.array(h["train_acc"][:len(epochs)])
        ax.plot(epochs, train_mean, color=c, linestyle=ls, linewidth=1.8,
                label=f"{NORM_LABELS[norm]}")
        if s and "train_acc" in s:
            train_std = np.array(s["train_acc"][:len(epochs)])
            ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                            color=c, alpha=0.08)

        # Val accuracy (thinner, same color)
        val_mean = np.array(h["val_acc"][:len(epochs)])
        ax.plot(epochs, val_mean, color=c, linestyle=ls, linewidth=1.0, alpha=0.6)
        if s and "val_acc" in s:
            val_std = np.array(s["val_acc"][:len(epochs)])
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                            color=c, alpha=0.08)

    # Custom legend: norm name only (bold = train, thin = val explained in subtitle)
    handles = [plt.Line2D([0], [0], color=NORM_COLORS[n], linestyle=NORM_STYLES[n],
                           linewidth=1.8, label=NORM_LABELS[n])
               for n in NORMS if ecovalid.get(n) and best_val(ecovalid.get(n)) and best_val(ecovalid.get(n)) > 2.0]
    ax.legend(handles=handles, fontsize=9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.text(0.98, 0.02, "Bold = train, thin = val", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    out = os.path.join(output_dir, "overfitting.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# 8. OOD analysis from ecovalid train logs
# ---------------------------------------------------------------------------

STANDARD_OOD = ["imagenet_r", "imagenet_a", "imagenet_sketch"]
STANDARD_OOD_LABELS = {
    "imagenet_r":       "ImageNet-R",
    "imagenet_a":       "ImageNet-A",
    "imagenet_sketch":  "ImageNet-Sketch",
}

C_CORRUPTION_DATASETS = [
    "imagenet_c_gaussian_noise_s3",
    "imagenet_c_defocus_blur_s3",
    "imagenet_c_fog_s3",
    "imagenet_c_contrast_s3",
]
C_CORRUPTION_LABELS = {
    "imagenet_c_gaussian_noise_s3": "Noise (s3)",
    "imagenet_c_defocus_blur_s3":   "Blur (s3)",
    "imagenet_c_fog_s3":            "Fog (s3)",
    "imagenet_c_contrast_s3":       "Contrast (s3)",
}


def plot_ood_analysis(ecovalid, ood_final, imagenet_c_full, output_dir):
    """4-panel OOD analysis: standard OOD, C-corruptions, ID/OOD scatter, robustness gap."""
    if not ood_final:
        print("  Skipping OOD analysis (no OOD data found in train logs)")
        return
    set_style()

    has_c_full = bool(imagenet_c_full)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "OOD Robustness: ResNet-50, Eco-Valid Augmentations (5 seeds, mean ± std)",
        fontsize=14, fontweight="bold",
    )

    # ---- Panel 1: Standard OOD (r, a, sketch) ----
    ax = axes[0, 0]
    avail_std = [ds for ds in STANDARD_OOD
                 if any(ds in ood_final.get(n, {}) for n in NORMS)]
    if avail_std:
        x = np.arange(len(avail_std))
        width = 0.13
        offsets = np.linspace(-(len(NORMS) - 1) / 2, (len(NORMS) - 1) / 2, len(NORMS)) * width
        for i, norm in enumerate(NORMS):
            means = [ood_final.get(norm, {}).get(ds, (0, 0))[0] for ds in avail_std]
            stds  = [ood_final.get(norm, {}).get(ds, (0, 0))[1] for ds in avail_std]
            ax.bar(x + offsets[i], means, width, label=NORM_LABELS[norm],
                   color=NORM_COLORS[norm], alpha=0.85, edgecolor="white", linewidth=0.4,
                   yerr=stds, capsize=2, error_kw={"linewidth": 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels([STANDARD_OOD_LABELS[ds] for ds in avail_std])
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Standard OOD Datasets")
        ax.legend(fontsize=8)
        ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    # ---- Panel 2: C-corruptions (full mCE if available, else 4-corruption proxy) ----
    ax = axes[0, 1]
    plotted_c = False
    if has_c_full:
        mce_key = "ood/imagenet_c_full/mce"
        norms_c = [n for n in NORMS if mce_key in imagenet_c_full.get(n, {})]
        if norms_c:
            x = np.arange(len(norms_c))
            means = [imagenet_c_full[n][mce_key][0] for n in norms_c]
            stds  = [imagenet_c_full[n][mce_key][1] for n in norms_c]
            ax.bar(x, means, color=[NORM_COLORS[n] for n in norms_c],
                   alpha=0.85, edgecolor="white", linewidth=0.4,
                   yerr=stds, capsize=3, error_kw={"linewidth": 0.8})
            ax.set_xticks(x)
            ax.set_xticklabels([NORM_LABELS[n] for n in norms_c], rotation=15, ha="right")
            ax.set_ylabel("mCE (lower = more robust)")
            ax.set_title("ImageNet-C Full: Mean Corruption Error")
            plotted_c = True
    if not plotted_c:
        avail_c = [ds for ds in C_CORRUPTION_DATASETS
                   if any(ds in ood_final.get(n, {}) for n in NORMS)]
        if avail_c:
            x = np.arange(len(avail_c))
            width = 0.13
            offsets = np.linspace(-(len(NORMS) - 1) / 2, (len(NORMS) - 1) / 2, len(NORMS)) * width
            for i, norm in enumerate(NORMS):
                means = [ood_final.get(norm, {}).get(ds, (0, 0))[0] for ds in avail_c]
                stds  = [ood_final.get(norm, {}).get(ds, (0, 0))[1] for ds in avail_c]
                ax.bar(x + offsets[i], means, width, label=NORM_LABELS[norm],
                       color=NORM_COLORS[norm], alpha=0.85, edgecolor="white", linewidth=0.4,
                       yerr=stds, capsize=2, error_kw={"linewidth": 0.8})
            ax.set_xticks(x)
            ax.set_xticklabels([C_CORRUPTION_LABELS[ds] for ds in avail_c], rotation=20, ha="right")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("ImageNet-C Corruptions (sev=3, proxy)")
            ax.legend(fontsize=8)
            ax.axhline(CHANCE_ACC, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    # ---- Panel 3: ID vs OOD scatter ----
    ax = axes[1, 0]
    for norm in NORMS:
        id_acc = best_val(ecovalid.get(norm)) or 0
        if id_acc < 2.0:
            continue
        ood_vals = [ood_final.get(norm, {}).get(ds, (0, 0))[0]
                    for ds in STANDARD_OOD if ds in ood_final.get(norm, {})]
        if not ood_vals:
            continue
        ood_mean = np.mean(ood_vals)
        c = NORM_COLORS[norm]
        ax.scatter(id_acc, ood_mean, color=c, s=130, zorder=5)
        ax.annotate(NORM_LABELS[norm], (id_acc, ood_mean),
                    textcoords="offset points", xytext=(6, 3), fontsize=9, color=c)
    ax.set_xlabel("In-Distribution Val Accuracy (%)")
    ax.set_ylabel("Mean OOD Accuracy (R + A + Sketch, %)")
    ax.set_title("ID vs OOD Trade-off")
    ax.plot([0, 100], [0, 100], color="gray", linestyle=":", alpha=0.4, linewidth=1)

    # ---- Panel 4: Robustness gap (OOD_mean − ID_acc) ----
    ax = axes[1, 1]
    norms_valid, gap_vals, gap_errs = [], [], []
    for norm in NORMS:
        id_acc = best_val(ecovalid.get(norm)) or 0
        if id_acc < 2.0:
            continue
        ood_means = [ood_final.get(norm, {}).get(ds, (0, 0))[0]
                     for ds in STANDARD_OOD if ds in ood_final.get(norm, {})]
        ood_stds  = [ood_final.get(norm, {}).get(ds, (0, 0))[1]
                     for ds in STANDARD_OOD if ds in ood_final.get(norm, {})]
        if not ood_means:
            continue
        norms_valid.append(norm)
        gap_vals.append(np.mean(ood_means) - id_acc)
        gap_errs.append(np.mean(ood_stds))
    if norms_valid:
        x = np.arange(len(norms_valid))
        ax.bar(x, gap_vals, color=[NORM_COLORS[n] for n in norms_valid],
               alpha=0.85, edgecolor="white", linewidth=0.4,
               yerr=gap_errs, capsize=3, error_kw={"linewidth": 0.8})
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([NORM_LABELS[n] for n in norms_valid], rotation=15, ha="right")
        ax.set_ylabel("OOD mean acc − ID val acc (pp)")
        ax.set_title("Robustness Gap (closer to 0 = more robust)")

    plt.tight_layout()
    out = os.path.join(output_dir, "ood_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_ood_evolution(ood_histories, ood_histories_std, ecovalid, ecovalid_std, output_dir):
    """Plot OOD metric evolution over training epochs with mean ± std bands."""
    if not ood_histories:
        print("  Skipping OOD evolution plot (no data)")
        return

    set_style()

    # Datasets logged during training (key as they appear in train.log)
    logged_datasets = [
        "imagenet_r",
        "imagenet_a",
        "imagenet_sketch",
        "imagenet_c_gaussian_noise_s3",
        "imagenet_c_defocus_blur_s3",
        "imagenet_c_fog_s3",
        "imagenet_c_contrast_s3",
    ]
    ds_labels = {
        "imagenet_r":                  "ImageNet-R",
        "imagenet_a":                  "ImageNet-A",
        "imagenet_sketch":             "ImageNet-Sketch",
        "imagenet_c_gaussian_noise_s3": "C-Noise (s3)",
        "imagenet_c_defocus_blur_s3":  "C-Blur (s3)",
        "imagenet_c_fog_s3":           "C-Fog (s3)",
        "imagenet_c_contrast_s3":      "C-Contrast (s3)",
    }

    n_ds = len(logged_datasets)
    ncols = 3
    nrows = -(-( n_ds + 1) // ncols)  # +1 for the ID val accuracy slot

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    fig.suptitle(
        "OOD Accuracy over Training — ResNet-50, Eco-Valid (5 seeds, mean ± 1 std)",
        fontsize=13, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for idx, ds_key in enumerate(logged_datasets):
        ax = axes_flat[idx]
        for norm in NORMS:
            h = ood_histories.get(norm)
            s = ood_histories_std.get(norm)
            if h is None or ds_key not in h:
                continue
            vals   = np.array(h[ds_key])
            epochs = np.array(h["epoch"][:len(vals)])
            c  = NORM_COLORS[norm]
            ls = NORM_STYLES[norm]
            ax.plot(epochs, vals, color=c, linestyle=ls,
                    linewidth=1.6, label=NORM_LABELS[norm])
            if s and ds_key in s:
                std = np.array(s[ds_key][:len(vals)])
                ax.fill_between(epochs, vals - std, vals + std,
                                color=c, alpha=0.12)

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
        s = ecovalid_std.get(norm)
        if h and best_val(h) and best_val(h) > 2.0:
            epochs = np.array(h["epoch"][:len(h["val_acc"])])
            mean = np.array(h["val_acc"])
            ax.plot(epochs, mean,
                    color=NORM_COLORS[norm], linestyle=NORM_STYLES[norm],
                    linewidth=1.6, label=NORM_LABELS[norm])
            if s and "val_acc" in s:
                std = np.array(s["val_acc"][:len(mean)])
                ax.fill_between(epochs, mean - std, mean + std,
                                color=NORM_COLORS[norm], alpha=0.12)
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


def print_ood_table(ood_final):
    if not ood_final:
        return
    all_ds = STANDARD_OOD + C_CORRUPTION_DATASETS
    available = [ds for ds in all_ds if any(ds in ood_final.get(n, {}) for n in NORMS)]
    short = {
        "imagenet_r": "IN-R", "imagenet_a": "IN-A", "imagenet_sketch": "Sketch",
        "imagenet_c_gaussian_noise_s3": "C-Noise",
        "imagenet_c_defocus_blur_s3":   "C-Blur",
        "imagenet_c_fog_s3":            "C-Fog",
        "imagenet_c_contrast_s3":       "C-Cntst",
    }
    print("\n" + "=" * 90)
    print("OOD ROBUSTNESS — ResNet-50, Eco-Valid Augmentations (final epoch, 5-seed mean)")
    print("=" * 90)
    header = f"{'Norm':<12}" + "".join(f"{short.get(ds, ds):>10}" for ds in available) + f"{'C-Mean':>10}"
    print(header)
    print("-" * len(header))
    for norm in NORMS:
        row = f"{NORM_LABELS[norm]:<12}"
        c_vals = []
        for ds in available:
            entry = ood_final.get(norm, {}).get(ds)
            if entry:
                mean, std = entry
                row += f"{mean:>9.1f}%"
                if ds in C_CORRUPTION_DATASETS:
                    c_vals.append(mean)
            else:
                row += f"{'—':>10}"
        row += f"{np.mean(c_vals):>9.1f}%" if c_vals else f"{'—':>10}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# 9. LocalNorm K-sweep analysis
# ---------------------------------------------------------------------------

# K_VARIANTS: norm key names used in run directories (K=2 reuses the existing 'localnorm' runs)
K_VARIANTS = ["localnorm_k1", "localnorm", "localnorm_k4", "localnorm_k8", "localnorm_k16"]
K_VALUES   = [1, 2, 4, 8, 16]


def load_ksweep_data(log_dir):
    """Load history and OOD results for LocalNorm K variants (seed-averaged).

    Returns:
        histories: dict[norm_key] -> (mean_history, std_history)
        ood_data:  dict[norm_key] -> dict[metric -> (mean, std)]
    """
    import json

    histories = {}
    ood_data  = {}

    for norm_key in K_VARIANTS:
        seed_histories = []
        seed_oods = []
        for seed in SEEDS:
            exp = f"resnet50_{norm_key}_imagenet_ecoset_ecovalid_seed{seed}"
            h = load_history(os.path.join(log_dir, exp, "history.csv"))
            if h:
                seed_histories.append(h)
            ood_path = os.path.join(log_dir, exp, "ood_results.json")
            if os.path.exists(ood_path):
                with open(ood_path) as f:
                    seed_oods.append(json.load(f))

        if seed_histories:
            mean_h, std_h = mean_std_histories(seed_histories)
            histories[norm_key] = (mean_h, std_h)
        if seed_oods:
            all_keys = set().union(*[set(d.keys()) for d in seed_oods])
            ood_data[norm_key] = {
                k: (
                    float(np.mean([d[k] for d in seed_oods if k in d])),
                    float(np.std( [d[k] for d in seed_oods if k in d])),
                )
                for k in all_keys
            }

    return histories, ood_data


def plot_ksweep(ksweep_histories, ksweep_ood, output_dir):
    """Plot LocalNorm K-sweep: ID val accuracy and OOD metrics vs K."""
    if not ksweep_histories:
        print("  Skipping K-sweep plot (no data yet)")
        return

    set_style()

    ood_metrics = [
        ("ood/imagenet_r/acc",      "ImageNet-R",  "#e377c2"),
        ("ood/imagenet_sketch/acc", "Sketch",       "#17becf"),
        ("ood/imagenet_a/acc",      "ImageNet-A",   "#bcbd22"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "LocalNorm K Sweep — ResNet-50, Eco-Valid (5 seeds, mean ± std)",
        fontsize=13, fontweight="bold",
    )

    # ---- Panel 1: Best val accuracy vs K ----
    ax = axes[0]
    k_plot, id_means, id_stds = [], [], []
    for norm_key, k_val in zip(K_VARIANTS, K_VALUES):
        mean_h, std_h = ksweep_histories.get(norm_key, (None, None))
        if mean_h is None or not mean_h.get("val_acc"):
            continue
        bv = best_val(mean_h)
        best_ep_idx = int(np.argmax(mean_h["val_acc"]))
        bv_std = (std_h["val_acc"][best_ep_idx]
                  if std_h and "val_acc" in std_h and best_ep_idx < len(std_h["val_acc"])
                  else 0.0)
        k_plot.append(k_val)
        id_means.append(bv)
        id_stds.append(bv_std)

    if k_plot:
        ax.errorbar(k_plot, id_means, yerr=id_stds, fmt="o-",
                    color="#1f77b4", linewidth=2, markersize=8, capsize=4,
                    label="LocalNorm (mean ± std)")
        if len(set(k_plot)) > 1:
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([str(k) for k in K_VALUES])
    else:
        ax.text(0.5, 0.5, "No training data yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xlabel("K (number of batch groups)")
    ax.set_ylabel("Best Val Accuracy (%)")
    ax.set_title("ID Accuracy vs K")
    ax.legend(fontsize=9)

    # ---- Panel 2: OOD metrics vs K ----
    ax = axes[1]
    has_ood = False
    for ood_key, ood_label, col in ood_metrics:
        k_ood, oo_means, oo_stds = [], [], []
        for norm_key, k_val in zip(K_VARIANTS, K_VALUES):
            entry = ksweep_ood.get(norm_key, {}).get(ood_key)
            if entry is not None:
                k_ood.append(k_val)
                oo_means.append(entry[0])
                oo_stds.append(entry[1])
        if k_ood:
            ax.errorbar(k_ood, oo_means, yerr=oo_stds, fmt="o-",
                        color=col, linewidth=2, markersize=8, capsize=4, label=ood_label)
            has_ood = True

    if has_ood:
        if len(set(k_plot)) > 1:
            ax.set_xscale("log", base=2)
            ax.set_xticks(K_VALUES)
            ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "No OOD data yet\n(run submit_snellius_ood_eval.sh)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xlabel("K (number of batch groups)")
    ax.set_ylabel("OOD Accuracy (%)")
    ax.set_title("OOD Accuracy vs K")

    plt.tight_layout()
    out = os.path.join(output_dir, "localnorm_k_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_ksweep_table(ksweep_histories, ksweep_ood):
    """Print K-sweep summary table to stdout."""
    import json

    ood_keys = [
        ("ood/imagenet_r/acc",      "IN-R"),
        ("ood/imagenet_sketch/acc", "Sketch"),
        ("ood/imagenet_a/acc",      "IN-A"),
    ]

    print("\n" + "=" * 75)
    print("LocalNorm K-SWEEP — ResNet-50, Eco-Valid (5-seed mean ± std)")
    print("=" * 75)
    header = f"{'K':<6}{'Best Val Acc':>16}" + "".join(f"{lbl:>12}" for _, lbl in ood_keys)
    print(header)
    print("-" * len(header))
    for norm_key, k_val in zip(K_VARIANTS, K_VALUES):
        mean_h, std_h = ksweep_histories.get(norm_key, (None, None))
        if mean_h is None:
            print(f"K={k_val:<4}{'—':>16}")
            continue
        bv = best_val(mean_h)
        best_ep_idx = int(np.argmax(mean_h["val_acc"]))
        bv_std = (std_h["val_acc"][best_ep_idx]
                  if std_h and "val_acc" in std_h and best_ep_idx < len(std_h["val_acc"])
                  else 0.0)
        row = f"K={k_val:<4}{bv:>13.2f}%±{bv_std:.1f}"
        for ood_key, _ in ood_keys:
            entry = ksweep_ood.get(norm_key, {}).get(ood_key)
            row += f"{entry[0]:>10.1f}%" if entry else f"{'—':>12}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze imagenet_ecoset norm experiment")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--output_dir", default="results/imagenet_ecoset_ecovalid_5seed")
    args = parser.parse_args()

    log_dir = args.log_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading runs from: {log_dir}")
    noaug, ecovalid, ecovalid_std = load_all(log_dir)

    n_noaug   = len(noaug)
    n_ecovalid = len(ecovalid)
    print(f"Loaded {n_noaug}/15 no-augment runs, {n_ecovalid}/6 ecovalid runs\n")

    # Summary table to stdout
    print_summary_table(noaug, ecovalid)

    # Figures
    print("Generating figures...")
    if ecovalid:
        plot_training_curves(ecovalid, ecovalid_std, output_dir)
        plot_summary_bars(ecovalid, ecovalid_std, output_dir)
        plot_heatmap(ecovalid, ecovalid_std, output_dir)
        plot_overfitting(ecovalid, ecovalid_std, output_dir)
        if noaug:
            plot_augmentation_comparison(noaug, ecovalid, output_dir)

    # OOD analysis — loaded from train logs + ood_results.json (if available)
    ood_final = load_ood_final(log_dir)
    imagenet_c_full = load_imagenet_c_full(log_dir)
    if imagenet_c_full:
        print(f"  Loaded full imagenet-c results for {len(imagenet_c_full)} norms")
    else:
        print("  No ood_results.json found; using 4-corruption proxy from training logs")
    print_ood_table(ood_final)
    plot_ood_analysis(ecovalid, ood_final, imagenet_c_full, output_dir)

    # OOD evolution over training (parsed from train.log)
    ood_histories, ood_histories_std = load_ood_history(log_dir)
    plot_ood_evolution(ood_histories, ood_histories_std, ecovalid, ecovalid_std, output_dir)
    plot_id_ood_trajectory(ecovalid, ood_histories, ood_histories_std, output_dir)
    plot_robustness_ratio(ecovalid, ood_histories, ood_histories_std, output_dir)

    # LocalNorm K-sweep analysis
    print("\nLoading LocalNorm K-sweep data...")
    ksweep_histories, ksweep_ood = load_ksweep_data(log_dir)
    n_ksweep = sum(1 for k in K_VARIANTS if k in ksweep_histories)
    print(f"  Loaded {n_ksweep}/{len(K_VARIANTS)} K variants")
    if ksweep_histories:
        print_ksweep_table(ksweep_histories, ksweep_ood)
        plot_ksweep(ksweep_histories, ksweep_ood, output_dir)

    print(f"\nAll outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
