#!/usr/bin/env python3
"""Extract penultimate features from trained models for representation analysis.

Loads a checkpoint, runs the validation set through the model, and saves
the penultimate-layer features + labels. Also computes analysis metrics:
  - Effective rank of the feature matrix
  - Class separability (Fisher's discriminant ratio)
  - Feature norms statistics

Usage:
    python scripts/extract_features.py --checkpoint logs/resnet_medium_batchnorm/best.pt
    python scripts/extract_features.py --checkpoint_dir /path/to/logs --experiments resnet_medium_batchnorm resnet_medium_layernorm
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.build import build_dataset


class FeatureExtractor:
    """Hook-based feature extractor for the penultimate layer."""

    def __init__(self, model):
        self.features = []
        self._hook = None
        self._register(model)

    def _register(self, model):
        """Find and hook the penultimate layer (layer before final classifier)."""
        # For our models, the penultimate layer is:
        # - VGG: pool (AdaptiveAvgPool2d) -> we hook after pool
        # - ResNet: avgpool (AdaptiveAvgPool2d) -> we hook after avgpool
        # - ConvNeXt: norm (final norm before GAP) -> we hook after GAP
        # Simplest: hook the final classifier's input by wrapping forward
        self.model = model

    def extract(self, x):
        """Forward pass, return (logits, features)."""
        model = self.model

        # VGG
        if hasattr(model, 'features') and hasattr(model, 'pool') and hasattr(model, 'classifier'):
            x = model.features(x)
            x = model.pool(x)
            feat = x.flatten(1)
            logits = model.classifier(feat)
            return logits, feat

        # ResNet
        if hasattr(model, 'layer4') and hasattr(model, 'avgpool') and hasattr(model, 'fc'):
            x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            feat = x.flatten(1)
            logits = model.fc(feat)
            return logits, feat

        # ConvNeXt
        if hasattr(model, 'stages') and hasattr(model, 'head'):
            for i in range(len(model.depths)):
                x = model.downsample_layers[i](x)
                x = model.stages[i](x)
            x = model.norm(x)
            feat = x.mean([-2, -1])  # GAP
            logits = model.head(feat)
            return logits, feat

        raise ValueError("Unknown model architecture for feature extraction")


def compute_effective_rank(features):
    """Compute effective rank of feature matrix via singular values."""
    # features: (N, D)
    _, s, _ = np.linalg.svd(features, full_matrices=False)
    # Normalize singular values to form a probability distribution
    s_norm = s / s.sum()
    # Shannon entropy
    entropy = -np.sum(s_norm * np.log(s_norm + 1e-12))
    # Effective rank = exp(entropy)
    return np.exp(entropy)


def compute_fisher_ratio(features, labels):
    """Compute Fisher's discriminant ratio (class separability)."""
    classes = np.unique(labels)
    global_mean = features.mean(axis=0)
    n_features = features.shape[1]

    # Between-class scatter
    sb = np.zeros((n_features, n_features))
    # Within-class scatter (trace only, for efficiency)
    sw_trace = 0.0

    for c in classes:
        mask = labels == c
        class_features = features[mask]
        n_c = class_features.shape[0]
        class_mean = class_features.mean(axis=0)
        diff = (class_mean - global_mean).reshape(-1, 1)
        sb += n_c * (diff @ diff.T)
        sw_trace += np.sum((class_features - class_mean) ** 2)

    sb_trace = np.trace(sb)
    # Fisher ratio = trace(Sb) / trace(Sw)
    return sb_trace / (sw_trace + 1e-12)


def extract_and_analyze(checkpoint_path, output_dir, device="cuda"):
    """Extract features from a checkpoint and compute analysis metrics."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]

    # Build model and load weights
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # Build val dataset
    val_dataset = build_dataset(cfg, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                            num_workers=4, pin_memory=True)

    extractor = FeatureExtractor(model)

    all_features = []
    all_labels = []

    print(f"Extracting features from {len(val_dataset)} samples...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            _, features = extractor.extract(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Features shape: {features.shape}")

    # Compute metrics
    eff_rank = compute_effective_rank(features)
    fisher = compute_fisher_ratio(features, labels)
    feat_norms = np.linalg.norm(features, axis=1)

    metrics = {
        "effective_rank": eff_rank,
        "fisher_ratio": fisher,
        "feat_norm_mean": feat_norms.mean(),
        "feat_norm_std": feat_norms.std(),
        "feature_dim": features.shape[1],
        "num_samples": features.shape[0],
    }

    print(f"  Effective rank:  {eff_rank:.1f} / {features.shape[1]}")
    print(f"  Fisher ratio:    {fisher:.4f}")
    print(f"  Feature norm:    {feat_norms.mean():.2f} +/- {feat_norms.std():.2f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(os.path.join(output_dir, "features.npz"),
                        features=features, labels=labels)

    with open(os.path.join(output_dir, "feature_metrics.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(metrics.keys())
        writer.writerow(metrics.values())

    return metrics


def compute_cka(features_x, features_y):
    """Compute linear CKA (Centered Kernel Alignment) between two feature matrices."""
    # Center features
    features_x = features_x - features_x.mean(axis=0)
    features_y = features_y - features_y.mean(axis=0)

    # Gram matrices (linear kernel)
    xx = features_x @ features_x.T
    yy = features_y @ features_y.T

    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    hsic_xy = np.sum(xx * yy)
    hsic_xx = np.sum(xx * xx)
    hsic_yy = np.sum(yy * yy)

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Extract features for representation analysis")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing experiment folders")
    parser.add_argument("--experiments", nargs="+", default=None, help="Experiment names to process")
    parser.add_argument("--output_dir", type=str, default="results/features", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.checkpoint:
        exp_name = Path(args.checkpoint).parent.name
        extract_and_analyze(args.checkpoint, os.path.join(args.output_dir, exp_name), args.device)
    elif args.checkpoint_dir and args.experiments:
        all_metrics = {}
        for exp in args.experiments:
            ckpt = os.path.join(args.checkpoint_dir, exp, "best.pt")
            if not os.path.exists(ckpt):
                print(f"  Skipping {exp}: no best.pt found")
                continue
            out = os.path.join(args.output_dir, exp)
            metrics = extract_and_analyze(ckpt, out, args.device)
            all_metrics[exp] = metrics

        # Compute pairwise CKA if multiple experiments
        if len(all_metrics) > 1:
            print("\n=== CKA Similarity Matrix ===")
            exp_names = sorted(all_metrics.keys())
            features_dict = {}
            for exp in exp_names:
                data = np.load(os.path.join(args.output_dir, exp, "features.npz"))
                features_dict[exp] = data["features"]

            # Print CKA matrix
            header = f"{'':>30}" + "".join(f"{e:>20}" for e in exp_names)
            print(header)
            for e1 in exp_names:
                row = f"{e1:>30}"
                for e2 in exp_names:
                    cka = compute_cka(features_dict[e1], features_dict[e2])
                    row += f"{cka:>20.4f}"
                print(row)

            # Save CKA matrix
            cka_path = os.path.join(args.output_dir, "cka_matrix.csv")
            with open(cka_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([""] + exp_names)
                for e1 in exp_names:
                    row = [e1]
                    for e2 in exp_names:
                        row.append(f"{compute_cka(features_dict[e1], features_dict[e2]):.6f}")
                    writer.writerow(row)
            print(f"\nCKA matrix saved to: {cka_path}")
    else:
        parser.error("Provide either --checkpoint or --checkpoint_dir + --experiments")


if __name__ == "__main__":
    main()
