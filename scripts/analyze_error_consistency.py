#!/usr/bin/env python3
"""Cross-model error consistency analysis (Geirhos 2021).

Computes pairwise error consistency between trained models to determine
whether different normalization strategies lead to similar or different
decision-making strategies.

Usage:
    python scripts/analyze_error_consistency.py --runs \
        logs/resnet50_batchnorm_noaug \
        logs/resnet50_layernorm_noaug \
        logs/resnet50_groupnorm_noaug \
        logs/resnet50_rmsnorm_noaug \
        logs/resnet50_derf_noaug
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.error_consistency import (
    compute_error_consistency,
    load_predictions_from_runs,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def plot_consistency_heatmap(matrix, output_path: str):
    """Generate and save a heatmap of the error consistency matrix."""
    if not HAS_PLOTTING:
        print("Warning: matplotlib/seaborn not available, skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Shorten model names for display
    labels = [name.replace("resnet50_", "").replace("_noaug", "") for name in matrix.index]

    sns.heatmap(
        matrix.values.astype(float),
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="RdYlBu_r",
        center=0,
        vmin=-0.2,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title("Error Consistency (Cohen's κ)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model error consistency analysis")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Directories of training runs (e.g., logs/resnet50_batchnorm_noaug)")
    parser.add_argument("--dataset", type=str, default="val",
                        help="Which dataset predictions to use (default: val)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="Specific epoch (default: latest)")
    parser.add_argument("--output_dir", type=str, default="logs/error_consistency",
                        help="Output directory for results")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Log results to wandb project")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load predictions from all runs
    print(f"Loading predictions from {len(args.runs)} runs...")
    predictions, labels = load_predictions_from_runs(
        args.runs, args.dataset, args.epoch
    )

    if len(predictions) < 2:
        print(f"Error: Need at least 2 models, got {len(predictions)}")
        print(f"Models found: {list(predictions.keys())}")
        sys.exit(1)

    if labels is None:
        print("Error: No labels found in prediction files")
        sys.exit(1)

    print(f"Loaded predictions for {len(predictions)} models:")
    for name, preds in predictions.items():
        acc = 100.0 * np.mean(preds == labels)
        print(f"  {name}: {len(preds)} samples, {acc:.2f}% accuracy")

    # Compute error consistency matrix
    print("\nComputing error consistency matrix...")
    matrix = compute_error_consistency(predictions, labels)

    print("\nError consistency matrix (Cohen's κ):")
    print(matrix.to_string(float_format="{:.3f}".format))

    # Save results
    matrix_path = os.path.join(args.output_dir, "error_consistency_matrix.csv")
    matrix.to_csv(matrix_path)
    print(f"\nMatrix saved to {matrix_path}")

    # Save as JSON too
    json_path = os.path.join(args.output_dir, "error_consistency.json")
    result = {
        "models": list(predictions.keys()),
        "n_samples": len(labels),
        "dataset": args.dataset,
        "matrix": matrix.to_dict(),
        "per_model_accuracy": {
            name: float(100.0 * np.mean(preds == labels))
            for name, preds in predictions.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate heatmap
    heatmap_path = os.path.join(args.output_dir, "error_consistency_heatmap.png")
    plot_consistency_heatmap(matrix, heatmap_path)

    # Log to wandb
    if args.wandb_project and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name="error_consistency_analysis")
        wandb.log({"error_consistency/matrix": wandb.Table(dataframe=matrix.reset_index())})
        if os.path.exists(heatmap_path):
            wandb.log({"error_consistency/heatmap": wandb.Image(heatmap_path)})
        # Log pairwise values
        model_names = list(predictions.keys())
        for i, a in enumerate(model_names):
            for j, b in enumerate(model_names):
                if i < j:
                    short_a = a.replace("resnet50_", "").replace("_noaug", "")
                    short_b = b.replace("resnet50_", "").replace("_noaug", "")
                    wandb.log({
                        f"error_consistency/{short_a}_vs_{short_b}": float(matrix.loc[a, b])
                    })
        wandb.finish()
        print("Results logged to wandb")

    print("\nDone!")


if __name__ == "__main__":
    main()
