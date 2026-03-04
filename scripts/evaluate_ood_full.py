#!/usr/bin/env python3
"""Full post-training OOD evaluation on a trained checkpoint.

Usage:
    python scripts/evaluate_ood_full.py \
        --checkpoint logs/resnet50_batchnorm_noaug/best.pt \
        --datasets imagenet_r imagenet_sketch imagenet_a imagenet_c sin cue_conflict

    python scripts/evaluate_ood_full.py \
        --checkpoint logs/resnet50_batchnorm_noaug/best.pt \
        --datasets all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.factory import build_model
from src.datasets.build import build_ood_dataloaders
from src.evaluation.class_mapping import ClassMapping
from src.evaluation.ood_eval import (
    evaluate_all_ood,
    evaluate_imagenet_c_full,
    save_predictions_to_file,
)
from src.evaluation.shape_bias import evaluate_shape_bias

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

ALL_DATASETS = ["imagenet_r", "imagenet_sketch", "imagenet_a", "imagenet_c", "sin", "cue_conflict"]


def load_checkpoint_and_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, returning model and config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = build_model(cfg)
    # Handle DDP state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, cfg, checkpoint.get("epoch", -1), checkpoint.get("best_acc", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Full OOD evaluation on a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--datasets", nargs="+", default=["all"],
                        help="Which OOD datasets to evaluate (default: all)")
    parser.add_argument("--datasets_root", type=str, default="/export/scratch1/home/melle/datasets",
                        help="Root directory for OOD datasets")
    parser.add_argument("--source_dataset", type=str, default=None,
                        help="Source dataset (default: from checkpoint config)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save per-image predictions for error consistency")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Resume wandb run to log results (optional)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint dir)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, cfg, epoch, best_acc = load_checkpoint_and_model(args.checkpoint, device)
    print(f"Model loaded (epoch {epoch}, best_acc {best_acc:.2f}%)")

    # Determine output dir
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # Build class mapping
    source_dataset = args.source_dataset or cfg.get("data", {}).get("dataset", "imagenet100")
    imagenet_root = cfg.get("data", {}).get("root", "")
    class_mapping = ClassMapping(source_dataset, imagenet_train_root=imagenet_root)
    print(f"Class mapping: {source_dataset} ({class_mapping.num_classes} classes)")

    requested = set(args.datasets)
    if "all" in requested:
        requested = set(ALL_DATASETS)

    all_results = {}

    # Channels last
    use_channels_last = cfg.get("training", {}).get("channels_last", False)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    # 1. Standard OOD datasets (R, Sketch, A, lightweight C)
    standard_datasets = requested & {"imagenet_r", "imagenet_sketch", "imagenet_a"}
    if standard_datasets:
        # Build OOD config to pass to build_ood_dataloaders
        ood_cfg = {
            "ood_eval": {
                "enabled": True,
                "source_dataset": source_dataset,
                "datasets_root": args.datasets_root,
                "imagenet_c": {"during_training": False},
            },
            "training": {"batch_size": args.batch_size},
            "data": {"num_workers": args.num_workers},
        }
        ood_loaders = build_ood_dataloaders(ood_cfg, class_mapping)
        # Filter to requested
        ood_loaders = {k: v for k, v in ood_loaders.items() if k in standard_datasets}

        if ood_loaders:
            print(f"\nEvaluating standard OOD datasets: {list(ood_loaders.keys())}")
            results = evaluate_all_ood(
                model, ood_loaders, device,
                use_amp=use_amp,
                use_channels_last=use_channels_last,
                save_predictions=args.save_predictions,
            )
            predictions_store = results.pop("_predictions", None)
            all_results.update(results)

            if predictions_store and args.save_predictions:
                save_predictions_to_file(
                    predictions_store,
                    os.path.join(output_dir, "predictions"),
                    epoch,
                )

            for k, v in sorted(results.items()):
                if k.endswith("/acc"):
                    print(f"  {k}: {v:.2f}%")

    # 2. Full ImageNet-C (15 corruptions × 5 severities)
    if "imagenet_c" in requested:
        print("\nRunning full ImageNet-C evaluation (15 corruptions × 5 severities)...")
        # Need the base val dataset for on-the-fly corruption
        from src.datasets.build import build_dataloader
        from torchvision import transforms

        # Build a simple val dataset (no corruption) for the base
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Don't normalize here — CorruptedDataset needs raw images
        ])

        # We need to build the val dataset to get base images
        # Use the training config to build the val loader
        val_cfg = dict(cfg)
        val_cfg["data"]["root"] = val_cfg["data"].get("root", "")
        _, val_loader, _, _ = build_dataloader(val_cfg)
        base_val_dataset = val_loader.dataset

        imagenet_c_results = evaluate_imagenet_c_full(
            model, base_val_dataset, class_mapping, device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )

        # Add to results with ood/ prefix
        for k, v in imagenet_c_results.items():
            all_results[f"ood/imagenet_c_full/{k}"] = v

        print(f"  Mean accuracy: {imagenet_c_results['mean_acc']:.2f}%")
        print(f"  mCE: {imagenet_c_results['mce']:.2f}")
        for cat in ["noise", "blur", "weather", "digital"]:
            key = f"category_{cat}_mean"
            if key in imagenet_c_results:
                print(f"  {cat}: {imagenet_c_results[key]:.2f}%")

    # 3. Stylized ImageNet
    if "sin" in requested:
        from src.datasets.ood_datasets import StylizedImageNet
        from torchvision import transforms
        from torch.utils.data import DataLoader

        sin_path = os.path.join(args.datasets_root, f"sin-{source_dataset}")
        if os.path.isdir(sin_path):
            print(f"\nEvaluating Stylized ImageNet at {sin_path}")
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            sin_dataset = StylizedImageNet(sin_path, class_mapping, transform)
            if len(sin_dataset) > 0:
                sin_loader = DataLoader(
                    sin_dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True,
                )
                from src.evaluation.ood_eval import evaluate_ood
                sin_result = evaluate_ood(model, sin_loader, device, use_amp=use_amp,
                                          use_channels_last=use_channels_last)
                all_results["ood/sin/acc"] = sin_result["acc"]
                all_results["ood/sin/loss"] = sin_result["loss"]
                print(f"  SIN accuracy: {sin_result['acc']:.2f}%")
            else:
                print("  Warning: SIN dataset empty after filtering")
        else:
            print(f"  Warning: SIN not found at {sin_path}")

    # 4. Cue-conflict (texture-shape bias)
    if "cue_conflict" in requested:
        cue_conflict_path = os.path.join(args.datasets_root, "cue-conflict")
        if os.path.isdir(cue_conflict_path):
            print(f"\nEvaluating texture-shape bias (cue-conflict stimuli)")
            shape_results = evaluate_shape_bias(
                model, cue_conflict_path, class_mapping, device,
                use_amp=use_amp,
            )
            all_results["shape_bias/shape"] = shape_results["shape_bias"]
            all_results["shape_bias/texture"] = shape_results["texture_bias"]
            all_results["shape_bias/n_evaluated"] = shape_results["n_evaluated"]
            all_results["shape_bias/n_overlapping_classes"] = shape_results["n_overlapping_classes"]
            print(f"  Shape bias: {shape_results['shape_bias']:.3f}")
            print(f"  Texture bias: {shape_results['texture_bias']:.3f}")
            print(f"  Evaluated: {shape_results['n_evaluated']} images, "
                  f"{shape_results['n_overlapping_classes']} overlapping classes")
        else:
            print(f"  Warning: cue-conflict stimuli not found at {cue_conflict_path}")

    # Save results
    results_path = os.path.join(output_dir, "ood_results.json")
    # Convert numpy types for JSON serialization
    json_results = {k: float(v) if hasattr(v, 'item') else v for k, v in all_results.items()}
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Log to wandb if requested
    if args.wandb_run_id and WANDB_AVAILABLE:
        wandb_cfg = cfg.get("wandb", {})
        wandb.init(
            id=args.wandb_run_id,
            resume="must",
            project=wandb_cfg.get("project", "norm-comparison"),
        )
        wandb.log(all_results)
        wandb.finish()
        print("Results logged to wandb")

    print("\nDone!")


if __name__ == "__main__":
    main()
