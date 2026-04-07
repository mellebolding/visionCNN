#!/usr/bin/env python3
"""Download history.csv and ood_results.json for the 15 single-seed 3-arch ecovalid runs from WandB.

Usage:
    python scripts/fetch_wandb_ecovalid_3arch.py [--output_dir local_logs]
"""
import argparse
import csv
import json
import os

import wandb

NORMS = ["batchnorm", "layernorm", "groupnorm", "rmsnorm", "derf"]
ARCHS = ["resnet50", "convnext_small", "vit_small"]

GROUPS = {
    "resnet50":       "resnet50-imagenet-ecoset-ecovalid-norm-experiment",
    "convnext_small": "convnext-imagenet-ecoset-ecovalid-norm-experiment",
    "vit_small":      "vit-imagenet-ecoset-ecovalid-norm-experiment",
}

OOD_WANDB_KEYS = [
    "ood/imagenet_r/acc",
    "ood/imagenet_a/acc",
    "ood/imagenet_sketch/acc",
    "ood/imagenet_c/mean_acc",
    "ood/imagenet_c_gaussian_noise_s3/acc",
    "ood/imagenet_c_fog_s3/acc",
    "ood/imagenet_c_defocus_blur_s3/acc",
    "ood/imagenet_c_contrast_s3/acc",
]


def fetch_run(api, run_name, group, output_dir):
    """Fetch a single run's history and OOD summary, save locally."""
    runs = api.runs(
        "norm-comparison",
        filters={"display_name": run_name, "group": group, "state": "finished"},
    )
    run = None
    for r in runs:
        run = r
        break

    if run is None:
        print(f"  [WARN] Not found: {run_name} (group={group})")
        return False

    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # --- Download full history ---
    hist = run.history(samples=10000, pandas=False)
    rows = []
    for step in hist:
        epoch = step.get("epoch")
        train_acc = step.get("train/acc")
        val_acc = step.get("val/acc")
        if epoch is None or train_acc is None or val_acc is None:
            continue
        rows.append({
            "epoch":      int(epoch),
            "train_loss": step.get("train/loss", ""),
            "train_acc":  train_acc,
            "val_loss":   step.get("val/loss", ""),
            "val_acc":    val_acc,
            "lr":         step.get("lr", ""),
        })

    if not rows:
        print(f"  [WARN] Empty history for {run_name}")
        return False

    csv_path = os.path.join(run_dir, "history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()
        writer.writerows(rows)

    # --- Save OOD summary from run.summary ---
    ood = {}
    for key in OOD_WANDB_KEYS:
        val = run.summary.get(key)
        if val is not None:
            ood[key] = float(val)

    if ood:
        ood_path = os.path.join(run_dir, "ood_results.json")
        with open(ood_path, "w") as f:
            json.dump(ood, f, indent=2)

    n_epochs = len(rows)
    best_val = max(r["val_acc"] for r in rows)
    print(f"  [OK] {run_name}: {n_epochs} epochs, best val={best_val:.2f}%  OOD keys={len(ood)}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="local_logs")
    args = parser.parse_args()

    api = wandb.Api()
    loaded = 0
    for arch in ARCHS:
        for norm in NORMS:
            run_name = f"{arch}_{norm}_imagenet_ecoset_ecovalid"
            group = GROUPS[arch]
            print(f"Fetching {run_name} ...")
            ok = fetch_run(api, run_name, group, args.output_dir)
            if ok:
                loaded += 1

    print(f"\nDone: {loaded}/15 runs fetched → {args.output_dir}/")


if __name__ == "__main__":
    main()
