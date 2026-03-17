"""OOD evaluation functions for robustness benchmarking."""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast
    TORCH_AMP_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast
    TORCH_AMP_NEW_API = False

# ImageNet-C corruption types grouped by category
CORRUPTION_CATEGORIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

ALL_CORRUPTIONS = [c for cats in CORRUPTION_CATEGORIES.values() for c in cats]


@torch.no_grad()
def evaluate_ood(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    use_channels_last: bool = False,
    gpu_transforms: nn.Module | None = None,
    save_predictions: bool = False,
    bn_adapt: bool = False,
) -> dict:
    """Evaluate model on a single OOD dataset.

    Args:
        bn_adapt: If True, BatchNorm layers are set to train mode so they use
            live batch statistics from the OOD data rather than running statistics
            estimated on the training distribution. Useful for measuring how much
            distribution shift affects BN stats.

    Returns:
        dict with keys: loss, acc, n_samples, n_classes_present.
        If save_predictions=True, also includes 'predictions' and 'labels' arrays.
    """
    model_to_eval = model.module if hasattr(model, "module") else model
    model_to_eval.eval()

    # Save BN running stats before any adaptation so training is not affected
    saved_bn_stats = {}
    if bn_adapt:
        for name, m in model_to_eval.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                saved_bn_stats[name] = (
                    m.running_mean.clone(),
                    m.running_var.clone(),
                    m.num_batches_tracked.clone(),
                )
                m.train()  # use live batch stats from OOD data

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if gpu_transforms is not None:
            images = gpu_transforms(images)
        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)

        if use_amp:
            if TORCH_AMP_NEW_API:
                with autocast(device_type="cuda"):
                    outputs = model_to_eval(images)
                    loss = criterion(outputs, labels)
            else:
                with autocast():
                    outputs = model_to_eval(images)
                    loss = criterion(outputs, labels)
        else:
            outputs = model_to_eval(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if save_predictions:
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Restore BN running stats so training is unaffected
    model_to_eval.eval()
    if saved_bn_stats:
        for name, m in model_to_eval.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)):
                m.running_mean.copy_(saved_bn_stats[name][0])
                m.running_var.copy_(saved_bn_stats[name][1])
                m.num_batches_tracked.copy_(saved_bn_stats[name][2])

    result = {
        "loss": total_loss / max(total, 1),
        "acc": 100.0 * correct / max(total, 1),
        "n_samples": total,
    }

    if save_predictions and all_preds:
        result["predictions"] = np.concatenate(all_preds)
        result["labels"] = np.concatenate(all_labels)

    return result


def evaluate_all_ood(
    model: nn.Module,
    ood_loaders: dict[str, DataLoader],
    device: torch.device,
    use_amp: bool = True,
    use_channels_last: bool = False,
    gpu_transforms: nn.Module | None = None,
    save_predictions: bool = False,
    bn_adapt: bool = False,
) -> dict:
    """Evaluate model on all OOD datasets.

    Returns flat dict for wandb logging, e.g.:
        {"ood/imagenet_r/acc": 42.5, "ood/imagenet_r/loss": 3.1, ...}
    """
    all_metrics = {}
    predictions_store = {}

    for name, loader in ood_loaders.items():
        if loader is None or (hasattr(loader.dataset, "__len__") and len(loader.dataset) == 0):
            continue

        result = evaluate_ood(
            model, loader, device, use_amp, use_channels_last,
            gpu_transforms, save_predictions, bn_adapt=bn_adapt,
        )

        all_metrics[f"ood/{name}/acc"] = result["acc"]
        all_metrics[f"ood/{name}/loss"] = result["loss"]
        all_metrics[f"ood/{name}/n_samples"] = result["n_samples"]

        if hasattr(loader.dataset, "n_classes_present"):
            all_metrics[f"ood/{name}/n_classes"] = loader.dataset.n_classes_present

        if save_predictions and "predictions" in result:
            predictions_store[name] = {
                "predictions": result["predictions"],
                "labels": result["labels"],
            }

    # Compute mean across ImageNet-C corruptions if any
    c_accs = [v for k, v in all_metrics.items()
              if k.startswith("ood/imagenet_c_") and k.endswith("/acc")]
    if c_accs:
        all_metrics["ood/imagenet_c/mean_acc"] = sum(c_accs) / len(c_accs)

    if save_predictions:
        all_metrics["_predictions"] = predictions_store

    return all_metrics


def evaluate_imagenet_c_full(
    model: nn.Module,
    base_val_dataset,
    class_mapping,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 8,
    use_amp: bool = True,
    use_channels_last: bool = False,
    gpu_transforms: nn.Module | None = None,
) -> dict:
    """Full ImageNet-C evaluation: 15 corruptions × 5 severities.

    Generates corrupted images on-the-fly using imagecorruptions package.

    Returns:
        dict with per-corruption, per-severity, and summary metrics.
    """
    from src.datasets.ood_datasets import CorruptedDataset

    results = {}
    for corruption in ALL_CORRUPTIONS:
        for severity in range(1, 6):
            corrupted = CorruptedDataset(base_val_dataset, corruption, severity)
            loader = DataLoader(
                corrupted,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            result = evaluate_ood(
                model, loader, device, use_amp, use_channels_last, gpu_transforms
            )
            key = f"{corruption}_s{severity}"
            results[key] = result["acc"]

    # Compute per-corruption mean (across severities)
    for corruption in ALL_CORRUPTIONS:
        accs = [results[f"{corruption}_s{s}"] for s in range(1, 6)]
        results[f"{corruption}_mean"] = sum(accs) / len(accs)

    # Compute per-category mean
    for cat, corruptions in CORRUPTION_CATEGORIES.items():
        cat_accs = [results[f"{c}_mean"] for c in corruptions]
        results[f"category_{cat}_mean"] = sum(cat_accs) / len(cat_accs)

    # Compute overall mCE (as error rate, lower is better)
    all_means = [results[f"{c}_mean"] for c in ALL_CORRUPTIONS]
    results["mean_acc"] = sum(all_means) / len(all_means)
    results["mce"] = 100.0 - results["mean_acc"]  # mean corruption error

    return results


def save_predictions_to_file(predictions_store: dict, save_dir: str, epoch: int):
    """Save per-image predictions to .npz for error consistency analysis."""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"predictions_epoch_{epoch:03d}.npz")
    np.savez_compressed(filepath, **{
        f"{name}_preds": data["predictions"]
        for name, data in predictions_store.items()
    }, **{
        f"{name}_labels": data["labels"]
        for name, data in predictions_store.items()
    })
    return filepath
