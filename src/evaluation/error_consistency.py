"""Error consistency analysis across models (Geirhos 2021).

Measures whether different models make errors on the same images,
indicating similar or different decision-making strategies.
"""
import os
import numpy as np
import pandas as pd
from typing import Optional


def compute_error_consistency(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
) -> pd.DataFrame:
    """Compute pairwise error consistency between models.

    Error consistency = P(both correct or both wrong) - P(expected by chance)
    Normalized to [-1, 1] range using Cohen's kappa style normalization.

    Args:
        predictions: {model_name: array of predicted class indices}
        labels: ground truth class indices

    Returns:
        DataFrame with pairwise error consistency scores.
    """
    model_names = sorted(predictions.keys())
    n = len(labels)

    # Compute correctness arrays
    correct = {
        name: (predictions[name] == labels).astype(float)
        for name in model_names
    }

    # Pairwise consistency matrix
    matrix = pd.DataFrame(
        index=model_names, columns=model_names, dtype=float
    )

    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i == j:
                matrix.loc[name_a, name_b] = 1.0
                continue

            ca = correct[name_a]
            cb = correct[name_b]

            # Observed agreement: both correct or both wrong
            observed = np.mean((ca == cb).astype(float))

            # Expected agreement by chance
            pa = np.mean(ca)
            pb = np.mean(cb)
            expected = pa * pb + (1 - pa) * (1 - pb)

            # Cohen's kappa style normalization
            if expected == 1.0:
                kappa = 1.0
            else:
                kappa = (observed - expected) / (1.0 - expected)

            matrix.loc[name_a, name_b] = kappa

    return matrix


def load_predictions_from_runs(
    run_dirs: list[str],
    dataset_name: str = "val",
    epoch: Optional[int] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load predictions from multiple training runs.

    Args:
        run_dirs: List of log directories (e.g., logs/resnet50_batchnorm_noaug)
        dataset_name: Which dataset's predictions to load (default: "val")
        epoch: Specific epoch, or None for latest.

    Returns:
        Tuple of (predictions_dict, labels_array)
    """
    predictions = {}
    labels = None

    for run_dir in run_dirs:
        pred_dir = os.path.join(run_dir, "predictions")
        if not os.path.isdir(pred_dir):
            print(f"Warning: No predictions dir at {pred_dir}")
            continue

        # Find prediction files
        pred_files = sorted([
            f for f in os.listdir(pred_dir)
            if f.startswith("predictions_epoch_") and f.endswith(".npz")
        ])

        if not pred_files:
            print(f"Warning: No prediction files in {pred_dir}")
            continue

        if epoch is not None:
            target = f"predictions_epoch_{epoch:03d}.npz"
            if target not in pred_files:
                print(f"Warning: {target} not found in {pred_dir}")
                continue
            pred_file = target
        else:
            pred_file = pred_files[-1]  # latest

        filepath = os.path.join(pred_dir, pred_file)
        data = np.load(filepath)

        preds_key = f"{dataset_name}_preds"
        labels_key = f"{dataset_name}_labels"

        if preds_key not in data:
            print(f"Warning: {preds_key} not in {filepath}")
            continue

        run_name = os.path.basename(run_dir)
        predictions[run_name] = data[preds_key]

        if labels is None and labels_key in data:
            labels = data[labels_key]

    return predictions, labels
