"""Ecoset136: The 136 ecoset categories that share class names with ImageNet-1k.

Ecoset (Mehrer et al., 2021) is an ImageNet alternative with more ecologically
valid category selection. ~565 categories total, organized as NNNN_label folders
(e.g. 0001_bee/, 0045_volcano/). Multi-word labels use underscores in folder
names (e.g. 0099_tree_frog/ → label "tree frog").

This module provides the 136-class subset whose labels exactly match the primary
name of an ImageNet-1k class, enabling direct cross-dataset comparison.

IDENTICAL label indices to ImageNetEcoset (class 0 → same synset in both),
so models trained on either dataset can be compared with the same ClassMapping
and OOD evaluation setup.

For fair comparison with ImageNet-trained models, use max_per_class=1300 to
match ImageNet's ~1300 training images per class (isolates image diversity
effect from dataset size effect).
"""
import os
import random
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "configs" / "ecoset_imagenet_classes.txt"


def load_ecoset_class_list(
    class_list_cache: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Load the cached list of 136 ecoset-ImageNet overlapping synsets.

    Returns:
        (sorted_synsets, label_to_synset): sorted list of synset IDs +
        mapping from ecoset label string to synset ID.

    Raises FileNotFoundError if the cache is missing.
    """
    cache_path = Path(class_list_cache) if class_list_cache else _DEFAULT_CACHE
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Class list not found at {cache_path}.\n"
            "Run: python scripts/compute_ecoset_overlap.py"
        )
    synsets: list[str] = []
    label_to_synset: dict[str, str] = {}
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            synset = parts[0]
            # Label may be absent in old single-column format
            label = " ".join(parts[1:]) if len(parts) > 1 else ""
            synsets.append(synset)
            if label:
                label_to_synset[label] = synset
    return sorted(synsets), label_to_synset


class Ecoset136(Dataset):
    """Ecoset images for the 136 ImageNet-overlapping classes.

    Ecoset directory structure (NNNN_label folder naming):
        root/
          train/
            0001_bee/
              image001.jpg
              ...
            0045_volcano/
              ...
          val/
            ...

    Labels use the same 0..135 indices as ImageNetEcoset (sorted synset order),
    so models trained on either dataset are directly comparable.

    Args:
        root: Path to ecoset root directory (containing train/ and/or val/).
        split: "train" or "val".
        max_per_class: If set, randomly subsample to at most N images per class.
            Use max_per_class=1300 for size-controlled comparison with ImageNet.
        seed: Random seed for subsampling (reproducible).
        class_list_cache: Path to the 136-synset cache file.
        transform: Image transform pipeline.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        max_per_class: Optional[int] = None,
        seed: int = 42,
        class_list_cache: str | None = None,
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.split = split
        self.max_per_class = max_per_class
        self.transform = transform

        # Load the 136 overlapping synsets + label→synset mapping
        self.selected_synsets, label_to_synset = load_ecoset_class_list(class_list_cache)
        self.synset_to_idx = {s: i for i, s in enumerate(self.selected_synsets)}

        # Find the split directory
        split_root = os.path.join(root, split)
        if not os.path.isdir(split_root):
            # Fallback: root itself might be the split
            split_root = root

        if not os.path.isdir(split_root):
            raise FileNotFoundError(
                f"Ecoset split directory not found: {split_root}\n"
                f"Expected root/{{train|val}}/ or root/ to contain NNNN_label folders."
            )

        # Scan folder names: NNNN_label → eco_label → synset_id → class_idx
        # e.g. '0001_bee' → 'bee' → 'n02206856' → 0
        rng = random.Random(seed)
        self.samples: list[tuple[str, int]] = []

        for folder_name in sorted(os.listdir(split_root)):
            folder_path = os.path.join(split_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Parse NNNN_label format
            parts = folder_name.split("_", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue

            eco_label = parts[1].replace("_", " ")  # 'tree_frog' → 'tree frog'

            # Look up synset from label
            synset = label_to_synset.get(eco_label)
            if synset is None:
                continue  # Not in the 136-class subset

            class_idx = self.synset_to_idx[synset]

            # Collect image files
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            # Subsample if max_per_class set
            if max_per_class is not None and len(image_files) > max_per_class:
                image_files = rng.sample(image_files, max_per_class)

            for fname in sorted(image_files):
                self.samples.append((os.path.join(folder_path, fname), class_idx))

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def n_classes(self) -> int:
        return len(self.selected_synsets)

    @property
    def n_classes_present(self) -> int:
        """How many of the 136 classes actually have images in this split."""
        return len(set(label for _, label in self.samples))

    def class_counts(self) -> dict[int, int]:
        """Return per-class image counts."""
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts
