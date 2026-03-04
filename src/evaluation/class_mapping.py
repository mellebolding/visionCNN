"""Dataset-agnostic class mapping between a model's training classes and ImageNet synsets.

Supports ImageNet-100, ImageNet-Ecoset136, full ImageNet, and ecoset. OOD evaluation
datasets (ImageNet-R/A/S/C) all use ImageNet synset folder names, so this mapping
determines which OOD samples belong to the model's output space.

Key: both imagenet_ecoset136 and ecoset136 models should use
ClassMapping("imagenet_ecoset136") so their OOD evaluation is identical.
"""
import os
from typing import Optional


class ClassMapping:
    """Maps between a model's training classes and ImageNet synsets.

    Args:
        source_dataset: One of "imagenet100", "imagenet_ecoset136", "imagenet", "ecoset".
        imagenet_train_root: Path to imagenet/train/ (needed for imagenet100, imagenet,
            and imagenet_ecoset136).
        ecoset_root: Path to ecoset/ root (needed for ecoset).
    """

    def __init__(
        self,
        source_dataset: str,
        imagenet_train_root: Optional[str] = None,
        ecoset_root: Optional[str] = None,
    ):
        self.source_dataset = source_dataset.lower()
        self._synset_to_idx: dict[str, int] = {}
        self._idx_to_synset: dict[int, str] = {}

        if self.source_dataset == "imagenet100":
            self._init_imagenet100(imagenet_train_root)
        elif self.source_dataset == "imagenet_ecoset136":
            self._init_imagenet_ecoset136(imagenet_train_root)
        elif self.source_dataset == "imagenet":
            self._init_imagenet(imagenet_train_root)
        elif self.source_dataset == "ecoset":
            self._init_ecoset(ecoset_root)
        else:
            raise ValueError(
                f"Unknown source_dataset '{source_dataset}'. "
                "Choose from: imagenet100, imagenet_ecoset136, imagenet, ecoset"
            )

    def _init_imagenet_ecoset136(self, imagenet_train_root: Optional[str]):
        """136 ImageNet synsets that have ecoset counterparts.

        Both imagenet_ecoset136-trained and ecoset136-trained models should use
        this mapping so OOD evaluation is directly comparable.
        """
        if imagenet_train_root is None:
            raise ValueError("imagenet_train_root required for imagenet_ecoset136")
        from src.datasets.imagenet_ecoset import get_imagenet_ecoset_classes

        selected = get_imagenet_ecoset_classes(imagenet_train_root)
        for idx, synset in enumerate(selected):
            self._synset_to_idx[synset] = idx
            self._idx_to_synset[idx] = synset

    def _init_imagenet100(self, imagenet_train_root: Optional[str]):
        if imagenet_train_root is None:
            raise ValueError("imagenet_train_root required for imagenet100")
        from src.datasets.imagenet100 import get_imagenet100_classes

        selected = get_imagenet100_classes(imagenet_train_root)
        for idx, synset in enumerate(selected):
            self._synset_to_idx[synset] = idx
            self._idx_to_synset[idx] = synset

    def _init_imagenet(self, imagenet_train_root: Optional[str]):
        if imagenet_train_root is None:
            raise ValueError("imagenet_train_root required for imagenet")
        all_classes = sorted(os.listdir(imagenet_train_root))
        all_classes = [
            c for c in all_classes
            if os.path.isdir(os.path.join(imagenet_train_root, c))
        ]
        for idx, synset in enumerate(all_classes):
            self._synset_to_idx[synset] = idx
            self._idx_to_synset[idx] = synset

    def _init_ecoset(self, ecoset_root: Optional[str]):
        """Ecoset folders are named synset_id_label (e.g., n01440764_dog).

        Only ~16% of ecoset's 565 categories overlap with ImageNet-1k,
        so OOD evaluation on ImageNet-based benchmarks covers a limited subset.
        """
        if ecoset_root is None:
            raise ValueError("ecoset_root required for ecoset")
        # Ecoset uses train/ split for class discovery
        train_root = os.path.join(ecoset_root, "train")
        if not os.path.isdir(train_root):
            train_root = ecoset_root

        folders = sorted(os.listdir(train_root))
        folders = [f for f in folders if os.path.isdir(os.path.join(train_root, f))]

        for idx, folder in enumerate(folders):
            # Extract synset from folder name: "n01440764_dog" -> "n01440764"
            parts = folder.split("_", 1)
            synset = parts[0] if parts[0].startswith("n") else folder
            self._synset_to_idx[synset] = idx
            self._idx_to_synset[idx] = synset

    @property
    def synset_to_model_idx(self) -> dict[str, int]:
        """Map from ImageNet synset folder name to model's output class index."""
        return self._synset_to_idx

    @property
    def model_idx_to_synset(self) -> dict[int, str]:
        """Map from model's output class index to ImageNet synset folder name."""
        return self._idx_to_synset

    @property
    def num_classes(self) -> int:
        return len(self._synset_to_idx)

    @property
    def synsets(self) -> list[str]:
        """All synsets the model was trained on, sorted by model index."""
        return [self._idx_to_synset[i] for i in range(self.num_classes)]

    def __repr__(self) -> str:
        return (
            f"ClassMapping(source={self.source_dataset}, "
            f"num_classes={self.num_classes})"
        )
