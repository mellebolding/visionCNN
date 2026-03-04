"""OOD dataset loaders for robustness evaluation.

All datasets are filtered to match the model's training classes via ClassMapping.
Supports ImageNet-R, ImageNet-Sketch, ImageNet-A, on-the-fly ImageNet-C, and
Stylized ImageNet.
"""
import os
import numpy as np
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

from src.evaluation.class_mapping import ClassMapping


class OODDataset(ImageFolder):
    """Base class for OOD datasets filtered to a model's training classes.

    Loads an ImageFolder-structured dataset, keeps only samples whose class
    synset appears in the model's ClassMapping, and remaps labels to the
    model's output indices.
    """

    def __init__(
        self,
        root: str,
        class_mapping: ClassMapping,
        transform: Optional[transforms.Compose] = None,
    ):
        self.class_mapping = class_mapping
        self._synset_to_model_idx = class_mapping.synset_to_model_idx

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        super().__init__(root, transform=transform)

        # Build remap: original ImageFolder class_to_idx -> model's class index
        self._class_remap = {}
        for cls_name, orig_idx in self.class_to_idx.items():
            if cls_name in self._synset_to_model_idx:
                self._class_remap[orig_idx] = self._synset_to_model_idx[cls_name]

        # Filter samples to only classes in the model's training set
        filtered_samples = []
        filtered_targets = []
        for path, label in self.samples:
            if label in self._class_remap:
                new_label = self._class_remap[label]
                filtered_samples.append((path, new_label))
                filtered_targets.append(new_label)

        self.samples = filtered_samples
        self.targets = filtered_targets
        self.imgs = self.samples

    @property
    def n_classes_present(self) -> int:
        """How many of the model's training classes are present in this OOD dataset."""
        return len(self._class_remap)

    @property
    def n_total_ood_classes(self) -> int:
        """Total number of classes in the original OOD dataset (before filtering)."""
        return len(self.class_to_idx)


class ImageNetR(OODDataset):
    """ImageNet-R (Renditions) filtered to model's training classes."""
    pass


class ImageNetSketch(OODDataset):
    """ImageNet-Sketch filtered to model's training classes."""
    pass


class ImageNetA(OODDataset):
    """ImageNet-A (Adversarial) filtered to model's training classes."""
    pass


class StylizedImageNet(OODDataset):
    """Stylized ImageNet filtered to model's training classes."""
    pass


class CorruptedDataset(Dataset):
    """On-the-fly corruption of a base dataset using imagecorruptions.

    Applies a single corruption at a given severity to each image.
    The base dataset should return PIL images (before ToTensor).

    Args:
        base_dataset: An ImageFolder-like dataset (returns PIL images if
            transform is None, or we intercept before transform).
        corruption_name: Name of corruption (e.g., "gaussian_noise").
        severity: Corruption severity (1-5).
        transform: Transform to apply AFTER corruption.
    """

    def __init__(
        self,
        base_dataset,
        corruption_name: str,
        severity: int = 3,
        transform: Optional[transforms.Compose] = None,
    ):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        try:
            from imagecorruptions import corrupt
            self._corrupt = corrupt
        except ImportError:
            raise ImportError(
                "imagecorruptions package required for ImageNet-C evaluation. "
                "Install with: pip install imagecorruptions"
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get image path and label from base dataset
        path, label = self.base_dataset.samples[idx]

        # Load as PIL, resize/crop to 224x224, convert to numpy for corruption
        image = Image.open(path).convert("RGB")
        image = transforms.Resize(256)(image)
        image = transforms.CenterCrop(224)(image)
        image_np = np.array(image)  # uint8, (224, 224, 3)

        # Apply corruption
        corrupted = self._corrupt(image_np, severity=self.severity,
                                  corruption_name=self.corruption_name)

        # Convert back to PIL for transforms
        corrupted_pil = Image.fromarray(corrupted.astype(np.uint8))

        if self.transform:
            corrupted_pil = self.transform(corrupted_pil)

        return corrupted_pil, label
