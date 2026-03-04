"""Texture-shape bias evaluation using cue-conflict stimuli (Geirhos 2019)."""
import os
import json
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.evaluation.class_mapping import ClassMapping

# The 16 cue-conflict classes from Geirhos et al., 2019
# These are plain English names, need to map to ImageNet synsets
CUE_CONFLICT_CLASSES = [
    "airplane", "bear", "bicycle", "bird", "boat", "bottle",
    "car", "cat", "chair", "clock", "dog", "elephant",
    "keyboard", "knife", "oven", "truck",
]

# Map cue-conflict class names to representative ImageNet synsets
# (multiple synsets may match; we use the most common one)
CUE_CONFLICT_TO_SYNSETS = {
    "airplane": ["n02690373", "n02692877", "n04552348"],
    "bear": ["n02132136", "n02133161", "n02134084", "n02134418"],
    "bicycle": ["n02835271", "n03792782"],
    "bird": [],  # Too generic — many bird synsets
    "boat": ["n02951358", "n03344393", "n03662601", "n04273569"],
    "bottle": ["n02823428", "n03937543", "n04557648", "n04560804"],
    "car": ["n02814533", "n03100240", "n03459775", "n03770679", "n04285008"],
    "cat": ["n02123045", "n02123159", "n02123394", "n02123597"],
    "chair": ["n02791124", "n03376595", "n04099969"],
    "clock": ["n02708093", "n03196217", "n04548280"],
    "dog": [],  # Too generic — many dog synsets
    "elephant": ["n02504013", "n02504458"],
    "keyboard": ["n03085013", "n04505470"],
    "knife": ["n03041632"],
    "oven": ["n03259280", "n04111531"],
    "truck": ["n03345487", "n03417042", "n03796401", "n04461696", "n04467665"],
}


class CueConflictDataset(Dataset):
    """Dataset for cue-conflict stimuli (Geirhos 2019).

    Each image has a shape class and a texture class (conflicting).
    The model's response indicates whether it's shape-biased or texture-biased.
    """

    def __init__(self, root: str, class_mapping: ClassMapping, transform=None):
        self.root = root
        self.class_mapping = class_mapping
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = []  # (path, shape_class, texture_class)
        self._overlapping_classes = set()

        # Determine which cue-conflict classes overlap with model's training classes
        model_synsets = set(class_mapping.synset_to_model_idx.keys())
        self._class_to_model_idxs = {}

        for cc_class, synsets in CUE_CONFLICT_TO_SYNSETS.items():
            matching = [s for s in synsets if s in model_synsets]
            if matching:
                self._class_to_model_idxs[cc_class] = [
                    class_mapping.synset_to_model_idx[s] for s in matching
                ]
                self._overlapping_classes.add(cc_class)

        # Load cue-conflict images
        if os.path.isdir(root):
            self._load_images(root)

    def _load_images(self, root: str):
        """Load cue-conflict images. Expected format: shape-texture.png/jpg"""
        for fname in sorted(os.listdir(root)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            # Parse filename: typically "shape_texture_N.png" or similar
            name = os.path.splitext(fname)[0]
            parts = name.split("-")
            if len(parts) >= 2:
                shape_class = parts[0].strip()
                texture_class = parts[1].strip().split("_")[0]
                # Only keep if at least shape or texture class overlaps
                if (shape_class in self._overlapping_classes or
                        texture_class in self._overlapping_classes):
                    self.samples.append((
                        os.path.join(root, fname),
                        shape_class,
                        texture_class,
                    ))

    @property
    def n_overlapping_classes(self) -> int:
        return len(self._overlapping_classes)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, shape_class, texture_class = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, shape_class, texture_class


@torch.no_grad()
def evaluate_shape_bias(
    model: nn.Module,
    cue_conflict_dir: str,
    class_mapping: ClassMapping,
    device: torch.device,
    use_amp: bool = True,
    batch_size: int = 32,
) -> dict:
    """Evaluate texture vs shape bias on cue-conflict stimuli.

    Returns:
        dict with shape_bias, texture_bias, n_evaluated, n_overlapping_classes,
        and per-class breakdown.
    """
    dataset = CueConflictDataset(cue_conflict_dir, class_mapping)

    if len(dataset) == 0:
        return {
            "shape_bias": 0.0,
            "texture_bias": 0.0,
            "n_evaluated": 0,
            "n_overlapping_classes": dataset.n_overlapping_classes,
        }

    model_to_eval = model.module if hasattr(model, "module") else model
    model_to_eval.eval()

    shape_correct = 0
    texture_correct = 0
    other = 0
    total = 0

    # Process images one by one (small dataset)
    for i in range(len(dataset)):
        image, shape_class, texture_class = dataset[i]
        image = image.unsqueeze(0).to(device)

        if use_amp:
            try:
                from torch.amp import autocast
                with autocast(device_type="cuda"):
                    output = model_to_eval(image)
            except ImportError:
                from torch.cuda.amp import autocast
                with autocast():
                    output = model_to_eval(image)
        else:
            output = model_to_eval(image)

        pred_idx = output.argmax(1).item()

        # Check if prediction matches shape or texture class
        shape_idxs = dataset._class_to_model_idxs.get(shape_class, [])
        texture_idxs = dataset._class_to_model_idxs.get(texture_class, [])

        if pred_idx in shape_idxs:
            shape_correct += 1
        elif pred_idx in texture_idxs:
            texture_correct += 1
        else:
            other += 1
        total += 1

    model_to_eval.train()

    decided = shape_correct + texture_correct
    shape_bias = shape_correct / max(decided, 1)
    texture_bias = texture_correct / max(decided, 1)

    return {
        "shape_bias": shape_bias,
        "texture_bias": texture_bias,
        "shape_correct": shape_correct,
        "texture_correct": texture_correct,
        "other": other,
        "n_evaluated": total,
        "n_overlapping_classes": dataset.n_overlapping_classes,
    }
