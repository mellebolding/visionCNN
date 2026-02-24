"""ImageNet-100: a fixed 100-class subset of ImageNet-1k.

Uses every 10th class (sorted by synset folder name) for a reproducible,
balanced subset. Labels are remapped to 0-99.
"""
import os
from torchvision.datasets import ImageFolder


def get_imagenet100_classes(imagenet_train_root):
    """Return sorted list of 100 class folder names (every 10th class).

    Args:
        imagenet_train_root: Path to imagenet/train/ directory.

    Returns:
        List of 100 class directory names.
    """
    all_classes = sorted(os.listdir(imagenet_train_root))
    # Filter to actual directories (skip hidden files etc.)
    all_classes = [c for c in all_classes if os.path.isdir(os.path.join(imagenet_train_root, c))]
    # Every 10th class -> exactly 100 classes from 1000
    selected = all_classes[::10]
    return selected


class ImageNet100(ImageFolder):
    """ImageNet-100: 100-class subset of ImageNet-1k.

    Loads the full ImageFolder, then filters samples to only the selected
    100 classes and remaps labels to 0-99.

    Args:
        root: Path to imagenet/{train,val} split directory.
        transform: Image transform pipeline.
        imagenet_train_root: Path to imagenet/train/ (used to derive the
            canonical class list). If None, inferred from root.
    """

    def __init__(self, root, transform=None, imagenet_train_root=None):
        # We need the train root to get a consistent class list across splits
        if imagenet_train_root is None:
            parent = os.path.dirname(root)
            imagenet_train_root = os.path.join(parent, "train")

        self.selected_classes = get_imagenet100_classes(imagenet_train_root)
        self.class_set = set(self.selected_classes)

        # Initialize ImageFolder normally (loads all 1000 classes)
        super().__init__(root, transform=transform)

        # Build remap: original class_to_idx -> new 0-99 indices
        self.class_remap = {}
        for new_idx, cls_name in enumerate(self.selected_classes):
            if cls_name in self.class_to_idx:
                self.class_remap[self.class_to_idx[cls_name]] = new_idx

        # Filter samples to only the 100 selected classes, and remap labels
        filtered_samples = []
        filtered_targets = []
        for path, label in self.samples:
            if label in self.class_remap:
                new_label = self.class_remap[label]
                filtered_samples.append((path, new_label))
                filtered_targets.append(new_label)

        self.samples = filtered_samples
        self.targets = filtered_targets
        self.imgs = self.samples  # ImageFolder alias
