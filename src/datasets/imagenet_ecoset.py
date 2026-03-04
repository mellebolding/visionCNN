"""ImageNet-Ecoset136: ImageNet subset using the 136 synsets shared with ecoset.

Unlike ImageNet-100 (arbitrary every-10th selection), this subset uses exactly
those ImageNet classes that have a counterpart in the ecoset dataset, enabling
direct comparison between ImageNet-trained and ecoset-trained models using the
same semantic categories and OOD benchmarks.

The 136 synsets are determined by matching ecoset's 565 plain-English category
labels (e.g. "bee", "volcano") against the primary class name of each ImageNet
synset. Ecoset folders use NNNN_label format (e.g. 0001_bee/), not synset IDs.

Class list is cached in configs/ecoset_imagenet_classes.txt (tab-separated:
synset_id<TAB>ecoset_label). Run scripts/compute_ecoset_overlap.py to
generate/update this cache.
"""
import os
from pathlib import Path
from torchvision.datasets import ImageFolder


# Absolute path to the default cache file (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "configs" / "ecoset_imagenet_classes.txt"


def get_imagenet_ecoset_classes(
    imagenet_train_root: str,
    ecoset_root: str | None = None,
    class_list_cache: str | None = None,
) -> list[str]:
    """Return sorted list of ImageNet synsets that appear in ecoset.

    Priority:
    1. Load from cache file (reproducible, preferred)
    2. Compute dynamically from ecoset_root and imagenet_train_root
    3. Raise informative error

    Args:
        imagenet_train_root: Path to imagenet/train/ for cross-referencing.
        ecoset_root: Path to ecoset/ for dynamic computation (optional).
        class_list_cache: Path to cached synset list (default: configs/ecoset_imagenet_classes.txt).

    Returns:
        Sorted list of synset strings, e.g. ["n01440764", "n01530575", ...]
    """
    cache_path = Path(class_list_cache) if class_list_cache else _DEFAULT_CACHE

    # 1. Try cache file (supports both single-column and tab-separated formats)
    if cache_path.exists():
        synsets = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Tab-separated: "synset_id\teco_label" — take first column
                    synset = line.split()[0]
                    synsets.append(synset)
        if synsets:
            return sorted(synsets)

    # 2. Compute dynamically (requires timm + imagenet)
    if imagenet_train_root and os.path.isdir(imagenet_train_root):
        from scripts.compute_ecoset_overlap import compute_overlap, save_overlap, ECOSET_LABELS
        matches = compute_overlap(ECOSET_LABELS, imagenet_train_root)
        if matches:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_overlap(matches, str(cache_path))
            return sorted(m[0] for m in matches)

    # 3. Error
    raise FileNotFoundError(
        f"Ecoset-ImageNet class list not found at {cache_path}.\n"
        "Run: python scripts/compute_ecoset_overlap.py\n"
        f"   --ecoset_root /path/to/ecoset\n"
        f"   --imagenet_root {imagenet_train_root}"
    )


class ImageNetEcoset(ImageFolder):
    """ImageNet subset filtered to the 136 ecoset-overlapping synsets.

    Labels remapped to 0..135 in sorted synset order — IDENTICAL ordering
    to Ecoset136, so models trained on both datasets can be directly compared
    using the same ClassMapping and OOD evaluation setup.

    Args:
        root: Path to imagenet/{train,val} split directory.
        imagenet_train_root: Path to imagenet/train/ (for class list derivation).
            If None, inferred from root's parent.
        ecoset_root: Optional path to ecoset/ for dynamic class computation.
        class_list_cache: Path to cached synset list.
        transform: Image transform pipeline.
    """

    def __init__(
        self,
        root: str,
        imagenet_train_root: str | None = None,
        ecoset_root: str | None = None,
        class_list_cache: str | None = None,
        transform=None,
    ):
        if imagenet_train_root is None:
            # Infer from root (e.g., imagenet/val → imagenet/train)
            imagenet_train_root = os.path.join(os.path.dirname(root), "train")

        self.selected_classes = get_imagenet_ecoset_classes(
            imagenet_train_root, ecoset_root, class_list_cache
        )
        self.class_set = set(self.selected_classes)

        # Initialize ImageFolder (loads all classes in root)
        super().__init__(root, transform=transform)

        # Build remap: original ImageFolder class index → new 0..135 index
        self.class_remap = {}
        for new_idx, cls_name in enumerate(self.selected_classes):
            if cls_name in self.class_to_idx:
                self.class_remap[self.class_to_idx[cls_name]] = new_idx

        # Filter and remap samples
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

    @property
    def n_classes(self) -> int:
        return len(self.selected_classes)
