"""NVIDIA DALI pipelines for GPU-accelerated ImageNet data loading.

DALI (Data Loading Library) moves data preprocessing to the GPU, significantly
reducing CPU bottlenecks and improving training throughput.

Usage:
    from src.datasets.dali_imagenet import build_dali_dataloader

    train_loader, val_loader = build_dali_dataloader(cfg, dist_manager)

    for images, labels in train_loader:
        # images are already on GPU, normalized, and augmented
        outputs = model(images)

Requirements:
    - nvidia-dali-cuda120 (install via conda or pip)
    - ImageNet dataset organized as train/val with class subdirectories
"""

import os
from typing import Optional, Tuple, Any

try:
    from nvidia.dali import pipeline_def, fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


# ImageNet normalization constants
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def check_dali_available():
    """Check if DALI is available and raise helpful error if not."""
    if not DALI_AVAILABLE:
        raise ImportError(
            "NVIDIA DALI is not installed. Install with:\n"
            "  conda install -c nvidia nvidia-dali-cuda120\n"
            "or:\n"
            "  pip install nvidia-dali-cuda120"
        )


@pipeline_def
def imagenet_train_pipeline(
    data_dir: str,
    img_size: int = 224,
    random_area: Tuple[float, float] = (0.08, 1.0),
    random_aspect_ratio: Tuple[float, float] = (0.75, 1.333),
    shard_id: int = 0,
    num_shards: int = 1,
    shuffle: bool = True,
    decoder_device: str = "mixed",  # "mixed" = decode on CPU, "gpu" = decode on GPU
    dali_cpu: bool = False,  # Run entire pipeline on CPU (fallback)
    random_seed: int = 42,
):
    """DALI training pipeline for ImageNet.

    Args:
        data_dir: Path to ImageNet train directory (with class subdirs)
        img_size: Output image size
        random_area: Range for random crop area
        random_aspect_ratio: Range for random crop aspect ratio
        shard_id: GPU/process ID for distributed training
        num_shards: Total number of GPUs/processes
        shuffle: Shuffle data
        decoder_device: "mixed" (CPU decode + GPU ops) or "gpu" (full GPU)
        dali_cpu: If True, run entire pipeline on CPU
        random_seed: Random seed for reproducibility
    """
    # Read files from directory structure
    # For DDP: each GPU reads its own shard (stick_to_shard=True)
    # random_shuffle=True shuffles the shard initially, giving variety across epochs
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=shuffle,  # Initial shuffle of this shard's files
        stick_to_shard=True,     # Each GPU reads only its portion (proper DDP)
        pad_last_batch=True,     # Ensure consistent batch sizes across shards
        name="Reader",
        seed=random_seed + shard_id,  # Different seed per GPU for variety
    )

    # Decode images
    # DALI decoder options:
    # - "cpu": libjpeg-turbo on CPU (slower)
    # - "mixed": nvJPEG on GPU (fast!) - despite the name, this IS GPU-accelerated
    #   On Ampere+ GPUs, this also uses the dedicated HW JPEG decoder
    device = "cpu" if dali_cpu else "gpu"

    if dali_cpu or decoder_device == "cpu":
        # Pure CPU decode - use only if GPU decode causes issues
        images = fn.decoders.image(images, device="cpu", output_type=types.RGB)
        if not dali_cpu:
            images = images.gpu()
    else:
        # GPU-accelerated decode with nvJPEG (this is what you want!)
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # Random resized crop (GPU-accelerated)
    images = fn.random_resized_crop(
        images,
        size=img_size,
        random_area=random_area,
        random_aspect_ratio=random_aspect_ratio,
        device=device,
        seed=random_seed + 1,
    )

    # Random horizontal flip
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5, seed=random_seed + 2),
        device=device,
    )

    # Normalize to [0, 1] then apply ImageNet normalization
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device=device,
    )

    # Move labels to GPU for consistency
    if not dali_cpu:
        labels = labels.gpu()

    return images, labels


@pipeline_def
def imagenet_val_pipeline(
    data_dir: str,
    img_size: int = 224,
    resize_size: Optional[int] = None,
    shard_id: int = 0,
    num_shards: int = 1,
    stick_to_shard: bool = False,  # Validation: all ranks see all data
    decoder_device: str = "mixed",
    dali_cpu: bool = False,
    random_seed: int = 42,
):
    """DALI validation pipeline for ImageNet.

    Args:
        data_dir: Path to ImageNet val directory
        img_size: Output image size after center crop
        resize_size: Resize short edge to this before center crop (default: img_size/0.875)
        shard_id: GPU/process ID
        num_shards: Total number of GPUs
        stick_to_shard: If True, each shard reads only its portion
        decoder_device: "mixed" or "gpu"
        dali_cpu: Run on CPU
        random_seed: Random seed
    """
    if resize_size is None:
        resize_size = int(img_size / 0.875)  # 256 for img_size=224

    # Read files sequentially (no shuffle for validation)
    images, labels = fn.readers.file(
        file_root=data_dir,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=False,
        stick_to_shard=stick_to_shard,
        name="Reader",
        seed=random_seed,
    )

    device = "cpu" if dali_cpu else "gpu"

    # Decode (same logic as training pipeline)
    if dali_cpu or decoder_device == "cpu":
        images = fn.decoders.image(images, device="cpu", output_type=types.RGB)
        if not dali_cpu:
            images = images.gpu()
    else:
        # GPU-accelerated decode with nvJPEG
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)

    # Resize (short edge) + Center crop
    images = fn.resize(
        images,
        resize_shorter=resize_size,
        device=device,
    )

    images = fn.crop(
        images,
        crop=(img_size, img_size),
        device=device,
    )

    # Normalize
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device=device,
    )

    if not dali_cpu:
        labels = labels.gpu()

    return images, labels


class DALIWrapper:
    """PyTorch-compatible wrapper for DALI iterators.

    This wrapper makes DALI iterators behave like PyTorch DataLoaders,
    returning (images, labels) tuples and supporting len() and iteration.
    """

    def __init__(
        self,
        pipeline,
        output_map: list = ["images", "labels"],
        reader_name: str = "Reader",
        last_batch_policy: Any = None,
        auto_reset: bool = True,
    ):
        """Initialize DALI wrapper.

        Args:
            pipeline: Built DALI pipeline
            output_map: Names for pipeline outputs
            reader_name: Name of the file reader in the pipeline
            last_batch_policy: How to handle last batch (PARTIAL, DROP, FILL)
            auto_reset: Auto-reset iterator at epoch end
        """
        if last_batch_policy is None:
            last_batch_policy = LastBatchPolicy.PARTIAL

        self.pipeline = pipeline
        self.reader_name = reader_name

        self._iterator = DALIGenericIterator(
            pipeline,
            output_map,
            reader_name=reader_name,
            last_batch_policy=last_batch_policy,
            auto_reset=auto_reset,
        )

        # Cache length
        self._len = self._iterator._size // pipeline.max_batch_size
        if self._iterator._size % pipeline.max_batch_size != 0:
            self._len += 1

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self._iterator)
            # DALI returns list of dicts, one per GPU
            # We take the first one (single GPU per process in DDP)
            images = data[0]["images"]
            labels = data[0]["labels"].squeeze(-1).long()
            return images, labels
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return self._len

    def reset(self):
        """Reset iterator for new epoch."""
        self._iterator.reset()

    @property
    def dataset(self):
        """Fake dataset property for compatibility with training scripts."""
        return _FakeDataset(self._iterator._size)


class _FakeDataset:
    """Minimal dataset interface for compatibility."""
    def __init__(self, size: int):
        self._size = size

    def __len__(self):
        return self._size


def build_dali_dataloader(
    cfg: dict,
    dist_manager: Optional[Any] = None,
) -> Tuple[DALIWrapper, DALIWrapper]:
    """Build DALI train and validation dataloaders.

    Args:
        cfg: Configuration dictionary with data and training settings
        dist_manager: DistributedManager for DDP training

    Returns:
        Tuple of (train_loader, val_loader) as DALIWrapper objects
    """
    check_dali_available()

    # Extract config
    data_cfg = cfg.get("data", {})
    dali_cfg = data_cfg.get("dali", {})
    training_cfg = cfg.get("training", {})

    data_root = data_cfg.get("root", "./data")
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # Validate paths
    if not os.path.exists(train_dir):
        raise ValueError(f"DALI: ImageNet train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise ValueError(f"DALI: ImageNet val directory not found: {val_dir}")

    # Settings
    img_size = data_cfg.get("img_size", 224)
    batch_size = training_cfg.get("batch_size", 64)
    seed = training_cfg.get("seed", 42)

    # DALI-specific settings
    num_threads = dali_cfg.get("num_threads", 4)
    prefetch_queue_depth = dali_cfg.get("prefetch_queue_depth", 2)
    decoder_device = dali_cfg.get("decoder_device", "mixed")
    dali_cpu = dali_cfg.get("cpu", False)

    # Distributed settings
    if dist_manager is not None and dist_manager.is_distributed:
        shard_id = dist_manager.rank
        num_shards = dist_manager.world_size
        device_id = dist_manager.local_rank
    else:
        shard_id = 0
        num_shards = 1
        device_id = 0

    # Build training pipeline
    train_pipeline = imagenet_train_pipeline(
        data_dir=train_dir,
        img_size=img_size,
        shard_id=shard_id,
        num_shards=num_shards,
        decoder_device=decoder_device,
        dali_cpu=dali_cpu,
        random_seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    train_pipeline.build()

    # Build validation pipeline
    # Note: For validation, we don't shard by default (all ranks see all data)
    # This matches the original PyTorch behavior in build.py
    val_pipeline = imagenet_val_pipeline(
        data_dir=val_dir,
        img_size=img_size,
        shard_id=0,  # All ranks read all validation data
        num_shards=1,
        decoder_device=decoder_device,
        dali_cpu=dali_cpu,
        random_seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    val_pipeline.build()

    # Wrap pipelines
    train_loader = DALIWrapper(
        train_pipeline,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.DROP,  # Drop incomplete batches for training
    )

    val_loader = DALIWrapper(
        val_pipeline,
        reader_name="Reader",
        last_batch_policy=LastBatchPolicy.PARTIAL,  # Keep all samples for validation
    )

    return train_loader, val_loader


def is_dali_available() -> bool:
    """Check if DALI is available."""
    return DALI_AVAILABLE
