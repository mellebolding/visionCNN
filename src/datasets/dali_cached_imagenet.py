"""DALI-augmented pipeline with RAM-cached ImageNet data.

Combines the speed of RAM-cached data loading with DALI's GPU-accelerated
augmentation pipeline. Images are pre-loaded into RAM at a larger resolution
(e.g., 256x256) in HWC format, then fed into a DALI pipeline via
external_source for full GPU augmentation.

This gives you:
- Zero disk I/O during training (data is in RAM via memmap)
- Full GPU-accelerated augmentations (RandomResizedCrop, flip, color jitter, etc.)
- No CPU bottleneck for data preprocessing

Memory requirements (approximate, cache_size=256):
- Train (1.28M images) at 256x256x3 uint8: ~252 GB
- Val (50K images) at 256x256x3 uint8: ~10 GB
- Total: ~262 GB (shared across all DDP processes via memmap)

Usage:
    Set data.backend: dali_cached in your config YAML.

    data:
      backend: dali_cached
      cache_size: 256          # Resolution to cache at (default 256)
      color_jitter: false      # Enable color augmentation
      dali:
        num_threads: 4
        prefetch_queue_depth: 2

Requirements:
    - nvidia-dali-cuda120
    - Sufficient RAM for the cache (~262 GB for cache_size=256)
"""

import collections
import gc
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

try:
    from nvidia.dali import pipeline_def, fn
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    import nvidia.dali.types as types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False


# ImageNet normalization constants (scaled to [0, 255] for uint8 input)
IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


class CachedExternalSource:
    """Batch-mode external source for DALI reading from disk-cached memmap.

    Yields entire batches as dense numpy arrays for DALI's external_source
    with batch=True. Uses a small ThreadPoolExecutor (2 workers) with a
    bounded sliding window to overlap memmap I/O with GPU compute.

    Args:
        images: numpy memmap of shape [N, H, W, C] uint8
        labels: numpy array of shape [N]
        batch_size: Samples per batch
        shard_id: GPU/process ID for DDP sharding
        num_shards: Total number of GPUs/processes
        shuffle: Shuffle indices each epoch
        seed: Random seed for reproducibility
        drop_last: Drop the last incomplete batch
    """

    PREFETCH_WORKERS = 2
    PREFETCH_DEPTH = 4  # batches in flight

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shard_id: int = 0,
        num_shards: int = 1,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Compute this shard's indices for DDP
        total = len(labels)
        per_shard = total // num_shards
        self.shard_start = shard_id * per_shard
        self.shard_end = (shard_id + 1) * per_shard if shard_id < num_shards - 1 else total
        self.indices = np.arange(self.shard_start, self.shard_end)
        self.n = len(self.indices)

        self._executor = None
        self._futures = collections.deque()
        self._batch_indices = []
        self._submit_pos = 0

    def _load_batch(self, idx):
        """Load a single batch from memmap. GIL is released during page faults."""
        sorted_idx = np.sort(idx)
        images = np.ascontiguousarray(self.images[sorted_idx])
        labels = self.labels[sorted_idx].astype(np.int32)
        return images, labels

    def _submit_next(self):
        if self._submit_pos < len(self._batch_indices):
            f = self._executor.submit(self._load_batch, self._batch_indices[self._submit_pos])
            self._futures.append(f)
            self._submit_pos += 1

    def _shutdown(self):
        """Fully stop old executor and free all prefetched data."""
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._futures.clear()
            self._executor = None

    def __iter__(self):
        if self.shuffle:
            rng = np.random.RandomState(self.seed + self.epoch)
            rng.shuffle(self.indices)
        self.epoch += 1

        # Pre-compute all batch index arrays
        self._batch_indices = []
        pos = 0
        while pos < self.n:
            remaining = self.n - pos
            if self.drop_last and remaining < self.batch_size:
                break
            end = min(pos + self.batch_size, self.n)
            self._batch_indices.append(self.indices[pos:end].copy())
            pos = end

        # Fully stop old executor before starting new one
        self._shutdown()

        # Start new executor with bounded prefetch
        self._executor = ThreadPoolExecutor(max_workers=self.PREFETCH_WORKERS)
        self._futures = collections.deque()
        self._submit_pos = 0
        for _ in range(min(self.PREFETCH_DEPTH, len(self._batch_indices))):
            self._submit_next()

        return self

    def __next__(self):
        if not self._futures:
            self._shutdown()
            raise StopIteration
        result = self._futures.popleft().result()
        self._submit_next()
        return result

    def __len__(self):
        """Number of samples in this shard."""
        return self.n

    def __del__(self):
        self._shutdown()


@pipeline_def
def cached_train_pipeline(
    source,
    img_size: int = 224,
    random_area: Tuple[float, float] = (0.08, 1.0),
    random_aspect_ratio: Tuple[float, float] = (0.75, 1.333),
    color_jitter: bool = False,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    random_seed: int = 42,
):
    """DALI training pipeline that reads from cached external source.

    Pipeline: external_source → GPU → RandomResizedCrop → Flip
              → [ColorJitter] → Normalize → CHW output

    All augmentation ops run on GPU.
    """
    images, labels = fn.external_source(
        source=source,
        num_outputs=2,
        batch=True,
        layout=["HWC", ""],
    )

    # Move to GPU
    images = images.gpu()

    # RandomResizedCrop on GPU (the key augmentation)
    images = fn.random_resized_crop(
        images,
        size=img_size,
        random_area=random_area,
        random_aspect_ratio=random_aspect_ratio,
        device="gpu",
        seed=random_seed + 1,
    )

    # Random horizontal flip
    images = fn.flip(
        images,
        horizontal=fn.random.coin_flip(probability=0.5, seed=random_seed + 2),
        device="gpu",
    )

    # Optional: Color augmentation on GPU
    if color_jitter:
        images = fn.brightness_contrast(
            images,
            brightness=fn.random.uniform(
                range=(max(0.0, 1.0 - brightness), 1.0 + brightness),
                seed=random_seed + 3,
            ),
            contrast=fn.random.uniform(
                range=(max(0.0, 1.0 - contrast), 1.0 + contrast),
                seed=random_seed + 4,
            ),
            device="gpu",
        )
        images = fn.saturation(
            images,
            saturation=fn.random.uniform(
                range=(max(0.0, 1.0 - saturation), 1.0 + saturation),
                seed=random_seed + 5,
            ),
            device="gpu",
        )

    # Normalize: uint8 [0,255] → float32 with ImageNet normalization
    # Also converts HWC → CHW layout
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device="gpu",
    )

    labels = labels.gpu()
    return images, labels


@pipeline_def
def cached_val_pipeline(
    source,
    img_size: int = 224,
    random_seed: int = 42,
):
    """DALI validation pipeline that reads from cached external source.

    Pipeline: external_source → GPU → CenterCrop → Normalize → CHW output
    """
    images, labels = fn.external_source(
        source=source,
        num_outputs=2,
        batch=True,
        layout=["HWC", ""],
    )

    images = images.gpu()

    # Center crop (cached images are e.g. 256x256, crop to 224x224)
    images = fn.crop(
        images,
        crop=(img_size, img_size),
        device="gpu",
    )

    # Normalize and convert HWC → CHW
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        device="gpu",
    )

    labels = labels.gpu()
    return images, labels


class DALICachedWrapper:
    """PyTorch-compatible wrapper for DALI cached pipelines.

    Makes DALI pipelines behave like PyTorch DataLoaders, returning
    (images, labels) tuples. On reset(), explicitly resets the Python
    source and DALI pipeline, then rebuilds the iterator.
    """

    def __init__(
        self,
        pipeline,
        source: CachedExternalSource,
        num_samples: int,
        output_map: list = None,
        last_batch_policy=None,
    ):
        if output_map is None:
            output_map = ["images", "labels"]
        if last_batch_policy is None:
            last_batch_policy = LastBatchPolicy.PARTIAL

        self.pipeline = pipeline
        self._source = source
        self._num_samples = num_samples
        self._output_map = output_map
        self._last_batch_policy = last_batch_policy

        self._iterator = DALIGenericIterator(
            pipeline,
            output_map,
            size=num_samples,
            last_batch_policy=last_batch_policy,
            auto_reset=False,
        )

        self._len = num_samples // pipeline.max_batch_size
        if num_samples % pipeline.max_batch_size != 0:
            self._len += 1

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self._iterator)
            images = data[0]["images"]
            labels = data[0]["labels"].squeeze(-1).long()
            return images, labels
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return self._len

    def reset(self):
        """Reset for new epoch: reset source, pipeline, rebuild iterator."""
        del self._iterator
        gc.collect()  # ensure DALI C++ buffers are freed before reallocating
        iter(self._source)
        try:
            self.pipeline.reset()
        except Exception:
            pass
        self._iterator = DALIGenericIterator(
            self.pipeline,
            self._output_map,
            size=self._num_samples,
            last_batch_policy=self._last_batch_policy,
            auto_reset=False,
        )

    @property
    def dataset(self):
        """Fake dataset property for compatibility with training scripts."""
        return _FakeDataset(self._num_samples)


class _FakeDataset:
    """Minimal dataset interface for len() compatibility."""
    def __init__(self, size: int):
        self._size = size

    def __len__(self):
        return self._size


def _cache_imagenet_hwc(
    root: str,
    split: str,
    cache_size: int,
    cache_dir: str,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load or create HWC uint8 cache for DALI.

    Images are stored as [N, H, W, C] uint8 numpy memmap. Cache is created
    once by rank 0 and memory-mapped by all ranks.

    Args:
        root: ImageNet root directory (containing train/ and val/)
        split: 'train' or 'val'
        cache_size: Resolution to cache at (short edge resize + center crop)
        cache_dir: Directory to store cache files
        rank: DDP rank
        world_size: DDP world size
        verbose: Print progress

    Returns:
        Tuple of (images_memmap, labels_array)
    """
    cache_name = f"imagenet_dali_cache_{split}_{cache_size}"
    images_path = os.path.join(cache_dir, f"{cache_name}_images.npy")
    labels_path = os.path.join(cache_dir, f"{cache_name}_labels.npy")
    meta_path = os.path.join(cache_dir, f"{cache_name}_meta.txt")

    def read_meta():
        with open(meta_path, 'r') as f:
            parts = f.read().strip().split(',')
            return tuple(int(p) for p in parts)

    cache_exists = (
        os.path.exists(images_path)
        and os.path.exists(labels_path)
        and os.path.exists(meta_path)
    )

    is_distributed = world_size > 1

    if is_distributed:
        if rank == 0 and not cache_exists:
            _create_hwc_cache(root, split, cache_size, images_path, labels_path, meta_path, verbose)
        else:
            # Wait for rank 0 to finish creating the cache (file-based sync
            # to avoid NCCL barrier timeout on long cache builds)
            while not os.path.exists(meta_path):
                time.sleep(5)
        if dist.is_initialized():
            dist.barrier()
    else:
        if not cache_exists:
            _create_hwc_cache(root, split, cache_size, images_path, labels_path, meta_path, verbose)

    # All ranks: memory-map the cache
    shape = read_meta()
    images = np.memmap(images_path, dtype=np.uint8, mode='r', shape=shape)
    # Tell OS not to read-ahead on this memmap — reduces page cache pressure
    # when the file (252 GB) is larger than RAM (251 GB)
    images._mmap.madvise(mmap.MADV_RANDOM)
    labels = np.load(labels_path)

    if rank == 0 and verbose:
        mem_gb = images.nbytes / 1e9
        print(f"[DALICached] {split}: {len(labels)} images, shape={shape}, {mem_gb:.1f} GB")

    return images, labels


def _create_hwc_cache(
    root: str,
    split: str,
    cache_size: int,
    images_path: str,
    labels_path: str,
    meta_path: str,
    verbose: bool,
):
    """Create HWC uint8 cache from ImageNet directory on disk.

    Images are resized (short edge) and center-cropped to cache_size x cache_size,
    then stored in [N, H, W, C] format for efficient DALI consumption.
    """
    data_path = os.path.join(root, split)
    if not os.path.exists(data_path):
        raise ValueError(f"ImageNet {split} directory not found: {data_path}")

    # Resize short edge to cache_size, then center crop to cache_size x cache_size
    transform = transforms.Compose([
        transforms.Resize(cache_size),
        transforms.CenterCrop(cache_size),
        transforms.PILToTensor(),  # [C, H, W] uint8
    ])

    dataset = ImageFolder(data_path, transform=transform)
    n = len(dataset)

    if verbose:
        mem_gb = n * cache_size * cache_size * 3 / 1e9
        print(f"[DALICached] Creating {split} cache: {n} images at {cache_size}x{cache_size} (~{mem_gb:.0f} GB)")
        print(f"[DALICached] Output: {images_path}")

    # Create memmap: [N, H, W, C] uint8 (HWC format for DALI)
    shape = (n, cache_size, cache_size, 3)
    os.makedirs(os.path.dirname(images_path), exist_ok=True)
    images_mm = np.memmap(images_path, dtype=np.uint8, mode='w+', shape=shape)
    labels_list = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, num_workers=0, shuffle=False
    )
    offset = 0
    iterator = tqdm(loader, desc=f"Caching {split} (HWC)") if verbose else loader

    for imgs, lbls in iterator:
        # imgs: [B, C, H, W] uint8 → [B, H, W, C] for DALI
        imgs_hwc = imgs.permute(0, 2, 3, 1).numpy()
        bs = len(imgs_hwc)
        images_mm[offset:offset + bs] = imgs_hwc
        labels_list.append(lbls.numpy())
        offset += bs

    images_mm.flush()
    labels_arr = np.concatenate(labels_list)
    np.save(labels_path, labels_arr)

    # Write metadata (shape for memmap reconstruction)
    with open(meta_path, 'w') as f:
        f.write(f"{shape[0]},{shape[1]},{shape[2]},{shape[3]}")

    if verbose:
        print(f"[DALICached] Cache created: {n} images ({images_mm.nbytes / 1e9:.1f} GB)")


def build_dali_cached_dataloader(
    cfg: dict,
    dist_manager: Optional[Any] = None,
) -> Tuple["DALICachedWrapper", "DALICachedWrapper"]:
    """Build DALI-augmented dataloaders with RAM-cached ImageNet data.

    Args:
        cfg: Configuration dictionary
        dist_manager: DistributedManager for DDP training

    Returns:
        Tuple of (train_loader, val_loader) as DALICachedWrapper objects
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            "NVIDIA DALI is not installed. Install with:\n"
            "  conda install -c nvidia nvidia-dali-cuda120\n"
            "or:\n"
            "  pip install nvidia-dali-cuda120"
        )

    data_cfg = cfg.get("data", {})
    dali_cfg = data_cfg.get("dali", {})
    training_cfg = cfg.get("training", {})

    data_root = data_cfg.get("root", "./data")
    img_size = data_cfg.get("img_size", 224)
    cache_size = data_cfg.get("cache_size", 256)
    batch_size = training_cfg.get("batch_size", 64)
    seed = training_cfg.get("seed", 42)

    # DALI settings
    num_threads = dali_cfg.get("num_threads", 4)
    prefetch_queue_depth = dali_cfg.get("prefetch_queue_depth", 2)

    # Augmentation settings
    color_jitter = data_cfg.get("color_jitter", False)
    brightness = data_cfg.get("brightness", 0.4)
    contrast = data_cfg.get("contrast", 0.4)
    saturation = data_cfg.get("saturation", 0.4)

    # DDP settings
    rank = dist_manager.rank if dist_manager else 0
    world_size = dist_manager.world_size if dist_manager else 1
    device_id = dist_manager.local_rank if dist_manager else 0
    is_distributed = dist_manager is not None and dist_manager.is_distributed
    verbose = rank == 0

    # Determine cache directory
    cache_dir = data_cfg.get("cache_dir", "/dev/shm")
    if cache_dir == "/dev/shm":
        import shutil
        shm_free = shutil.disk_usage("/dev/shm").free / 1e9
        needed_gb = 1.28e6 * cache_size * cache_size * 3 / 1e9 + 10
        if shm_free < needed_gb:
            cache_dir = os.path.join(os.path.dirname(data_root), "imagenet_dali_cache")
            if verbose:
                print(f"[DALICached] /dev/shm too small ({shm_free:.0f}GB free, need ~{needed_gb:.0f}GB)")
                print(f"[DALICached] Using disk cache: {cache_dir}")

    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)

    if is_distributed and dist.is_initialized():
        dist.barrier()

    # Load or create caches
    if verbose:
        print(f"[DALICached] Cache directory: {cache_dir}")
        print(f"[DALICached] Cache resolution: {cache_size}x{cache_size}, output: {img_size}x{img_size}")

    train_images, train_labels = _cache_imagenet_hwc(
        data_root, "train", cache_size, cache_dir, rank, world_size, verbose
    )
    val_images, val_labels = _cache_imagenet_hwc(
        data_root, "val", cache_size, cache_dir, rank, world_size, verbose
    )

    if is_distributed and dist.is_initialized():
        dist.barrier()

    # Create external sources
    train_source = CachedExternalSource(
        train_images, train_labels, batch_size,
        shard_id=rank if is_distributed else 0,
        num_shards=world_size if is_distributed else 1,
        shuffle=True,
        seed=seed,
        drop_last=True,
    )

    val_source = CachedExternalSource(
        val_images, val_labels, batch_size,
        shard_id=0, num_shards=1,
        shuffle=False,
        seed=seed,
        drop_last=False,
    )

    # Build DALI pipelines
    train_pipe = cached_train_pipeline(
        source=train_source,
        img_size=img_size,
        color_jitter=color_jitter,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        random_seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    train_pipe.build()

    val_pipe = cached_val_pipeline(
        source=val_source,
        img_size=img_size,
        random_seed=seed,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        prefetch_queue_depth=prefetch_queue_depth,
    )
    val_pipe.build()

    # Wrap pipelines as PyTorch-compatible loaders
    train_loader = DALICachedWrapper(
        train_pipe,
        source=train_source,
        num_samples=len(train_source),
        last_batch_policy=LastBatchPolicy.DROP,
    )

    val_loader = DALICachedWrapper(
        val_pipe,
        source=val_source,
        num_samples=len(val_source),
        last_batch_policy=LastBatchPolicy.PARTIAL,
    )

    if verbose:
        print(f"[DALICached] Train: {len(train_source)} samples, {len(train_loader)} batches/epoch")
        print(f"[DALICached] Val: {len(val_source)} samples, {len(val_loader)} batches/epoch")
        aug_str = "RandomResizedCrop + Flip"
        if color_jitter:
            aug_str += f" + ColorJitter(b={brightness}, c={contrast}, s={saturation})"
        print(f"[DALICached] GPU augmentations: {aug_str}")

    return train_loader, val_loader


def is_dali_available() -> bool:
    """Check if DALI is available."""
    return DALI_AVAILABLE
