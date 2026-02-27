"""RAM-cached ImageNet dataset for eliminating CPU bottleneck.

This module loads the entire ImageNet dataset into RAM as uint8 tensors,
eliminating disk I/O and JPEG decoding during training. Augmentations
are performed on GPU.

For DDP training with torchrun:
- Rank 0 loads data and saves to a shared temp file
- Other ranks wait, then memory-map the same file
- This ensures only ONE copy exists in RAM

Memory requirements (approximate):
- Train (1.28M images) at 224x224 uint8: ~170 GB
- Val (50K images) at 224x224 uint8: ~7 GB
- Total: ~177 GB RAM (shared across all DDP processes)

Usage:
    from src.datasets.cached_imagenet import build_cached_dataloader
    train_loader, val_loader, train_sampler, val_sampler = build_cached_dataloader(cfg, dist_manager)
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


class CachedImageNet(Dataset):
    """ImageNet dataset fully loaded in RAM as uint8 tensors.

    Images are stored as uint8 [0, 255] to minimize memory usage.
    Conversion to float and normalization happens on GPU during training.

    For DDP with torchrun: Uses numpy memmap for cross-process memory sharing.

    Args:
        root: Path to ImageNet root directory (containing train/ and val/)
        split: 'train' or 'val'
        image_size: Target image size (default: 224)
        num_workers: Workers for initial loading (default: 32)
        resize_size: Size to resize before crop (default: 256)
        verbose: Print progress (default: True)
        rank: DDP rank (0 loads data, others wait)
        world_size: DDP world size
        cache_dir: Directory for shared cache files (default: /dev/shm for RAM)
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 224,
        num_workers: int = 32,
        resize_size: int = 256,
        verbose: bool = True,
        rank: int = 0,
        world_size: int = 1,
        cache_dir: str = "/dev/shm",
    ):
        self.image_size = image_size
        self.split = split
        self.rank = rank
        self.world_size = world_size

        is_distributed = world_size > 1

        # Cache file paths
        # Default to user's scratch space if /dev/shm is too small
        if cache_dir == "/dev/shm":
            # Check if /dev/shm has enough space (need ~180GB)
            import shutil
            shm_free = shutil.disk_usage("/dev/shm").free / 1e9
            if shm_free < 180:
                # Fall back to scratch directory next to imagenet
                cache_dir = os.path.join(os.path.dirname(root), "imagenet_cache")
                if rank == 0:
                    print(f"[CachedImageNet] /dev/shm too small ({shm_free:.0f}GB), using: {cache_dir}")

        if rank == 0:
            os.makedirs(cache_dir, exist_ok=True)
        cache_name = f"imagenet_cache_{split}_{image_size}"
        self.images_path = os.path.join(cache_dir, f"{cache_name}_images.npy")
        self.labels_path = os.path.join(cache_dir, f"{cache_name}_labels.npy")
        self.meta_path = os.path.join(cache_dir, f"{cache_name}_meta.txt")

        if is_distributed:
            self._load_distributed(root, split, image_size, num_workers, resize_size, verbose)
        else:
            self._load_single(root, split, image_size, num_workers, resize_size, verbose)

    def _read_shape_from_meta(self):
        """Read image array shape from metadata file."""
        with open(self.meta_path, 'r') as f:
            content = f.read().strip()
        # Handle both formats: "N,C,H,W" or "key=value" lines
        if '=' in content:
            for line in content.split('\n'):
                if line.startswith('shape='):
                    return eval(line.split('=', 1)[1])
        else:
            parts = content.split(',')
            return tuple(int(p) for p in parts)

    def _load_single(
        self,
        root: str,
        split: str,
        image_size: int,
        num_workers: int,
        resize_size: int,
        verbose: bool
    ):
        """Load data for single GPU training."""
        # Check if cache exists (need meta file for memmap shape)
        if os.path.exists(self.images_path) and os.path.exists(self.labels_path) and os.path.exists(self.meta_path):
            if verbose:
                print(f"[CachedImageNet] Loading {split} from cache: {self.images_path}")

            # Read shape from metadata (memmap files don't have headers)
            shape = self._read_shape_from_meta()

            # Use memmap directly (cache was created with np.memmap, not np.save)
            self.images = torch.from_numpy(np.memmap(self.images_path, dtype=np.uint8, mode='r', shape=shape))
            self.labels = torch.from_numpy(np.load(self.labels_path))  # labels were saved with np.save
            if verbose:
                print(f"[CachedImageNet] Loaded {len(self.images)} images from cache")
            return

        # Load fresh
        images, labels = self._load_from_disk(root, split, image_size, num_workers, resize_size, verbose)
        self.images = images
        self.labels = labels

    def _load_distributed(
        self,
        root: str,
        split: str,
        image_size: int,
        num_workers: int,
        resize_size: int,
        verbose: bool
    ):
        """Load data for DDP training with shared memory."""
        cache_exists = os.path.exists(self.images_path) and os.path.exists(self.labels_path)

        if self.rank == 0:
            if cache_exists:
                if verbose:
                    print(f"[CachedImageNet] Cache exists: {self.images_path}")
            else:
                if verbose:
                    print(f"[CachedImageNet] Rank 0: Loading {split} and creating shared cache...")

                images, labels = self._load_from_disk(root, split, image_size, num_workers, resize_size, verbose)

                # Save to /dev/shm (RAM-backed filesystem)
                if verbose:
                    print(f"[CachedImageNet] Saving to shared memory: {self.images_path}")
                np.save(self.images_path, images.numpy())
                np.save(self.labels_path, labels.numpy())

                # Write metadata to signal completion
                with open(self.meta_path, 'w') as f:
                    f.write(f"{len(images)},{images.shape[1]},{images.shape[2]},{images.shape[3]}")

                if verbose:
                    print(f"[CachedImageNet] Cache created successfully")

        # Barrier: wait for rank 0 to finish
        if dist.is_initialized():
            dist.barrier()

        # All ranks: memory-map the shared cache
        if verbose or self.rank == 0:
            print(f"[CachedImageNet] Rank {self.rank}: Attaching to shared cache")

        # Memory-map mode 'r' = read-only, shared across processes
        shape = self._read_shape_from_meta()
        self.images = torch.from_numpy(np.memmap(self.images_path, dtype=np.uint8, mode='r', shape=shape))
        self.labels = torch.from_numpy(np.load(self.labels_path))  # labels were saved with np.save

        if self.rank == 0 and verbose:
            print(f"[CachedImageNet] All ranks attached. Shape: {self.images.shape}")

    def _load_from_disk(
        self,
        root: str,
        split: str,
        image_size: int,
        num_workers: int,
        resize_size: int,
        verbose: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load images from disk into tensors."""
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.PILToTensor(),  # Returns uint8 tensor [0, 255]
        ])

        data_path = os.path.join(root, split)
        if not os.path.exists(data_path):
            raise ValueError(f"ImageNet {split} directory not found: {data_path}")

        dataset = ImageFolder(data_path, transform=transform)

        if verbose:
            print(f"[CachedImageNet] Loading {split} set from disk...")
            print(f"[CachedImageNet] Path: {data_path}")
            print(f"[CachedImageNet] Images: {len(dataset)}")

        # Use num_workers=0 to avoid /dev/shm issues entirely
        loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=0,  # Single process - no shared memory needed
            pin_memory=False,
            shuffle=False,
        )

        images_list = []
        labels_list = []

        iterator = tqdm(loader, desc=f"Caching {split}") if verbose else loader
        for imgs, lbls in iterator:
            images_list.append(imgs)
            labels_list.append(lbls)

        images = torch.cat(images_list)  # [N, 3, H, W] uint8
        labels = torch.cat(labels_list)  # [N] int64

        if verbose:
            mem_gb = images.nbytes / 1e9
            print(f"[CachedImageNet] Loaded {len(images)} images ({mem_gb:.1f} GB)")

        return images, labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert memmap numpy array to tensor on access
        img = self.images[idx]
        lbl = self.labels[idx]
        # Ensure we return proper tensors (memmap might return numpy)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.array(img))
        if not isinstance(lbl, torch.Tensor):
            lbl = torch.tensor(lbl)
        return img, lbl


class FastCachedLoader:
    """High-throughput loader for cached uint8 data, bypassing DataLoader overhead.

    Uses double-buffered batch indexing + pre-pinned buffers + CUDA stream
    prefetch to overlap data loading with GPU compute.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int,
        device: torch.device,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        n = len(images)
        self.num_samples = n // world_size if drop_last else (n + world_size - 1) // world_size
        self.total_size = self.num_samples * world_size

        # Double-buffered pinned memory: fill buf[next] while GPU reads buf[current]
        C, H, W = images.shape[1], images.shape[2], images.shape[3]
        self._pinned_images = [
            torch.empty(batch_size, C, H, W, dtype=torch.uint8, pin_memory=True),
            torch.empty(batch_size, C, H, W, dtype=torch.uint8, pin_memory=True),
        ]
        self._pinned_labels = [
            torch.empty(batch_size, dtype=torch.int64, pin_memory=True),
            torch.empty(batch_size, dtype=torch.int64, pin_memory=True),
        ]
        self._stream = torch.cuda.Stream(device=device)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.images), generator=g)
        else:
            indices = torch.arange(len(self.images))
        if len(indices) < self.total_size:
            indices = torch.cat([indices, indices[:self.total_size - len(indices)]])
        indices = indices[:self.total_size]
        return indices[self.rank::self.world_size]

    def __len__(self):
        return self.num_samples // self.batch_size

    def _load_batch(self, batch_idx, buf_idx):
        """Load batch into pinned buffer[buf_idx] and start async H2D."""
        torch.index_select(self.images, 0, batch_idx, out=self._pinned_images[buf_idx])
        torch.index_select(self.labels, 0, batch_idx, out=self._pinned_labels[buf_idx])
        with torch.cuda.stream(self._stream):
            img_gpu = self._pinned_images[buf_idx].to(self.device, non_blocking=True)
            lbl_gpu = self._pinned_labels[buf_idx].to(self.device, non_blocking=True)
        return img_gpu, lbl_gpu

    def __iter__(self):
        indices = self._get_indices()
        bs = self.batch_size
        n_batches = len(indices) // bs
        if n_batches == 0:
            return

        # Prefetch first batch into buffer 0
        next_img, next_lbl = self._load_batch(indices[:bs], 0)

        for i in range(n_batches):
            # Wait for the prefetched batch to finish H2D
            self._stream.synchronize()
            cur_img, cur_lbl = next_img, next_lbl

            # Start prefetching next batch into the other buffer (while GPU processes current)
            if i + 1 < n_batches:
                buf = (i + 1) % 2
                next_img, next_lbl = self._load_batch(
                    indices[(i + 1) * bs:(i + 2) * bs], buf
                )

            yield cur_img, cur_lbl


def cleanup_cache(cache_dir: str = "/dev/shm", prefix: str = "imagenet_cache"):
    """Clean up cached files from shared memory."""
    import glob
    files = glob.glob(os.path.join(cache_dir, f"{prefix}*"))
    for f in files:
        try:
            os.remove(f)
            print(f"Removed: {f}")
        except Exception as e:
            print(f"Failed to remove {f}: {e}")


def build_cached_dataloader(
    cfg,
    dist_manager: Optional["DistributedManager"] = None  # type: ignore
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """Build train and validation dataloaders with RAM-cached ImageNet.

    For DDP: Data is loaded once by rank 0, saved to /dev/shm, and memory-mapped
    by all ranks. This ensures only ONE copy of the data exists in RAM.

    Args:
        cfg: Configuration dictionary
        dist_manager: Optional DistributedManager for DDP training

    Returns:
        Tuple of (train_loader, val_loader, train_sampler, val_sampler)
    """
    root = cfg['data']['root']
    image_size = cfg['data'].get('img_size', 224)
    batch_size = cfg['training']['batch_size']
    seed = cfg.get('training', {}).get('seed', 42)
    cache_workers = cfg.get('data', {}).get('cache_workers', 32)

    # DDP info
    rank = dist_manager.rank if dist_manager else 0
    world_size = dist_manager.world_size if dist_manager else 1
    is_distributed = dist_manager is not None and dist_manager.is_distributed
    verbose = rank == 0

    if verbose:
        if is_distributed:
            print(f"[CachedImageNet] DDP mode with {world_size} processes")
            print(f"[CachedImageNet] Using /dev/shm for cross-process shared memory")
        else:
            print(f"[CachedImageNet] Single GPU mode")

    # Load datasets (rank 0 creates cache, others attach)
    train_dataset = CachedImageNet(
        root=root,
        split='train',
        image_size=image_size,
        num_workers=cache_workers,
        verbose=verbose,
        rank=rank,
        world_size=world_size,
    )

    val_dataset = CachedImageNet(
        root=root,
        split='val',
        image_size=image_size,
        num_workers=cache_workers,
        verbose=verbose,
        rank=rank,
        world_size=world_size,
    )

    # Final sync
    if is_distributed and dist.is_initialized():
        dist.barrier()
        if verbose:
            print(f"[CachedImageNet] All ranks ready")

    # Determine device for this rank
    local_rank = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{local_rank}")

    # Use FastCachedLoader: batch indexing + pre-pinned buffer + CUDA stream prefetch
    # This bypasses DataLoader overhead (per-item __getitem__, collation, pin_memory)
    if verbose:
        print(f"[CachedImageNet] Using FastCachedLoader (batch indexing + pinned prefetch)")

    train_loader = FastCachedLoader(
        images=train_dataset.images,
        labels=train_dataset.labels,
        batch_size=batch_size,
        device=device,
        shuffle=True,
        drop_last=True,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    val_loader = FastCachedLoader(
        images=val_dataset.images,
        labels=val_dataset.labels,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        drop_last=False,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )

    # Return None for samplers — FastCachedLoader handles sharding internally
    return train_loader, val_loader, None, None
