#!/usr/bin/env python3
"""
Pre-cache ImageNet to disk/RAM for fast training.

Run this ONCE before training to create the cache files.
Then all DDP ranks can memory-map the same files.

Usage:
    # Cache to /dev/shm (fastest, but limited space)
    python scripts/precache_imagenet.py --root /path/to/imagenet --cache-dir /dev/shm

    # Cache to disk (slower but unlimited space)
    python scripts/precache_imagenet.py --root /path/to/imagenet --cache-dir /path/to/cache

    # Check cache status
    python scripts/precache_imagenet.py --root /path/to/imagenet --status

    # Clean up cache
    python scripts/precache_imagenet.py --clean --cache-dir /dev/shm
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm


def get_cache_paths(cache_dir: str, split: str, image_size: int):
    """Get cache file paths for a given split."""
    cache_name = f"imagenet_cache_{split}_{image_size}"
    return {
        'images': os.path.join(cache_dir, f"{cache_name}_images.npy"),
        'labels': os.path.join(cache_dir, f"{cache_name}_labels.npy"),
        'meta': os.path.join(cache_dir, f"{cache_name}_meta.txt"),
    }


def check_cache_status(cache_dir: str, image_size: int = 224):
    """Check if cache exists and print status."""
    print(f"\nCache directory: {cache_dir}")

    # Check disk space
    if os.path.exists(cache_dir):
        usage = shutil.disk_usage(cache_dir)
        print(f"Disk space: {usage.free/1e9:.1f} GB free / {usage.total/1e9:.1f} GB total")

    for split in ['train', 'val']:
        paths = get_cache_paths(cache_dir, split, image_size)

        if os.path.exists(paths['images']):
            size_gb = os.path.getsize(paths['images']) / 1e9
            arr = np.load(paths['images'], mmap_mode='r')
            print(f"\n{split}:")
            print(f"  ✓ Cache exists: {paths['images']}")
            print(f"  Shape: {arr.shape}")
            print(f"  Size: {size_gb:.1f} GB")
        else:
            print(f"\n{split}:")
            print(f"  ✗ Cache not found")


def create_cache(root: str, cache_dir: str, split: str, image_size: int = 224,
                 num_workers: int = 16, resize_size: int = 256):
    """Create cache for a single split."""
    paths = get_cache_paths(cache_dir, split, image_size)

    # Check if already exists
    if os.path.exists(paths['images']) and os.path.exists(paths['labels']):
        print(f"[{split}] Cache already exists, skipping")
        return

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ])

    data_path = os.path.join(root, split)
    if not os.path.exists(data_path):
        print(f"[{split}] ERROR: Path not found: {data_path}")
        return

    print(f"\n[{split}] Loading from: {data_path}")
    dataset = ImageFolder(data_path, transform=transform)
    print(f"[{split}] Found {len(dataset)} images")

    # Use num_workers=0 to avoid /dev/shm issues entirely
    # Slower but guaranteed to work
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,  # Single process - no shared memory needed
        pin_memory=False,
        shuffle=False,
    )

    images_list = []
    labels_list = []

    for imgs, lbls in tqdm(loader, desc=f"Loading {split}"):
        images_list.append(imgs)
        labels_list.append(lbls)

    images = torch.cat(images_list).numpy()
    labels = torch.cat(labels_list).numpy()

    print(f"[{split}] Tensor shape: {images.shape}")
    print(f"[{split}] Memory: {images.nbytes / 1e9:.1f} GB")

    # Save
    print(f"[{split}] Saving to: {paths['images']}")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(paths['images'], images)
    np.save(paths['labels'], labels)

    # Write metadata
    with open(paths['meta'], 'w') as f:
        f.write(f"shape={images.shape}\n")
        f.write(f"dtype={images.dtype}\n")

    print(f"[{split}] Done!")


def clean_cache(cache_dir: str, prefix: str = "imagenet_cache"):
    """Remove all cache files."""
    import glob
    files = glob.glob(os.path.join(cache_dir, f"{prefix}*"))

    if not files:
        print(f"No cache files found in {cache_dir}")
        return

    print(f"Found {len(files)} cache files:")
    total_size = 0
    for f in files:
        size = os.path.getsize(f)
        total_size += size
        print(f"  {f} ({size/1e9:.1f} GB)")

    response = input(f"\nDelete {total_size/1e9:.1f} GB? [y/N] ")
    if response.lower() == 'y':
        for f in files:
            os.remove(f)
            print(f"  Removed: {f}")
        print("Done!")
    else:
        print("Cancelled")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache ImageNet for fast training")
    parser.add_argument("--root", type=str, help="ImageNet root directory")
    parser.add_argument("--cache-dir", type=str, default="/dev/shm",
                        help="Cache directory (default: /dev/shm)")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader workers")
    parser.add_argument("--status", action="store_true", help="Check cache status")
    parser.add_argument("--clean", action="store_true", help="Clean up cache files")
    parser.add_argument("--train-only", action="store_true", help="Only cache training set")
    parser.add_argument("--val-only", action="store_true", help="Only cache validation set")
    args = parser.parse_args()

    # Auto-detect ImageNet root on guppy
    if args.root is None:
        import socket
        host = socket.gethostname().lower()
        if "guppy" in host:
            args.root = "/export/scratch1/home/melle/datasets/imagenet"
            print(f"Auto-detected ImageNet root: {args.root}")
        else:
            print("ERROR: Please specify --root")
            sys.exit(1)

    if args.clean:
        clean_cache(args.cache_dir)
        return

    if args.status:
        check_cache_status(args.cache_dir, args.image_size)
        return

    # Check available space
    if os.path.exists(args.cache_dir):
        free_gb = shutil.disk_usage(args.cache_dir).free / 1e9
        print(f"Cache directory: {args.cache_dir}")
        print(f"Free space: {free_gb:.1f} GB")

        needed = 180 if not (args.train_only or args.val_only) else (170 if args.train_only else 10)
        if free_gb < needed:
            print(f"WARNING: May not have enough space (need ~{needed} GB)")
            response = input("Continue anyway? [y/N] ")
            if response.lower() != 'y':
                sys.exit(1)

    # Create cache
    splits = []
    if args.train_only:
        splits = ['train']
    elif args.val_only:
        splits = ['val']
    else:
        splits = ['train', 'val']

    for split in splits:
        create_cache(
            root=args.root,
            cache_dir=args.cache_dir,
            split=split,
            image_size=args.image_size,
            num_workers=args.num_workers,
        )

    print("\n" + "="*50)
    print("Cache creation complete!")
    print("="*50)
    check_cache_status(args.cache_dir, args.image_size)


if __name__ == "__main__":
    main()
