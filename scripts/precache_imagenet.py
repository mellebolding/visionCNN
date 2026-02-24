#!/usr/bin/env python3
"""
Pre-cache ImageNet to disk for fast training.

This script saves images incrementally to avoid OOM during the save step.
Uses memory-mapped files to write directly to disk without loading everything into RAM.

Usage:
    python scripts/precache_imagenet.py --cache-dir /path/to/cache

    # Check cache status
    python scripts/precache_imagenet.py --status

    # Clean up cache
    python scripts/precache_imagenet.py --clean
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

    if os.path.exists(cache_dir):
        usage = shutil.disk_usage(cache_dir)
        print(f"Disk space: {usage.free/1e9:.1f} GB free / {usage.total/1e9:.1f} GB total")

    for split in ['train', 'val']:
        paths = get_cache_paths(cache_dir, split, image_size)

        if os.path.exists(paths['images']) and os.path.exists(paths['meta']):
            size_gb = os.path.getsize(paths['images']) / 1e9

            # Read shape from metadata (memmap files don't have headers)
            meta = {}
            with open(paths['meta'], 'r') as f:
                for line in f:
                    key, val = line.strip().split('=', 1)
                    meta[key] = val
            shape = eval(meta['shape'])  # e.g., "(1281167, 3, 224, 224)"

            # Use memmap to verify file is readable
            arr = np.memmap(paths['images'], dtype=np.uint8, mode='r', shape=shape)
            print(f"\n{split}:")
            print(f"  ✓ Cache exists: {paths['images']}")
            print(f"  Shape: {arr.shape}")
            print(f"  Size: {size_gb:.1f} GB")
            del arr  # Close memmap
        else:
            print(f"\n{split}:")
            print(f"  ✗ Cache not found")


def create_cache_incremental(root: str, cache_dir: str, split: str, image_size: int = 224,
                              resize_size: int = 256):
    """Create cache incrementally using memory-mapped files.

    This avoids OOM by writing directly to disk instead of accumulating in RAM.
    """
    paths = get_cache_paths(cache_dir, split, image_size)

    # Check if already exists
    if os.path.exists(paths['images']) and os.path.exists(paths['labels']):
        print(f"[{split}] Cache already exists, skipping")
        return True

    # Setup transform
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.PILToTensor(),
    ])

    data_path = os.path.join(root, split)
    if not os.path.exists(data_path):
        print(f"[{split}] ERROR: Path not found: {data_path}")
        return False

    print(f"\n[{split}] Loading from: {data_path}")
    dataset = ImageFolder(data_path, transform=transform)
    num_images = len(dataset)
    print(f"[{split}] Found {num_images} images")

    # Calculate total size and create memory-mapped file
    # Shape: [N, 3, H, W] for images, [N] for labels
    img_shape = (num_images, 3, image_size, image_size)
    img_size_gb = np.prod(img_shape) / 1e9  # uint8 = 1 byte
    print(f"[{split}] Output shape: {img_shape}")
    print(f"[{split}] Output size: {img_size_gb:.1f} GB")

    # Check disk space
    disk_free = shutil.disk_usage(cache_dir).free / 1e9
    if disk_free < img_size_gb * 1.1:  # 10% buffer
        print(f"[{split}] ERROR: Not enough disk space ({disk_free:.1f} GB free, need {img_size_gb:.1f} GB)")
        return False

    os.makedirs(cache_dir, exist_ok=True)

    # Create memory-mapped files for writing
    print(f"[{split}] Creating memory-mapped file: {paths['images']}")
    images_mmap = np.memmap(paths['images'], dtype=np.uint8, mode='w+', shape=img_shape)
    labels_arr = np.zeros(num_images, dtype=np.int64)

    # Use single-process DataLoader (no shared memory issues)
    loader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=0,
        shuffle=False,
    )

    # Write directly to memory-mapped file
    idx = 0
    for imgs, lbls in tqdm(loader, desc=f"Caching {split}"):
        batch_size = imgs.shape[0]
        images_mmap[idx:idx + batch_size] = imgs.numpy()
        labels_arr[idx:idx + batch_size] = lbls.numpy()
        idx += batch_size

        # Periodically flush to disk
        if idx % 10000 == 0:
            images_mmap.flush()

    # Final flush and save labels
    print(f"[{split}] Flushing to disk...")
    images_mmap.flush()
    del images_mmap  # Close the memmap

    print(f"[{split}] Saving labels...")
    np.save(paths['labels'], labels_arr)

    # Write metadata
    with open(paths['meta'], 'w') as f:
        f.write(f"num_images={num_images}\n")
        f.write(f"shape={img_shape}\n")
        f.write(f"dtype=uint8\n")

    print(f"[{split}] Done!")
    return True


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
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Cache directory (default: next to ImageNet root)")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--status", action="store_true", help="Check cache status")
    parser.add_argument("--clean", action="store_true", help="Clean up cache files")
    parser.add_argument("--train-only", action="store_true", help="Only cache training set")
    parser.add_argument("--val-only", action="store_true", help="Only cache validation set")
    args = parser.parse_args()

    # Auto-detect ImageNet root from machine config
    if args.root is None:
        from pathlib import Path
        import yaml
        import socket

        machine_dir = Path(__file__).parent.parent / "configs" / "machines"
        env_machine = os.environ.get("VCNN_MACHINE", "").lower().strip()
        detected_root = None

        if env_machine:
            machine_file = machine_dir / f"{env_machine}.yaml"
            if machine_file.exists():
                with open(machine_file) as f:
                    detected_root = yaml.safe_load(f).get("paths", {}).get("imagenet_root")
        if detected_root is None:
            host = socket.gethostname().lower()
            for yaml_file in sorted(machine_dir.glob("*.yaml")):
                with open(yaml_file) as f:
                    candidate = yaml.safe_load(f)
                match = candidate.get("machine", {}).get("hostname_match", "")
                if match and match in host:
                    detected_root = candidate.get("paths", {}).get("imagenet_root")
                    break

        if detected_root:
            args.root = detected_root
            print(f"Auto-detected ImageNet root: {args.root}")
        else:
            print("ERROR: Please specify --root (no machine config matched)")
            sys.exit(1)

    # Default cache dir
    if args.cache_dir is None:
        args.cache_dir = os.path.join(os.path.dirname(args.root), "imagenet_cache")

    if args.clean:
        clean_cache(args.cache_dir)
        return

    if args.status:
        check_cache_status(args.cache_dir, args.image_size)
        return

    print(f"Cache directory: {args.cache_dir}")
    print(f"Free space: {shutil.disk_usage(args.cache_dir if os.path.exists(args.cache_dir) else os.path.dirname(args.cache_dir)).free/1e9:.1f} GB")

    # Create cache
    splits = []
    if args.train_only:
        splits = ['train']
    elif args.val_only:
        splits = ['val']
    else:
        splits = ['train', 'val']

    for split in splits:
        success = create_cache_incremental(
            root=args.root,
            cache_dir=args.cache_dir,
            split=split,
            image_size=args.image_size,
        )
        if not success:
            print(f"\n[{split}] FAILED")
            sys.exit(1)

    print("\n" + "="*50)
    print("Cache creation complete!")
    print("="*50)
    check_cache_status(args.cache_dir, args.image_size)


if __name__ == "__main__":
    main()
