"""Dataset factory for building train/val dataloaders."""
import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset, DistributedSampler
import torchvision
import torchvision.transforms as transforms


def get_transforms(cfg, is_train=True):
    """Build transforms based on config."""
    img_size = cfg.get("data", {}).get("img_size", 32)
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "cifar10").lower()
    
    # Use ImageNet normalization for ImageNet, otherwise use simple normalization
    if dataset_name == "imagenet":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # ImageNet-style transforms (larger images, need resize)
    if dataset_name == "imagenet" or img_size >= 224:
        if is_train:
            transform_list = [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
            
            # Optional: AutoAugment for ImageNet
            if data_cfg.get("auto_augment", False):
                transform_list.append(transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.IMAGENET
                ))
            
            # Optional: RandAugment
            if data_cfg.get("rand_augment", False):
                n_ops = data_cfg.get("rand_augment_n", 2)
                magnitude = data_cfg.get("rand_augment_m", 9)
                transform_list.append(transforms.RandAugment(num_ops=n_ops, magnitude=magnitude))
            
            # Optional: Color jitter
            if data_cfg.get("color_jitter", False):
                brightness = data_cfg.get("color_jitter_brightness", 0.4)
                contrast = data_cfg.get("color_jitter_contrast", 0.4)
                saturation = data_cfg.get("color_jitter_saturation", 0.4)
                hue = data_cfg.get("color_jitter_hue", 0.1)
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast,
                    saturation=saturation, hue=hue
                ))
            
            transform_list.append(transforms.ToTensor())
            
            # Optional: Random erasing
            if data_cfg.get("random_erasing", False):
                transform_list.append(transforms.RandomErasing(p=0.25))
            
            transform_list.append(normalize)
        else:
            # Validation: resize to 256, center crop to 224
            resize_size = int(img_size / 0.875)  # 256 for img_size=224
            transform_list = [
                transforms.Resize(resize_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
    else:
        # CIFAR-style transforms (small images, use padding)
        if is_train:
            transform_list = []
            
            # Optional: AutoAugment (applied before other augmentations)
            if data_cfg.get("auto_augment", False):
                transform_list.append(transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.CIFAR10
                ))
            
            # Optional: RandAugment (alternative to AutoAugment)
            if data_cfg.get("rand_augment", False):
                n_ops = data_cfg.get("rand_augment_n", 2)
                magnitude = data_cfg.get("rand_augment_m", 9)
                transform_list.append(transforms.RandAugment(num_ops=n_ops, magnitude=magnitude))
            
            # Basic augmentations
            transform_list.extend([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(img_size, padding=4),
            ])
            
            # Optional: Color jitter
            if data_cfg.get("color_jitter", False):
                brightness = data_cfg.get("color_jitter_brightness", 0.4)
                contrast = data_cfg.get("color_jitter_contrast", 0.4)
                saturation = data_cfg.get("color_jitter_saturation", 0.4)
                hue = data_cfg.get("color_jitter_hue", 0.1)
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast,
                    saturation=saturation, hue=hue
                ))
            
            # ToTensor and optional random erasing
            transform_list.append(transforms.ToTensor())
            
            if data_cfg.get("random_erasing", False):
                transform_list.append(transforms.RandomErasing(p=0.25))
            
            transform_list.append(normalize)
        else:
            transform_list = [
                transforms.ToTensor(),
                normalize,
            ]
    
    return transforms.Compose(transform_list)


def build_dataset(cfg, is_train=True):
    """Build dataset based on config."""
    dataset_name = cfg.get("data", {}).get("dataset", "cifar10").lower()
    data_root = cfg.get("data", {}).get("root", "./data")
    transform = get_transforms(cfg, is_train=is_train)
    
    if dataset_name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=is_train,
            download=True,
            transform=transform
        )
    elif dataset_name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=data_root,
            train=is_train,
            download=True,
            transform=transform
        )
    elif dataset_name == "svhn":
        split = "train" if is_train else "test"
        dataset = torchvision.datasets.SVHN(
            root=data_root,
            split=split,
            download=True,
            transform=transform
        )
    elif dataset_name == "imagenet":
        # Use ImageFolder for already-extracted ImageNet
        split = "train" if is_train else "val"
        data_path = os.path.join(data_root, split)
        
        # Check if path exists
        if not os.path.exists(data_path):
            raise ValueError(f"ImageNet split directory not found: {data_path}")
        
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def subsample_dataset(dataset, fraction: float, seed: int = 42):
    """Subsample a dataset to a fraction of its original size.
    
    Args:
        dataset: The dataset to subsample
        fraction: Fraction of data to keep (0.0 to 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        Subset of the original dataset
    """
    if fraction >= 1.0:
        return dataset
    
    n_samples = len(dataset)
    n_subset = int(n_samples * fraction)
    
    # Use a generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator)[:n_subset].tolist()
    
    return Subset(dataset, indices)


def build_dataloader(
    cfg,
    dist_manager: Optional["DistributedManager"] = None  # type: ignore
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    """Build train and validation dataloaders.
    
    Args:
        cfg: Configuration dictionary
        dist_manager: Optional DistributedManager for DDP training
        
    Returns:
        Tuple of (train_loader, val_loader, train_sampler, val_sampler)
        Samplers are returned so set_epoch can be called during training
    """
    batch_size = cfg.get("training", {}).get("batch_size", 64)
    num_workers = cfg.get("data", {}).get("num_workers", 4)
    pin_memory = cfg.get("data", {}).get("pin_memory", True)
    train_fraction = cfg.get("data", {}).get("train_fraction", 1.0)
    seed = cfg.get("training", {}).get("seed", 42)
    
    train_dataset = build_dataset(cfg, is_train=True)
    val_dataset = build_dataset(cfg, is_train=False)
    
    # Subsample training data if requested
    if train_fraction < 1.0:
        train_dataset = subsample_dataset(train_dataset, train_fraction, seed=seed)
    
    # Create distributed samplers if using DDP
    train_sampler = None
    val_sampler = None
    
    if dist_manager is not None and dist_manager.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=dist_manager.world_size,
            rank=dist_manager.rank,
            shuffle=True,
            seed=seed
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist_manager.world_size,
            rank=dist_manager.rank,
            shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def get_num_classes(dataset_name):
    """Return number of classes for a dataset."""
    num_classes_map = {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "imagenet": 1000,
    }
    return num_classes_map.get(dataset_name.lower(), 10)
