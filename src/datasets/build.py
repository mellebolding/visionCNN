"""Dataset factory for building train/val dataloaders."""
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms


def get_transforms(cfg, is_train=True):
    """Build transforms based on config."""
    img_size = cfg.get("data", {}).get("img_size", 32)
    
    if is_train:
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        # Optional: Add more augmentations
        if cfg.get("data", {}).get("auto_augment", False):
            transform_list.insert(0, transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10
            ))
    else:
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        split = "train" if is_train else "val"
        dataset = torchvision.datasets.ImageNet(
            root=data_root,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def build_dataloader(cfg):
    """Build train and validation dataloaders."""
    batch_size = cfg.get("training", {}).get("batch_size", 64)
    num_workers = cfg.get("data", {}).get("num_workers", 4)
    pin_memory = cfg.get("data", {}).get("pin_memory", True)
    
    train_dataset = build_dataset(cfg, is_train=True)
    val_dataset = build_dataset(cfg, is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


def get_num_classes(dataset_name):
    """Return number of classes for a dataset."""
    num_classes_map = {
        "cifar10": 10,
        "cifar100": 100,
        "svhn": 10,
        "imagenet": 1000,
    }
    return num_classes_map.get(dataset_name.lower(), 10)
