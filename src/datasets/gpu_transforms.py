"""GPU-accelerated transforms using native PyTorch.

This module provides data augmentation transforms that run on GPU,
reducing CPU bottleneck during training.
"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class GPUTransforms(nn.Module):
    """GPU-accelerated transforms using native PyTorch.

    Applies normalization and augmentations on GPU to reduce CPU load.
    All operations are differentiable but run in eval mode during training.
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        random_erasing_p=0.0,
        random_erasing_scale=(0.02, 0.33),
        random_erasing_ratio=(0.3, 3.3),
    ):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

        self.random_erasing_p = random_erasing_p
        self.random_erasing_scale = random_erasing_scale
        self.random_erasing_ratio = random_erasing_ratio

    def normalize(self, x):
        """Normalize tensor with mean and std."""
        return (x - self.mean) / self.std

    def random_erasing(self, x):
        """Apply random erasing augmentation on GPU."""
        if self.random_erasing_p == 0 or not self.training:
            return x

        batch_size, _, h, w = x.shape

        for i in range(batch_size):
            if torch.rand(1).item() > self.random_erasing_p:
                continue

            # Random area
            area = h * w
            target_area = torch.empty(1).uniform_(
                self.random_erasing_scale[0], self.random_erasing_scale[1]
            ).item() * area

            # Random aspect ratio
            aspect_ratio = torch.empty(1).uniform_(
                self.random_erasing_ratio[0], self.random_erasing_ratio[1]
            ).item()

            h_erase = int(round((target_area * aspect_ratio) ** 0.5))
            w_erase = int(round((target_area / aspect_ratio) ** 0.5))

            if h_erase < h and w_erase < w:
                top = torch.randint(0, h - h_erase + 1, (1,)).item()
                left = torch.randint(0, w - w_erase + 1, (1,)).item()

                # Fill with random values
                x[i, :, top:top+h_erase, left:left+w_erase] = torch.empty(
                    (3, h_erase, w_erase),
                    dtype=x.dtype,
                    device=x.device
                ).normal_()

        return x

    def forward(self, x):
        """Apply GPU transforms.

        Args:
            x: Tensor of shape (B, C, H, W) with values in [0, 1]

        Returns:
            Normalized and augmented tensor
        """
        # Normalize (always applied)
        x = self.normalize(x)

        # Optional: Random erasing (applied during training if enabled)
        if self.random_erasing_p > 0:
            x = self.random_erasing(x)

        return x


def build_gpu_transforms(cfg, is_train=True):
    """Build GPU transforms from config.

    Args:
        cfg: Configuration dictionary
        is_train: Whether this is for training (enables augmentations)

    Returns:
        GPUTransforms module
    """
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("dataset", "cifar10").lower()

    # Use ImageNet normalization for ImageNet, otherwise simple normalization
    if dataset_name == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    # Random erasing (only during training)
    random_erasing_p = 0.0
    if is_train and data_cfg.get("random_erasing", False):
        random_erasing_p = 0.25

    return GPUTransforms(
        mean=mean,
        std=std,
        random_erasing_p=random_erasing_p,
    )
