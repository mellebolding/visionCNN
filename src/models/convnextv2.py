"""ConvNeXtV2 model variants for CIFAR-style datasets."""
from .convnextv2_impl import ConvNeXtV2


def convnextv2_tiny(num_classes=10, **kwargs):
    """ConvNeXtV2 Tiny - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[2, 2, 2],
        dims=[80, 160, 320],
        num_classes=num_classes,
        drop_path_rate=0.1,
        **kwargs
    )


def convnextv2_base(num_classes=10, **kwargs):
    """ConvNeXtV2 Base - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[3, 3, 3],
        dims=[96, 192, 384],
        num_classes=num_classes,
        drop_path_rate=0.15,
        **kwargs
    )


def convnextv2_small(num_classes=10, **kwargs):
    """ConvNeXtV2 Small - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[1, 1, 1],
        dims=[64, 128, 256],
        num_classes=num_classes,
        drop_path_rate=0.05,
        **kwargs
    )