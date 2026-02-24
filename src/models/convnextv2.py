"""ConvNeXtV2 model variants for CIFAR-style and ImageNet datasets."""
from .convnextv2_impl import ConvNeXtV2


# --- CIFAR variants (3-stage, small stem) ---

def convnextv2_tiny(num_classes=10, norm_type="layernorm", **kwargs):
    """ConvNeXtV2 Tiny - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[2, 2, 2],
        dims=[80, 160, 320],
        num_classes=num_classes,
        drop_path_rate=0.1,
        norm_type=norm_type,
        stem_type="cifar",
        **kwargs
    )


def convnextv2_base(num_classes=10, norm_type="layernorm", **kwargs):
    """ConvNeXtV2 Base - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[3, 3, 3],
        dims=[96, 192, 384],
        num_classes=num_classes,
        drop_path_rate=0.15,
        norm_type=norm_type,
        stem_type="cifar",
        **kwargs
    )


def convnextv2_small(num_classes=10, norm_type="layernorm", **kwargs):
    """ConvNeXtV2 Small - optimized for CIFAR10/100 (32x32 images)."""
    return ConvNeXtV2(
        depths=[1, 1, 1],
        dims=[64, 128, 256],
        num_classes=num_classes,
        drop_path_rate=0.05,
        norm_type=norm_type,
        stem_type="cifar",
        **kwargs
    )


# --- ImageNet variants (4-stage, patchify stem) ---

def convnext_small(num_classes=100, norm_type="layernorm", **kwargs):
    """ConvNeXtV2 Pico-like for ImageNet (4-stage, ~9M params)."""
    return ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        num_classes=num_classes,
        drop_path_rate=0.1,
        norm_type=norm_type,
        stem_type="imagenet",
        **kwargs
    )


def convnext_medium(num_classes=100, norm_type="layernorm", **kwargs):
    """ConvNeXtV2 for ImageNet (4-stage, ~15M params)."""
    return ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[96, 192, 384, 768],
        num_classes=num_classes,
        drop_path_rate=0.15,
        norm_type=norm_type,
        stem_type="imagenet",
        **kwargs
    )
