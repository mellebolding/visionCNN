"""Normalization layer factory for configurable norm experiments."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class NoNorm(nn.Module):
    """Identity normalization (no-op). Replaces norm layers for ablation."""

    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, x):
        return x


class RMSNorm2d(nn.Module):
    """RMS Normalization for 2D feature maps (Zhang & Sennrich, 2019).

    Like LayerNorm but without mean centering — normalizes by the root mean
    square of activations over (C, H, W) per sample. Simpler and often
    faster than full LayerNorm.

    Used for VGG and ResNet where LayerNorm (GroupNorm(1,C)) also normalizes
    over (C, H, W), making the comparison fair.
    """

    def __init__(self, num_channels, eps=1e-6, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if bias else None
        self.eps = eps

    def forward(self, x):
        # x: (N, C, H, W) — compute RMS over C, H, W
        rms = x.float().pow(2).mean(dim=[1, 2, 3], keepdim=True).add(self.eps).sqrt()
        x = (x / rms) * self.weight[None, :, None, None]
        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x


class RMSNorm(nn.Module):
    """RMS Normalization over channel dimension only.

    Supports both channels_last (N, H, W, C) and channels_first (N, C, H, W)
    formats, normalizing over C only — matching F.layer_norm semantics.

    This is the correct RMSNorm variant for ConvNeXtV2, where LayerNorm
    normalizes per-spatial-position over channels only, NOT over (C, H, W).

    Args:
        dim: Number of channels.
        eps: Epsilon for numerical stability.
        data_format: 'channels_last' or 'channels_first'.
        bias: If True, add a learnable bias (default True, matching LayerNorm).
    """

    def __init__(self, dim, eps=1e-6, data_format="channels_last", bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            # x: (N, H, W, C) — normalize over C (last dim)
            rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
            x = (x / rms) * self.weight
            if self.bias is not None:
                x = x + self.bias
        else:
            # x: (N, C, H, W) — normalize over C (dim=1)
            rms = x.float().pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
            x = (x / rms) * self.weight[:, None, None]
            if self.bias is not None:
                x = x + self.bias[:, None, None]
        return x


class WSConv2d(nn.Conv2d):
    """Weight Standardized Conv2d (Qiao et al., 2019 / Brock et al., 2021).

    Standardizes the weight tensor (zero mean, unit variance per output filter)
    before each forward pass. This stabilizes training without batch-dependent
    normalization layers.
    """

    def forward(self, x):
        # Compute standardization in float32 to avoid AMP float16 underflow
        weight = self.weight.float()
        mean = weight.mean(dim=[1, 2, 3], keepdim=True)
        var = weight.var(dim=[1, 2, 3], keepdim=True)
        weight = ((weight - mean) / (var.sqrt() + 1e-5)).to(self.weight.dtype)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def adaptive_gradient_clip(model, clip_factor=0.01, eps=1e-3):
    """Adaptive Gradient Clipping (Brock et al., 2021).

    Clips gradients per-parameter based on the ratio of gradient norm to
    weight norm. Prevents training instability without damping learning.
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        p_norm = p.data.norm(2).clamp(min=eps)
        g_norm = p.grad.data.norm(2)
        max_norm = p_norm * clip_factor
        if g_norm > max_norm:
            p.grad.data.mul_(max_norm / (g_norm + 1e-6))


def get_norm_layer(name: str):
    """Return a norm layer constructor: norm_fn(num_channels) -> nn.Module.

    Args:
        name: One of 'batchnorm', 'layernorm', 'groupnorm', 'rmsnorm', 'nonorm', 'nonorm_ws'.

    Returns:
        Callable that takes num_channels and returns an nn.Module.
    """
    name = name.lower()
    if name == "batchnorm":
        return nn.BatchNorm2d
    elif name == "layernorm":
        # GroupNorm with 1 group is equivalent to LayerNorm over (C, H, W)
        return partial(nn.GroupNorm, 1)
    elif name == "groupnorm":
        return partial(nn.GroupNorm, 32)
    elif name == "rmsnorm":
        return RMSNorm2d
    elif name == "rmsnorm_bias":
        return partial(RMSNorm2d, bias=True)
    elif name in ("nonorm", "nonorm_ws"):
        return NoNorm
    else:
        raise ValueError(
            f"Unknown norm layer '{name}'. Choose from: batchnorm, layernorm, groupnorm, rmsnorm, rmsnorm_bias, nonorm, nonorm_ws"
        )
