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


class Derf2d(nn.Module):
    """Dynamic erf normalization for 2D feature maps (Chen et al., 2025).

    Pointwise function: output = erf(alpha * x + shift) * weight + bias
    where alpha and shift are learnable scalars, weight and bias are per-channel.
    Designed as a drop-in replacement for normalization layers.
    """

    def __init__(self, num_channels, alpha_init=0.5, shift_init=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.shift = nn.Parameter(torch.tensor(shift_init))
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        # x: (N, C, H, W)
        out = torch.erf(self.alpha * x + self.shift)
        return out * self.weight[None, :, None, None] + self.bias[None, :, None, None]


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


class _SeqBatchNorm(nn.Module):
    """BatchNorm1d adapted for token sequences (N, L, D).

    Permutes to (N, D, L) for BatchNorm1d, then permutes back.
    Normalizes each feature dimension across the batch and sequence positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x: (N, L, D)
        x = x.permute(0, 2, 1)  # → (N, D, L)
        x = self.bn(x)
        return x.permute(0, 2, 1)  # → (N, L, D)


class _SeqGroupNorm(nn.Module):
    """GroupNorm adapted for token sequences (N, L, D).

    Permutes to (N, D, L) for GroupNorm, then permutes back.
    Normalizes within groups of feature dimensions per token.
    """

    def __init__(self, dim: int, num_groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(min(num_groups, dim), dim)

    def forward(self, x):
        # x: (N, L, D)
        x = x.permute(0, 2, 1)  # → (N, D, L)
        x = self.gn(x)
        return x.permute(0, 2, 1)  # → (N, L, D)


class _SeqDerf(nn.Module):
    """Derf normalization for token sequences (N, L, D).

    1D analog of Derf2d: erf(alpha * x + shift) * weight + bias,
    applied elementwise over the D dimension. No statistics needed.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5, shift_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.shift = nn.Parameter(torch.tensor(shift_init))
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (N, L, D) — operates elementwise, weight/bias broadcast over D
        out = torch.erf(self.alpha * x + self.shift)
        return out * self.weight + self.bias


def get_norm_layer_seq(norm_type: str, dim: int) -> nn.Module:
    """Return a norm layer instance for transformer token sequences (N, L, D).

    All norms normalize over or operate on the D (embedding) dimension.

    Args:
        norm_type: One of 'layernorm', 'rmsnorm', 'batchnorm', 'groupnorm', 'derf'.
        dim: Embedding dimension D.

    Returns:
        nn.Module that accepts (N, L, D) and returns (N, L, D).
    """
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim=dim, data_format="channels_last")
    elif norm_type == "batchnorm":
        return _SeqBatchNorm(dim)
    elif norm_type == "groupnorm":
        return _SeqGroupNorm(dim)
    elif norm_type == "derf":
        return _SeqDerf(dim)
    else:
        raise ValueError(
            f"Unknown norm_type for sequences: '{norm_type}'. "
            "Choose from: layernorm, rmsnorm, batchnorm, groupnorm, derf"
        )


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
    elif name == "derf":
        return Derf2d
    elif name in ("nonorm", "nonorm_ws"):
        return NoNorm
    else:
        raise ValueError(
            f"Unknown norm layer '{name}'. Choose from: batchnorm, layernorm, groupnorm, rmsnorm, rmsnorm_bias, derf, nonorm, nonorm_ws"
        )
