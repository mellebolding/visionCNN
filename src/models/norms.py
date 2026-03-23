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
        # Compute mean-of-squares in input dtype to avoid a large float32 copy,
        # then upcast the scalar result for numerical stability of sqrt.
        rms = x.pow(2).mean(dim=[1, 2, 3], keepdim=True).float().add(self.eps).sqrt()
        x = (x / rms.to(x.dtype)) * self.weight[None, :, None, None]
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


def _derf_vp_weight_init(alpha: float) -> float:
    """Variance-preserving weight initializer for Derf.

    Computes the scaling factor so that erf(alpha*x)*weight has unit variance
    at init when x ~ N(0,1). Uses the closed-form result:

        Var[erf(alpha*X)] = (2/pi) * arcsin(2*alpha^2 / (1 + 2*alpha^2))

    where X ~ N(0,1). Returns 1/std so that weight*erf(alpha*x) has std=1.
    """
    import math
    var = (2.0 / math.pi) * math.asin(2.0 * alpha ** 2 / (1.0 + 2.0 * alpha ** 2))
    return 1.0 / math.sqrt(var + 1e-8)


class Derf2d(nn.Module):
    """Dynamic erf normalization for 2D feature maps (Chen et al., 2025).

    Pointwise function: output = erf(alpha * x + shift) * weight + bias
    where alpha and shift are learnable scalars, weight and bias are per-channel.
    Designed as a drop-in replacement for normalization layers.

    NOTE: This original version uses weight_init=1.0, which does NOT preserve
    activation variance through depth. Use Derf2dVP for variance-preserving init.
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


class Derf2dVP(nn.Module):
    """Variance-Preserving Derf normalization for 2D feature maps.

    Normalizes input over (C, H, W) per sample (like LayerNorm for CNNs),
    then applies the learnable erf transformation. Without the pre-normalization
    step, conv outputs are unscaled and erf saturates early in training,
    causing complete training collapse in ResNet-50.

    alpha controls nonlinearity: 0 → linear (≈ LayerNorm), large → hard sign.
    VP weight init ensures output variance ≈ 1 at initialization.
    """

    def __init__(self, num_channels, alpha_init=0.5, shift_init=0.0, eps=1e-5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.shift = nn.Parameter(torch.tensor(float(shift_init)))
        w_init = _derf_vp_weight_init(alpha_init)
        self.weight = nn.Parameter(torch.full((num_channels,), w_init))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (N, C, H, W)
        # Normalize per sample over (C, H, W) using fused group_norm (no large intermediates)
        x_norm = F.group_norm(x, 1, eps=self.eps)
        out = torch.erf(self.alpha * x_norm + self.shift)
        return out * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class LocalNorm2d(nn.Module):
    """LocalNorm for 2D feature maps (Yin et al., 2019, arXiv 1902.06550).

    Training: batch split into K groups; each group normalized with its own
    (mu_k, sigma_k) computed over (N/K, H, W) per channel, and its own
    learnable gamma_k, beta_k parameters (shape: K x C).
    Inference: running statistics accumulated during training (like BatchNorm),
    with parameters averaged across groups.

    K controls regularization strength — larger K approximates InstanceNorm,
    smaller K approximates BatchNorm.
    """

    def __init__(self, num_channels, K=2, eps=1e-5, momentum=0.1):
        super().__init__()
        self.K = K
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(K, num_channels))   # K x C
        self.bias   = nn.Parameter(torch.zeros(K, num_channels))  # K x C
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

    def forward(self, x):
        N, C, H, W = x.shape
        if self.training:
            K = min(self.K, N)
            splits = torch.chunk(x, K, dim=0)
            outs = []
            for k, x_k in enumerate(splits):
                mean = x_k.mean(dim=[0, 2, 3], keepdim=True)
                std  = x_k.var(dim=[0, 2, 3], keepdim=True, unbiased=False).float().add(self.eps).sqrt().to(x.dtype)
                w = self.weight[k % self.K][None, :, None, None]
                b = self.bias[k   % self.K][None, :, None, None]
                outs.append((x_k - mean) / std * w + b)
            # Update running stats from full-batch mean/var (per channel)
            with torch.no_grad():
                batch_mean = x.mean(dim=[0, 2, 3])
                batch_var  = x.var(dim=[0, 2, 3], unbiased=False)
                self.running_mean.lerp_(batch_mean.float(), self.momentum)
                self.running_var.lerp_(batch_var.float(), self.momentum)
            return torch.cat(outs, dim=0)
        else:
            # Inference: use running statistics, averaged parameters
            mean = self.running_mean.to(x.dtype)[None, :, None, None]
            std  = (self.running_var + self.eps).sqrt().to(x.dtype)[None, :, None, None]
            w = self.weight.mean(0)[None, :, None, None]
            b = self.bias.mean(0)[None, :, None, None]
            return (x - mean) / std * w + b


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

    NOTE: weight_init=1.0 does not preserve variance — use _SeqDerfVP instead.
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


class _SeqDerfVP(nn.Module):
    """Variance-Preserving Derf for token sequences (N, L, D).

    Same as _SeqDerf but with weight initialized to 1/std(erf(alpha*x)) so
    that activation variance is preserved at initialization.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5, shift_init: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.shift = nn.Parameter(torch.tensor(float(shift_init)))
        w_init = _derf_vp_weight_init(alpha_init)
        self.weight = nn.Parameter(torch.full((dim,), w_init))
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
    elif norm_type == "derf_vp":
        return _SeqDerfVP(dim)
    else:
        raise ValueError(
            f"Unknown norm_type for sequences: '{norm_type}'. "
            "Choose from: layernorm, rmsnorm, batchnorm, groupnorm, derf, derf_vp"
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
        # GroupNorm with 1 group normalizes over (C, H, W) — canonical LayerNorm for CNNs
        return partial(nn.GroupNorm, 1)
    elif name == "groupnorm":
        return partial(nn.GroupNorm, 32)
    elif name == "rmsnorm":
        return RMSNorm2d
    elif name == "rmsnorm_bias":
        return partial(RMSNorm2d, bias=True)
    elif name == "derf":
        return Derf2d
    elif name == "derf_vp":
        return Derf2dVP
    elif name == "localnorm":
        return partial(LocalNorm2d, K=2)
    elif name == "localnorm_k1":
        return partial(LocalNorm2d, K=1)
    elif name == "localnorm_k4":
        return partial(LocalNorm2d, K=4)
    elif name == "localnorm_k8":
        return partial(LocalNorm2d, K=8)
    elif name == "localnorm_k16":
        return partial(LocalNorm2d, K=16)
    elif name in ("nonorm", "nonorm_ws"):
        return NoNorm
    else:
        raise ValueError(
            f"Unknown norm layer '{name}'. Choose from: batchnorm, layernorm, groupnorm, rmsnorm, rmsnorm_bias, derf, derf_vp, localnorm, localnorm_k1, localnorm_k4, localnorm_k8, localnorm_k16, nonorm, nonorm_ws"
        )
