"""ConvNeXtV2 implementation with configurable normalization."""
import torch
import torch.nn as nn
from src.utils.layers import LayerNorm, GRN
from src.models.norms import NoNorm, WSConv2d, RMSNorm2d, RMSNorm
from timm.layers.drop import DropPath
from timm.layers.weight_init import trunc_normal_


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        norm_type (str): 'layernorm', 'batchnorm', 'groupnorm', 'nonorm', or 'nonorm_ws'.
    """
    def __init__(self, dim, drop_path=0., norm_type="layernorm"):
        super().__init__()
        self.norm_type = norm_type
        use_ws = norm_type == "nonorm_ws"
        conv_cls = WSConv2d if use_ws else nn.Conv2d
        self.dwconv = conv_cls(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv

        if norm_type == "layernorm":
            self.norm = LayerNorm(dim, eps=1e-6)
        elif norm_type == "batchnorm":
            self.norm = nn.BatchNorm2d(dim)
        elif norm_type == "groupnorm":
            self.norm = nn.GroupNorm(32, dim)
        elif norm_type == "rmsnorm":
            # Channels-last RMSNorm over C only — matches LayerNorm axes
            self.norm = RMSNorm(dim, bias=False)
        elif norm_type == "rmsnorm_bias":
            # RMSNorm + bias — ablation to isolate mean centering vs bias effect
            self.norm = RMSNorm(dim, bias=True)
        elif norm_type in ("nonorm", "nonorm_ws"):
            self.norm = NoNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        if self.norm_type in ("layernorm", "rmsnorm", "rmsnorm_bias"):
            # These norms operate on channels-last: permute first
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
        else:
            # BN/GN/NoNorm operate on channels-first: norm then permute
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ConvNeXtV2 supporting both CIFAR (3-stage) and ImageNet (4-stage) configs.

    Args:
        in_chans: Input channels.
        num_classes: Number of output classes.
        depths: Number of blocks per stage.
        dims: Channel dimensions per stage.
        drop_path_rate: Max stochastic depth rate.
        norm_type: 'layernorm', 'batchnorm', 'groupnorm', 'nonorm', or 'nonorm_ws'.
        stem_type: 'cifar' (3x3, stride 2) or 'imagenet' (4x4, stride 4).
    """
    def __init__(self, in_chans=3, num_classes=10, depths=[2, 2, 2],
                 dims=[80, 160, 320], drop_path_rate=0.1,
                 norm_type="layernorm", stem_type="cifar"):
        super().__init__()
        self.depths = depths
        self.norm_type = norm_type
        self._use_ws = norm_type == "nonorm_ws"
        self.downsample_layers = nn.ModuleList()

        conv_cls = WSConv2d if self._use_ws else nn.Conv2d

        # Stem layer
        if stem_type == "imagenet":
            stem = nn.Sequential(
                conv_cls(in_chans, dims[0], kernel_size=4, stride=4),
                self._make_downsample_norm(dims[0]),
            )
        else:
            stem = nn.Sequential(
                conv_cls(in_chans, dims[0], kernel_size=3, stride=2, padding=1),
                self._make_downsample_norm(dims[0]),
            )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages
        for i in range(len(depths) - 1):
            downsample_layer = nn.Sequential(
                self._make_downsample_norm(dims[i]),
                conv_cls(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stage blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], norm_type=norm_type)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Head
        self.norm = self._make_downsample_norm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

        # Fixup-style zero-init for NoNorm: make residual branches start as identity
        if norm_type in ("nonorm", "nonorm_ws"):
            for m in self.modules():
                if isinstance(m, Block):
                    nn.init.zeros_(m.pwconv2.weight)
                    if m.pwconv2.bias is not None:
                        nn.init.zeros_(m.pwconv2.bias)

    def _make_downsample_norm(self, dim):
        """Create norm layer for stem/downsampling (always channels-first)."""
        if self.norm_type == "layernorm":
            return LayerNorm(dim, eps=1e-6, data_format="channels_first")
        elif self.norm_type == "batchnorm":
            return nn.BatchNorm2d(dim)
        elif self.norm_type == "groupnorm":
            return nn.GroupNorm(32, dim)
        elif self.norm_type == "rmsnorm":
            return RMSNorm(dim, data_format="channels_first", bias=False)
        elif self.norm_type == "rmsnorm_bias":
            return RMSNorm(dim, data_format="channels_first", bias=True)
        elif self.norm_type in ("nonorm", "nonorm_ws"):
            return NoNorm(dim)
        else:
            raise ValueError(f"Unknown norm_type: {self.norm_type}")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, WSConv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.norm(x)
        x = x.mean([-2, -1])  # Global average pooling
        x = self.head(x)
        return x
