"""Custom layers for vision models."""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer normalization supporting both channels_first and channels_last formats."""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(
                x, (x.shape[-1],), self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]
        else:
            raise NotImplementedError


class GRN(nn.Module):
    """Global Response Normalization (GRN) layer."""
    
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (N, H, W, C)
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


__all__ = ['LayerNorm', 'GRN']