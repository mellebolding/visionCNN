"""Vision Transformer (ViT) with configurable normalization.

ViT-Small/16: patch_size=16, embed_dim=384, depth=12, num_heads=6.
Pre-norm architecture (standard DeiT-style).
"""
import torch
import torch.nn as nn
from timm.layers.weight_init import trunc_normal_

from src.models.norms import get_norm_layer_seq


class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project to embed_dim.

    Input:  (N, 3, H, W)
    Output: (N, L, D)  where L = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # (N, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (N, L, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        N, L, D = x.shape
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, N, heads, L, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(N, L, D)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward block: D → hidden_dim → D with GELU."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """ViT block with pre-norm and configurable normalization."""

    def __init__(self, dim, num_heads, mlp_ratio, norm_type):
        super().__init__()
        self.norm1 = get_norm_layer_seq(norm_type, dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = get_norm_layer_seq(norm_type, dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with configurable normalization.

    Args:
        img_size: Input image size (square assumed).
        patch_size: Patch size for patch embedding.
        num_classes: Number of output classes.
        embed_dim: Token embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim multiplier relative to embed_dim.
        norm_type: Normalization type for all blocks ('layernorm', 'rmsnorm',
                   'batchnorm', 'groupnorm', 'derf').
    """

    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                 norm_type="layernorm"):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, norm_type)
            for _ in range(depth)
        ])
        self.norm = get_norm_layer_seq(norm_type, embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        N = x.shape[0]
        x = self.patch_embed(x)                         # (N, L, D)
        cls = self.cls_token.expand(N, -1, -1)          # (N, 1, D)
        x = torch.cat([cls, x], dim=1)                  # (N, L+1, D)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]                                     # CLS token
        x = self.head(x)
        return x


def vit_small(num_classes: int = 1000, norm_type: str = "layernorm", **kwargs) -> VisionTransformer:
    """ViT-Small/16: embed_dim=384, depth=12, num_heads=6."""
    return VisionTransformer(
        img_size=224, patch_size=16, num_classes=num_classes,
        embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
        norm_type=norm_type, **kwargs,
    )
