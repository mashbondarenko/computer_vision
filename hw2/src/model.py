import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int = 8, patch_size: int = 2, in_channels: int = 1, embed_dim: int = 32):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, dim)
        out = self.proj_drop(self.proj(out))
        if return_attn:
            return out, attn
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
            return x, attn
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 8,
        patch_size: int = 2,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 32,
        depth: int = 2,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.use_cls_token = use_cls_token

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_tokens = self.patch_embed.num_patches + (1 if use_cls_token else 0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @property
    def seq_len(self) -> int:
        return self.patch_embed.num_patches + (1 if self.use_cls_token else 0)

    def forward_features(self, x: torch.Tensor, return_last_attn: bool = False):
        x = self.patch_embed(x)
        if self.use_cls_token:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        last_attn = None
        for idx, block in enumerate(self.blocks):
            if return_last_attn and idx == len(self.blocks) - 1:
                x, last_attn = block(x, return_attn=True)
            else:
                x = block(x)

        x = self.norm(x)
        pooled = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return pooled, last_attn

    def forward(self, x: torch.Tensor, return_last_attn: bool = False):
        pooled, attn = self.forward_features(x, return_last_attn=return_last_attn)
        logits = self.head(pooled)
        if return_last_attn:
            return logits, attn
        return logits
