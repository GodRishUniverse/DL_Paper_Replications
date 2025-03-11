import torch
from torch import nn
from typing import Union, List, Tuple, Optional, Dict

from einops import rearrange
from einops.layers.torch import Rearrange

from dataclasses import dataclass

@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    num_classes: int = 1000
    dim: int = 768
    hidden_dim: int = 3072
    batch_first_MSA: bool = True
    num_heads: int = 12
    num_layers: int = 12
    device: str = 'cuda'
    # dropout: float = 0.1 - not used in this implementation


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layer = nn.Sequential(
            nn.LayerNorm(config.dim, device = config.device),
            nn.Linear(config.dim, config.hidden_dim, device = config.device),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.dim, device= config.device),
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x
    
class MSA(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.device = config.device
        self.mha = nn.MultiheadAttention(config.dim, config.num_heads, batch_first=config.batch_first_MSA, device = config.device)
    def forward(self, x):
        x = x.to(self.device)
        x, _ = self.mha(x, x, x)
        return x

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # need to figure out how to compute D and then get the z array from x
        self.layers = nn.ModuleList([
            nn.ModuleList([
                MSA(config=config),
                MLP(config=config)
            ])
            for _ in range(config.num_layers)
        ])

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(config.dim, device = config.device)
        
    def forward(self, x):
        for attn, ffn in self.layers:
            x = x + attn(self.layer_norm(x)) 
            x = x + ffn(self.layer_norm(x))
        x = self.dropout(x)
        return x
    
class ViT(nn.Module):
    def __init__(self, config: ModelConfig):
        image_size = config.image_size
        super().__init__()
        self.device = config.device
        # image size can be be H * W
        H, W = image_size, image_size # square image
        patch_size = config.patch_size
        assert H % patch_size == 0 and W % patch_size == 0, "Height and Width must be divisible by patch size"
        # number of patches
        num_patches = (H // patch_size) * (W // patch_size)
        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_size * patch_size * config.num_channels),
            nn.Linear(patch_size * patch_size * config.num_channels, config.dim),
            nn.LayerNorm(config.dim)
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, config.dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config.dim))
        self.transformer = Transformer(config=config)
        self.mlp_head = MLP(dim, dim, num_classes, device) # to fix
    
    def forward(self, x):
        b, _, _, _ = x.shape
    
        # Create patch embeddings
        x = self.to_patch_embedding(x)
        
        # Expand class token to match batch size
        cls_tokens = self.class_token.expand(b, -1, -1)
        
        # Concatenate class token with patch embeddings
        z = torch.cat((cls_tokens, x), dim=1)

        z = z + self.pos_embedding
        z = self.transformer(z)
        z = self.mlp_head(z)

        logits = z[:, 0, :] # Get the logits for the class token which is the first token
        return z, logits
