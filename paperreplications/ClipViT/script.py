from paperreplications.ViT.script import ViT, ModelConfig as vit_config
from paperreplications.Transformer.script import Transformer, ModelConfig as transformer_config

import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass

from typing import Union, List, Tuple, Optional, Dict

from tqdm.auto import tqdm

from einops import rearrange
from einops.layers.torch import Rearrange


@dataclass
class ModelConfig:
    # vit specific
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

    class_dim: int = 768
    hid_class_dim: int = 3072

    # transformer specific
    d_model: int = 512 
    dff: int = 2048
    num_heads: int = 8
    dropout: float = 0.1

    src_vocab_size: int = 100
    tgt_vocab_size: int = 100
    pad_index: int = 0

    num_encoder_layers: int = 6 # num_encoder_layers
    num_decoder_layers: int = 6 # num_decoder_layers
    max_seq_len: int =1000

class ClipViT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        vit_configuration = vit_config(
            image_size = config.image_size,
            patch_size = config.patch_size,
            num_channels = config.num_channels,
            num_classes = config.num_classes,
            dim = config.dim,
            hidden_dim = config.hidden_dim,
            batch_first_MSA = config.batch_first_MSA,
            num_heads = config.num_heads,
            num_layers = config.num_layers,
            device = config.device,
            classify=False # no classification - just feature extraction
        )
        self.img_encoder = ViT(config=vit_configuration)

        transformer_configuration = transformer_config(
            d_model=config.d_model,
            dff=config.dff,
            num_heads=config.num_heads,
            dropout=config.dropout,
            src_vocab_size=config.src_vocab_size,
            tgt_vocab_size=config.tgt_vocab_size,
            pad_index=config.pad_index, 
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            max_seq_len=config.max_seq_len,
            device=config.device    
        )

        self.text_encoder = Transformer(config=transformer_configuration)

        # to fix as this learned projection is WRONG
        self.W_i = nn.Parameter(torch.randn(config.dim, config.hid_class_dim)) # to see if this works
        self.W_t = nn.Parameter(torch.randn(config.dim, config.class_dim)) # to see if this works

        self.temp = nn.Parameter(torch.tensor(0.1))
        
        self.device = config.device
        
    
    def forward(self, I, T, n=None):
        I_f, _ = self.img_encoder(I)
        T_f = self.text_encoder(T)

        # joint multimodal embedding [n, d_e]
        I_e = F.normalize(torch.dot(I_f, self.W_i), dim=1)
        T_e = F.normalize(torch.dot(T_f, self.W_t), dim=1)
        # scaled pairwise cosine similarities [n, n]
        logits = torch.dot(I_e, T_e.T) * torch.exp(self.temp) 

        if n is not None:
            # symmetric loss function
            labels = torch.arange(n).to(self.device)
            loss_1 = F.cross_entropy(logits, labels, dim = 0)
            loss_2 = F.cross_entropy(logits, labels, dim = 1)
            loss = loss_1 + loss_2
            return logits, loss/2
        else:
            return logits, None

        
        
