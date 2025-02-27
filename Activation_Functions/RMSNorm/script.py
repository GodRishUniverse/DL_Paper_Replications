from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict


@dataclass
class RMSNormConfig:
    out_dim: int = 100
    eps: float = 1e-6


class RMSNorm(nn.Module):
    def __init__(self, config: RMSNormConfig):
        super().__init__()
        self.eps = config.eps # eps for numerical stability
        self.g = nn.Parameter(torch.ones(config.out_dim))  # Learnable scale parameter

    def forward(self, x):
        rms = torch.pow(x,2).mean(dim = -1, keepdim = True) # mean of x^2
        a_hat = (torch.sqrt(x + self.eps)/ rms) 
        return a_hat * self.g

if __name__ == "__main__":
    config = RMSNormConfig()
    model = RMSNorm(config)
    print(model)
