import torch
import torch.nn as nn

from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

class RoPE(nn.Module):
    def __init__(self, d_model ,base = 10000):
        super().__init__()
        assert d_model % 2 ==0 
        self.d = d_model

        inverse = 1/base**(2*(torch.arange(0, d_model, 2))/d_model)
        self.register_buffer('inverse', inverse) # these are not trainable  - used for training and register_buffer saves the tensor to the model's state dictionary as well
        # self.register_buffer -  Stores inverse as a non-trainable parameter that moves with the model (GPU/CPU) and gets saved/loaded with model state.

        # self.r_theta = torch.zeros(size = (d_model, d_model)) - inefficient implementation - done in forward pass now
    def forward(self, x_m: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        x_m: is of type [batch, seq_len, d_model] - input for the embedding creation 
        m: positions so of type [seq_len]
        """
        freqs = torch.outer(m.float(), self.inverse) # does the outer product of m and self.inverse - inputs ares assumed to be 1d
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        x_even = x_m[..., ::2]  # even dims
        x_odd = x_m[..., 1::2]  # odd dims 
        
        rotated_even = x_even * cos_vals - x_odd * sin_vals  # cos m*theta, sin -m*theta
        rotated_odd = x_even * sin_vals + x_odd * cos_vals   # sin m*theta, cos m*theta
        
        # interleave the results back together - by stacking them first then flattening by -2
        # stack creates shape [batch, seq_len, d_model//2, 2]
        # flatten(-2) merges last two dims → [batch, seq_len, d_model]
        result = torch.stack([rotated_even, rotated_odd], dim=-1)
        result = result.flatten(-2)

        return result

        # Below implemenatation is mathematically sound but practically inefficient

        # for i in range(0, self.d//2):
        #     c = torch.cos(m*self.inverse[i])
        #     s = torch.sin(m*self.inverse[i])
            
        #     self.r_theta[2*i, 2*i] = c
        #     self.r_theta[2*i+1, 2*i+1] = c
        #     self.r_theta[2*i, 2*i+1] = -s
        #     self.r_theta[2*i+1, 2*i] = s
        # return self.r_theta @ x_m


if __name__ == "__main__":
    rope = RoPE(10)


