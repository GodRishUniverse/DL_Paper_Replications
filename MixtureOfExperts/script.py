from typing import Union, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as dist
 
from dataclasses import dataclass

@dataclass
class MoEConfig:
    num_experts: int = 4
    shape: Tuple[int, int, int] = (100, 100, 100)
    device: str = 'cuda'


# our experts are gonna be simple FFN with identical architectures  - we can make this configurable
class FFN(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(config.shape[-1], config.shape[-1], device = config.device),
            nn.ReLU(),
            nn.Linear(config.shape[-1], config.shape[-1], device = config.device)
        )
    def forward(self, x):
        return self.layer(x)


class Gating(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()

        # shape is batch, token_size, embedding_size
        batch, token_size, embedding_size = config.shape

        self.device = config.device
        self.num_experts = config.num_experts

        self.W_g = nn.Parameter(torch.zeros(embedding_size, self.num_experts).to(self.device))
        nn.init.xavier_uniform_(self.W_g)
        self.W_noise = nn.Parameter(torch.randn(embedding_size, self.num_experts).to(self.device))

        self.softmax= nn.Softmax(dim=-1) # dimension 0 will always be the batch so we do not apply softmax there
        self.softplus = nn.Softplus()
    
    def keepTopK(self, v, k):
        # Get the top-k values along the last dimension
        topk_values, _ = torch.topk(v, k, dim=-1)

        # Create a mask where only the top-k values remain
        v_mask = v >= topk_values[..., -1].unsqueeze(-1)

        # Set values outside the top-k to -inf
        v = torch.where(v_mask, v, torch.tensor(-float('inf'), device=self.device))

        return v
        
    def forward(self, x, k = 1):
        assert k <= self.num_experts

        prelim = torch.matmul(x, self.W_g)
        noise = torch.matmul(x, self.W_noise)

        # sample from standard normal
        standard_normal = dist.Normal(0, 1).sample((x.shape[0], x.shape[1], self.num_experts)).to(self.device)

        # apply softplus
        noise = self.softplus(noise)

        # comput H(x)
        h_x = prelim + standard_normal + noise
        return self.softmax(self.keepTopK(h_x, k))


class MixtureOfExperts(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()

        self.number_of_experts = config.num_experts  
        self.device = config.device 

        self.gatings = Gating(config)
        self.experts = nn.ModuleList([FFN(config) for _ in range(self.number_of_experts)])
        
    def forward(self, x, k = 1):

        assert len(x.shape) == 3 # check the shape 

        assert k <= self.number_of_experts # assert the shape as k should be less than or equal to number of experts

        gates = self.gatings(x, k=k)

        y_out = torch.zeros_like(x)
         # Apply each expert weighted by its gate value
        for i in range(self.number_of_experts):
            # Only compute if this expert was selected for at least some inputs
            if torch.any(gates[:,:,i] > 0):
                expert_output = self.experts[i](x)
                # Weight the expert output by its gate value
                # expert_output shape is batch, token_size, embedding_size and gates shape is batch, token_size, num_experts and unsqueeze makes the individual shape as batch, token_size, 1
                y_out += expert_output * gates[:,:,i].unsqueeze(-1) 
        return y_out


if __name__ == "__main__":
    config = MoEConfig(num_experts = 4, shape = (100, 100, 100), device = 'cpu')
    moe = MixtureOfExperts(config)
    x = torch.randn(100, 100, 100).to(config.device)

    output = moe(x, k=2)

    print(output.shape)
