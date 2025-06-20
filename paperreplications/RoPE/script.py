import torch
import torch.nn as nn

# TODO: this is VERY BASIC AND PARTIALLY INCORRECT - FIX THIS

class RoPE(nn.Module):
    def __init__(self, d_model ,base = 10000):
        super().__init__()
        assert d_model % 2 ==0 
        self.d = d_model
        self.inverse = 1/base**(2*(torch.arange(0, d_model//2, 1))/d_model)
        self.r_theta = torch.zeros(size = (d_model, d_model))
    def forward(self, x_m, m):
        for i in range(0, self.d//2):
            c = torch.cos(m*self.inverse[i])
            s = torch.sin(m*self.inverse[i])
            
            self.r_theta[2*i, 2*i] = c
            self.r_theta[2*i+1, 2*i+1] = c
            self.r_theta[2*i, 2*i+1] = -s
            self.r_theta[2*i+1, 2*i] = s
        return self.r_theta *x_m
