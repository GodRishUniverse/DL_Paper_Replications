import math
import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Dict, Union

from einops import rearrange, repeat

from dataclasses import dataclass

import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int
    projection_size: int

    conv_ker: int
    conv_bias: bool = True
    bias: bool = False  
    device: str = "cuda"
    dtype: str = None

# block used in the mamba model
class MambaBlock(nn.Module):
    def __init__(self, args: MambaConfig):

        kwargs = {"device" : args.device, "dtype" : args.dtype} # reduced code repetition here as same keyword arguments added
        self.args = args
      
        self.activation = nn.SiLU()

        self.proj_1 = nn.Linear(in_features = args.d_model, out_features = args.d_model*2, bias= args.bias, **kwargs )
        self.proj_2 = nn.Linear(in_features = args.d_model, out_features = args.d_model*2, bias = args.bias, **kwargs)

        self.conv_size = int(args.d_model * args.projection_size)
        self.conv = nn.Conv1d(in_channels = self.conv_size, out_channels= self.conv_size, kernel_size= args.conv_ker, groups=self.conv_size-1, bias=args.conv_bias,  **kwargs)
        

        ...
    def forward(self, hidden_states ):
        ...
    def ssm(self, x: torch.Tensor, y: torch.Tensor ):
        """
            x: [B,L,D]
            y: [B,L,D]
        """
       
        ...
    def selective_scan(self, x: torch.Tensor, y: torch.Tensor):
        """
            x: [B,L,D]
            y: [B,L,D]
        """
        ...
    

