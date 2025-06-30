import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Dict, Union

from dataclasses import dataclass

import torch.nn.functional as F


@dataclass
class MambaConfig:
    d_model: int
    step_size: int

# block used in the mamba model
class MambaBlock(nn.Module):
    def __init__(self, args: MambaConfig):
        ...
    def forward(self, ):
        ...
    def ssm(self, ):
        ...
    def selective_scan(self, ):
        ...
    
    def ssm_step():
        ...
