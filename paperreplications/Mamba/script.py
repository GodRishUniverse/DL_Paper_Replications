import torch
import torch.nn as nn

from typing import List, Optional, Tuple, Dict, Union

from dataclasses import dataclass

import torch.nn.functional as F

# block used in the mamba model
class MambaBlock(nn.Module):
    def __init__(self, args):
        ...
    def forward(self, ):
        ...
    def ssm(self, ):
        ...
    def selective_scan(self, ):
        ...
