from typing import Union, List, Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_in: int
    d_label: int
    hidden_size: int
    d_out: int
