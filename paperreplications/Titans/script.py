import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from typing import List, Optional, Tuple, Dict, Union

from dataclasses import dataclass

import torch.nn.functional as F


# Google Titans is a test time framework

# Abstract memory class - Can be inherited by other kinds of Memory blocks
class MemoryModuleLayer(ABC, nn.Module):
    def __init__(self, dim_in: int, depth: int =2, factor: int = 2 ):
        super().__init__()

        self.dim_in = dim_in
        assert depth >=2, "We want the L_M >= 2 as in the paper"

        self.depth = depth
        self.factor = factor

        self._mem_layers() # we assign the buffers for the weights

    @abstractmethod
    def init_memory_architecture(self,  ) -> nn.ParameterList:
        # We return the Parameter list
        ...

    def _mem_layers(self ):
        self.weights = self.init_memory_architecture()
        for i, weight in enumerate(self.weights):
            self.register_buffer(str(i), weight) # use the number of weights as buffer getters

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


# NEED to check if the expansion and contraction work or not
class LinearMLPMemory(MemoryModuleLayer):
    def init_memory_architecture(self):
        # we want the expansion to occur at the same rate and then the contraction should also happen to get the final dim as the sam as dim_in
        pairs = []
        temp_dim = self.dim_in
        for i in range(self.depth):
            if (i >= (self.depth // self.factor)):
                # we start contracting
                pairs.append((temp_dim, temp_dim//self.factor))
                temp_dim//=self.factor
            else:
                pairs.append((temp_dim, temp_dim*self.factor))
                temp_dim*=self.factor

        assert pairs[0][0]==self.dim_in and pairs[len(pairs)-1][1]==self.dim_in # check to see if they are the same or not

        weights = nn.ParameterList([nn.Parameter(torch.randn(d_in, d_out)) for (d_in, d_out) in pairs])

        # we xavier initialize our parameters now
        for weight in weights:
            nn.init.xavier_uniform_(weight)

        return weights # return the parameter list

    def forward(self, x: torch.Tensor):
        # we will be using the gelu activation
        for i,weight in enumerate(self.weights):
            if (i != 0):
                x = F.gelu(x)
            x = x @ weight
        return x

# TODO - understand how we will progress from here
class NeuralMemory(nn.Module):
    def __init__(self, d_in: int, ):
        super().__init__()

        self.w_k = nn.Parameter(torch.randn(d_in, d_in)) # projection for keys
        nn.init.xavier_uniform_(self.w_k)
        self.w_v = nn.Parameter(torch.randn(d_in, d_in)) # projection for values
        nn.init.xavier_uniform_(self.w_v)

        self.alpha_t = nn.Parameter(torch.randn(1)) # forget mechanism
        self.theta_t = nn.Parameter(torch.randn(1)) # step size
        self.naphta_t = nn.Parameter(torch.randn(1)) # to get the previous surprise

        self.previous_surpise = 0

    def store(self, x, previous_state):
        # we project to get our k_t and q_t values
        k_t = x @ self.w_k
        v_t = x @ self.w_v

    def retrieve(self, x):
        ...


    def forward(self,x):
       ...

# Basically we will also be implementing the memory an context
