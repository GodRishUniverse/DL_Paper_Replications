from webbrowser import get
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
        assert depth % 2 == 0, "We want to use even depths for expansion and contraction"
        assert factor >= 2, "We want the expansion factor to be larger than or equal to 2"

        self.depth = depth
        self.factor = factor

        self.init_memory_architecture() # we assign the weights here

    @abstractmethod
    def init_memory_architecture(self,  ) -> None:
        # We return the Parameter list
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def get_parameter_weights(self, )-> List[nn.Parameter]:
        ...


#TODO: NEED to check if the expansion and contraction work or not
class LinearMLPMemory(MemoryModuleLayer):
    def init_memory_architecture(self) -> None:
        # we want the expansion to occur at the same rate and then the contraction should also happen to get the final dim as the sam as dim_in
        pairs = []
        temp_dim = self.dim_in
        pivot = self.depth // 2
        for i in range(self.depth):
            if i < pivot:       # expand
                out = temp_dim * self.factor
            else:               # contract
                out = temp_dim // self.factor
                # sanity check: keep divisible; alternatively allow a Linear with bias
                if out == 0 or temp_dim % self.factor != 0:
                    raise ValueError("dim not divisible by factor; choose dims/factor accordingly.")
            pairs.append((temp_dim, out))
            temp_dim = out

        assert pairs[0][0]==self.dim_in and pairs[len(pairs)-1][1]==self.dim_in # check to see if they are the same or not

        self.weights = nn.ParameterList([nn.Parameter(torch.randn(d_in, d_out)) for (d_in, d_out) in pairs])

        # we xavier initialize our parameters now
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def get_parameter_weights(self, )-> List[nn.Parameter]:
        return list(self.weights)

    def forward(self, x: torch.Tensor):
        for i,weight in enumerate(self.weights):
            if i != 0:
                x = F.gelu(x)         # we will be using the gelu activation
            x = x @ weight
        return x

# TODO - understand how we will progress from here
# TODO: also update for chunking
class NeuralMemory(nn.Module):
    def __init__(self,
        d_in: int,
        depth: Optional[int],
        expansion_factor: Optional[int],

        lr: Optional[float] = None,
        alpha: Optional[float] = None,
        decay: Optional[float] = None,

        learn_hyper_params: Optional[bool] = True
    ):

        super().__init__()

        self.memory= LinearMLPMemory(dim_in=d_in, depth=depth if depth is not None else 2, factor= expansion_factor if expansion_factor is not None else 2) # this memort is our M_t (from paper)

        self.w_k = nn.Parameter(torch.randn(d_in, d_in)) # projection for keys
        self.w_v = nn.Parameter(torch.randn(d_in, d_in)) # projection for values
        self.w_q = nn.Parameter(torch.randn(d_in, d_in)) # projection for queries - for retrieval

        for w in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_uniform_(w)

        # TODO: change these to pass in values and also set up bounding values for them
        self.alpha_t = self._make_param(alpha, 0.0001, 1.0) # forget mechanism [0,1] <- range for alpha
        self.lr = self._make_param(lr, 0.0001, 1.0) # step size - learning rate [0,1] <- range for lr
        self.decay = self._make_param(decay, 0.0001, 1.0) # to get the previous surprise <- [0,1] range for decay factor

        self.previous_surpise = None

    def _make_param(self, init_val,  min, max ) -> nn.Parameter:
        return nn.Parameter(torch.clamp(torch.tensor(init_val) if init_val is not None else torch.randn(1), min= min, max=max) )

    # ------ Inner-loop "store" step ------
    def store(self, x, previous_state) -> torch.Tensor:
        # we project to get our k_t and q_t values
        k_t = x @ self.w_k
        v_t = x @ self.w_v

        loss = F.mse_loss(self.memory(k_t), v_t) # L2 Loss -> MSE [MAE is L1 Loss - paper uses loss = ||M_t(k_t) -v_t ||^2_2 - so the base 2 represents l2 loss]
        weights = self.memory.get_parameter_weights()

        grads = torch.autograd.grad(loss, weights, create_graph=True)

        #TODO: surprise metrics are different per parameter
        with torch.no_grad():
            #TODO: fix this - grad returns a list
            for weight in self.memory.get_parameter_weights():
                surprise = self.decay_t*(self.previous_surpise if self.previous_surpise is not None else torch.zeros(weight.shape)) + self.lr*grads[weight]
                weight=(1-self.alpha_t)*weight + surprise
                self.previous_surpise = surprise

        return torch.Tensor(1) # TODO: change this - boilerplate to remove the return error

    # we will not be updating anything here
    def retrieve(self, x)-> torch.Tensor:
        q = x @ self.w_q # project to space
        return self.memory(q)

    def forward(self,x, update_mem: bool = False) -> torch.Tensor :
       if update_mem:
           return self.store(x)
       return self.retrieve(x)

# Basically we will also be implementing the MAC, MLA, MAG architectures
