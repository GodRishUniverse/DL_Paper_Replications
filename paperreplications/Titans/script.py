import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from typing import List, Optional, Tuple, Dict, Union

from dataclasses import dataclass

from einops import rearrange

import torch.nn.functional as F

@dataclass
class NeuralMemoryConfig:
    # TODO: implement
    ...



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
        assert hasattr(self, 'weights'), "WE NEED TO HAVE THE WEIGHTS CREATED" # check if the weights attribute was created or not

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

class NeuralMemory(nn.Module):

    def __init__(self,
        d_in: int,
        mem_type :str = "linear",
        depth: Optional[int] =2,
        expansion_factor: Optional[int] = 2,

        lr: Optional[float] = None,
        alpha: Optional[float] = None,
        decay: Optional[float] = None,

        seq_len: Optional[int] = None,
        chunk_size: Optional[int] = None,
        parallelize: Optional[bool] = False, # we want to preserve the recurrent nature of the store mechanism

        learn_hyper_params: Optional[bool] = False,

    ):

        super().__init__()

        self.learnhparams = learn_hyper_params
        self.eps = 1e-6

        self.parallelize= parallelize

        # we want both self.seq_len and self.chunk_size to be defined
        assert self._xnor(seq_len is None, chunk_size is None) is True, "We want both to be defined or neither"

        if seq_len is not None and chunk_size is not None:
            assert seq_len % chunk_size ==0, "We want the seq length to be divisible by chunk size"
            # can be fixed if we use remainders and padding [we will technically be using broadcasting in that case to fix it]
            self.seq_len = seq_len
            self.chunk_size = chunk_size
            self.num_chunks = seq_len // chunk_size


        if mem_type== 'linear':
            self.memory= LinearMLPMemory(dim_in=d_in, depth=depth if depth is not None else 2, factor= expansion_factor if expansion_factor is not None else 2) # this memort is our M_t (from paper)
        elif mem_type == "...": #TODO: to implement
            ... # to implement other kinds of memory types
            raise Exception("To implement other memory modules")
        else:
            self.memory= LinearMLPMemory(dim_in=d_in, depth=depth if depth is not None else 2, factor= expansion_factor if expansion_factor is not None else 2) # this memort is our M_t (from paper)


        self.w_k = nn.Parameter(torch.randn(d_in, d_in)) # projection for keys
        self.w_v = nn.Parameter(torch.randn(d_in, d_in)) # projection for values
        self.w_q = nn.Parameter(torch.randn(d_in, d_in)) # projection for queries - for retrieval

        for w in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_uniform_(w)

        self.alpha_t = self._make_param(alpha) # forget mechanism [0,1] <- range for alpha
        self.lr = self._make_param(lr) # step size - learning rate [0,1] <- range for lr
        self.decay = self._make_param(decay) # to get the previous surprise <- [0,1] range for decay factor

        self.surprise = [torch.zeros_like(param) for param in self.memory.get_parameter_weights()]


    def _xnor(self, a: bool, b: bool) -> bool:
        return not(a ^ b)

    def _make_param(self, init_val ) -> nn.Parameter | torch.Tensor:
        # FIXED below
        # ~~PROBLEM here - we do not want to use clamping rather should use the unconstrained space like sigmoid or softplus space as that is where the optimizer will live~~
        # ~~The constrained space is where the problem that we have actually lives- that is clamping the values in the range -> [0,1]~~
        if init_val is not None:
            clamped_val = max(min(init_val, 1.0 - self.eps), self.eps)
            unconstrained = torch.log(torch.tensor(clamped_val / (1 - clamped_val))) # inverse of sigmoid
        else:
            unconstrained = torch.rand(1)

        if self.learnhparams:
            param = nn.Parameter(unconstrained)
            return param
        return unconstrained

    @property
    def _alpha(self):
        return torch.sigmoid(self.alpha_t)

    @property
    def _decay(self):
        return torch.sigmoid(self.decay)

    @property
    def _lr(self):
        return torch.sigmoid(self.lr)

    # Inner-loop
    def store(self, x):
        # we project to get our k_t and q_t values
        k_t = x @ self.w_k
        v_t = x @ self.w_v

        loss = F.mse_loss(self.memory(k_t), v_t) # L2 Loss -> MSE [MAE is L1 Loss - paper uses loss = ||M_t(k_t) -v_t ||^2_2 - so the base 2 represents l2 loss]
        weights = self.memory.get_parameter_weights()

        # ensures optimizer has smoother space to work with - but this converts back to actual values

        grads = torch.autograd.grad(loss, weights, create_graph=True) # returns tuple of tensor and other values
        # we do not use torch.no_grad here as then that wouldn't allow the gradients of the Params to be passed through and optimized
        for i, (w,g) in enumerate(zip(weights, grads)):
            self.surprise[i] = self._decay * self.surprise[i] - self._lr * g # S_t = eta*S_{t-1} -theta_t*gradient_of_loss
            w.data = (1 -self._alpha) * w.data + self.surprise[i] # M_t = (1-alpha)*M_{t-1} + S_t
            # NOT SURE ABOUT THIS need to check if this is inplace or not - cause we do not want inplace as that destroys gradients

    # we will not be updating anything here
    def retrieve(self, x)-> torch.Tensor:
        q = x @ self.w_q # project to space for retrieval
        return self.memory(q)

    def forward(self,x, update_mem: bool = False) -> torch.Tensor :
       if update_mem:
           if self.chunk_size is not None and self.seq_len is not None and self.chunk_size < x.shape[1]: # x shape is [batch, seq_len, data]
                assert x.shape[1] == self.seq_len, "Passed sequence length and actual sequence length need to match"
                if not self.parallelize: # DEFAULT BEHAVIOUR
                    # SEQUENTIAL
                    for chunk in torch.split(x, self.chunk_size, dim=1):
                        self.store(chunk)
                else:
                    # DO WE WANT TO PARALLELIZE this? BREAKS the recurrent nature as chunk2 has no info about chunk1
                    # PARALLEL- IMPLEMENTATION EXISTS BUT NOT RECOMMENDED
                    # seq_len should be divisible by chunk_size
                    x_parallel = rearrange(x, "b (n c) d -> (b n) c d", n= self.num_chunks, c= self.chunk_size)
                    self.store(x_parallel)
           else:
                self.store(x)
       return self.retrieve(x)

# Basically we will also be implementing the MAC (Memory as context), MAL (Memory as Layer), MAG (Memory as Gate) architectures
