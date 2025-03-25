from flax import nnx
import jax.numpy as jnp

from dataclasses import dataclass

#same as RMSNorm in pytorch
@dataclass
class RMSNormConfig:
    out_dim: int = 100
    eps: float = 1e-6

class RMSNormNNX(nnx.Module):
    def __init__(self, config: RMSNormConfig):
        super().__init__()
        self.eps = config.eps # eps for numerical stability
        self.g = nnx.Param(jnp.ones(config.out_dim))  # Learnable scale parameter

    def __call__(self, x):  
        rms = jnp.pow(x,2).mean() # mean of x^2
        a_hat = (jnp.sqrt(x + self.eps)/ rms) 
        return a_hat * self.g

if __name__ == '__main__':

     n = 2
     x = jnp.ones((1, n))
     rmsnorm = RMSNormNNX(config=RMSNormConfig(out_dim = n))
     a_hat = rmsnorm(x)
     print(a_hat)
     print(a_hat.shape)
