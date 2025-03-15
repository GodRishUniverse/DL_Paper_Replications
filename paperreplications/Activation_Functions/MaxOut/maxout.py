import torch
import torch.nn as nn
# Since I implemented this for my cGAN but found that using LeakyReLU would be a better fit than MaxOut - I have chosen to include it here
class MaxOut(nn.Module):
    def __init__(self,d_in, d_hid, d_out, k = 2,device= 'cuda'):
        """ 
        d_hid is equal to n for n units in the MLP
        k is the number of parallel transformations - k pieces of MLP
        """
        super().__init__()
        
        self.to(device=device)

        self.device = device
        self.k = k

        self._init_layer = nn.Linear(d_in, d_hid, device = self.device)

        # self._mid = nn.ModuleList([nn.Linear(d_hid, d_hid, device = self.device) for _ in range(k)])
        
        self._mid = nn.Linear(d_hid, d_hid*k, device = self.device) # numericaly equivalent to above commented out line but more efficient when k is large because of not having k different matrices for linear layers but a single large matrix
        self._output_layer = nn.Linear(d_hid, d_out, device = self.device)
        
    def forward(self, x):
        """ 
        x: (batch_size, d_in)
        """
        x = self._init_layer(x)
         # Apply all k transformations in parallel
        x   = self._mid(x).view(x.shape[0], x.shape[1], self.k)
        x, _ = torch.max(x, dim=-1)  # (batch_size, d_hid)
        return self._output_layer(x)


# Unit test for maxout:
def test_maxout_activation():
    """
    Unit test for MaxOut activation function. Passes if the output of the forward pass has the correct shape.
    """
    maxout = MaxOut(d_in=100, d_hid=100, d_out=100)
    x = torch.randn(1, 100, device='cuda')
    output = maxout(x)
    assert output.shape == (1, 100)
    print("MaxOut test passed")
test_maxout_activation()
