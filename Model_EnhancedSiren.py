import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, w0):
        super(SineLayer, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class ResidualSirenBlock(nn.Module):
    def __init__(self, hidden_dim, w0):
        super(ResidualSirenBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.sine1 = SineLayer(w0)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.sine2 = SineLayer(w0)
    
    def forward(self, x):
        # Save input for residual connection
        residual = x
        out = self.linear1(x)
        out = self.sine1(out)
        out = self.linear2(out)
        out = self.sine2(out)
        return residual + out

class EnhancedSiren(nn.Module):
    def __init__(self, w0=30, in_dim=2, hidden_dim=256, out_dim=1, num_blocks=4):
        """
        Enhanced SIREN with residual connections.
        - w0: Frequency parameter.
        - in_dim: Input dimension (e.g., 2 for (x,y) coordinates).
        - hidden_dim: Dimensionality of hidden layers.
        - out_dim: Output dimension (e.g., 1 for grayscale intensity).
        - num_blocks: Number of residual blocks.
        """
        super(EnhancedSiren, self).__init__()
        self.initial_linear = nn.Linear(in_dim, hidden_dim)
        self.initial_activation = SineLayer(w0)
        # Create a stack of residual blocks.
        self.blocks = nn.ModuleList([ResidualSirenBlock(hidden_dim, w0) for _ in range(num_blocks)])
        self.final_linear = nn.Linear(hidden_dim, out_dim)
        self.w0 = w0
        
        # Initialize weights similar to the original SIREN paper.
        with torch.no_grad():
            self.initial_linear.weight.uniform_(-1. / in_dim, 1. / in_dim)
            for block in self.blocks:
                block.linear1.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)
                block.linear2.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                               np.sqrt(6. / hidden_dim) / w0)
            self.final_linear.weight.uniform_(-np.sqrt(6. / hidden_dim) / w0,
                                              np.sqrt(6. / hidden_dim) / w0)
    
    def forward(self, x):
        x = self.initial_linear(x)
        x = self.initial_activation(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_linear(x)
        return x
