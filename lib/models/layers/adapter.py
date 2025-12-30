import torch
from torch import nn
from lib.models.layers.mamba import MABlock

class Mamba_adapter(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.MA = nn.ModuleList([
            MABlock(
                hidden_dim=dim,
                mlp_ratio=0.0,
                d_state=16, )
            for i in range(2)])

    def forward(self, x_adap, xi_adap):

        x_down = torch.cat([x_adap, xi_adap], 1)

        for i in range(2):
            x_down_flip = x_down.flip(dims=[1])
            x_down= self.MA[i](x_down)
            x_down_flip = self.MA[i](x_down_flip)
            x_down_flip = x_down_flip.flip(dims=[1])
            x_down = x_down + x_down_flip

        return x_down
