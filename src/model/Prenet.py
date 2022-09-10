from torch import nn
from torch.nn import functional as F

from src.model.layers import LinearReluInitNorm


class Prenet(nn.Module):
    r"""
    MLP prenet module
    """

    def __init__(self, in_dim, n_layers, prenet_dim, prenet_dropout):
        super().__init__()
        in_sizes = [in_dim] + [prenet_dim for _ in range(n_layers)]
        self.prenet_dropout = prenet_dropout
        self.layers = nn.ModuleList(
            [
                LinearReluInitNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes[:-1], in_sizes[1:])
            ]
        )

    def forward(self, x, dropout_flag):
        for linear in self.layers:
            x = F.dropout(linear(x), p=self.prenet_dropout, training=dropout_flag)
        return x
