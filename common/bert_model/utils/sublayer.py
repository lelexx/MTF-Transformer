import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, cfg, size, dropout):
        super(SublayerConnection, self).__init__()
        self.cfg = cfg
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def set_bn_momentum(self, momentum):
        self.norm.set_bn_momentum(momentum)
    def forward(self, x, mask, sublayer):
        "Apply residual connection to any sublayer with the same size."
    
        B, T, C = x.shape
        res = x
        out = sublayer(self.norm(x), mask)
        x = out
        x = self.dropout(x)
        if x.shape[1] == 1 and T > 1:
            res = res[:,T//2:T//2+1]
        return res + x
        