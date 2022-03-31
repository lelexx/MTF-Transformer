import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout, is_last = False, flag = False):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.is_last = is_last
        self.flag = flag
    def set_bn_momentum(self, momentum):
        self.norm.set_bn_momentum(momentum)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
    
        B, T, C = x.shape
        if self.flag:
            if self.is_last:
                res = x[:,T // 2:(T // 2 + 1)]
            else:
                res = x[:,T // 2:(T // 2 + 1)]
        else:
            res = x
        out = sublayer(self.norm(x))
        if out.__class__ == tuple:
            x, scores = out
            x = self.dropout(x)
            return res + x, scores
        else:
            x = out
            x = self.dropout(x)
            return res + x
        