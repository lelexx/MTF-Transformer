import torch.nn as nn
from .single import Attention
from ..utils.gelu import GELU
from ..utils.layer_norm import LayerNorm

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, T = 9):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(num_heads = h, dropout = dropout, T = T)

        self.dropout = nn.Dropout(p=dropout)
    def set_bn_momentum(self, momentum):
        self.attention.set_bn_momentum(momentum)
    def forward(self, query, key, value, mask):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x = self.attention(query, key, value, mask=mask,)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        x = self.output_linear(x)
        
        return x
