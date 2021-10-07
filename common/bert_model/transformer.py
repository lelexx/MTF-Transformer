import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, T):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, T = T)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    def set_bn_momentum(self, momentum):
        self.input_sublayer.set_bn_momentum(momentum)
        self.output_sublayer.set_bn_momentum(momentum)
        self.attention.set_bn_momentum(momentum)
    def forward(self, x, mask):
        B, T, C = x.shape
        
        x  = self.input_sublayer(x, mask, lambda _x, _mask: self.attention.forward(_x, _x, _x, _mask))
        
        x = self.output_sublayer(x, mask, lambda _x, _mask: self.feed_forward.forward(_x))
        return self.dropout(x)
