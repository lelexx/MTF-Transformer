import torch.nn as nn

from .attention import MultiHeadedAttention
from .utils import SublayerConnection, PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout, t,is_last = False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, t = t)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout, is_last = is_last, flag = True)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout, flag = False)
        self.dropout = nn.Dropout(p=dropout)
        self.is_last = is_last
    def set_bn_momentum(self, momentum):
        self.input_sublayer.set_bn_momentum(momentum)
        self.output_sublayer.set_bn_momentum(momentum)
        self.attention.set_bn_momentum(momentum)
    def forward(self, x, mask, other_score, pose):
        B, T, C = x.shape
        
        if self.is_last:
            x, scores = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, other_score = other_score, pose = pose))
        else:
            x, scores = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask, other_score = other_score, pose = pose))
        
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x), scores
