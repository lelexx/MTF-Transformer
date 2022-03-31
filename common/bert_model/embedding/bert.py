import torch.nn as nn
import torch
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    def __init__(self,cfg, inp_channels,  embed_size, dropout=0.1, max_len = 7):
        super().__init__()
        self.cfg = cfg
        self.token = TokenEmbedding(cfg, inp_channels = inp_channels, embed_size=embed_size)
        self.position = PositionalEmbedding(cfg, d_model=embed_size, max_len = max_len)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        position, mask = self.position(sequence)
        token = self.token(sequence)
        x = token + position
        return self.dropout(x), mask
