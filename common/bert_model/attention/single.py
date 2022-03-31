import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from ..utils.gelu import GELU
from ..utils.layer_norm import LayerNorm

        
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, cfg, num_heads, dropout, T):
        super().__init__()
        self.cfg = cfg
        
        self.activation = GELU()
        self.T = T

    
    def set_bn_momentum(self, momentum):
        pass
    def forward(self, query, key, value, mask=None):
        #mask: (B, H, Q, K) H:num_heads Q:num_query K:num_key
        
        B,H, T, C = query.shape
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        if self.training and mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        return torch.matmul(p_attn, value)
