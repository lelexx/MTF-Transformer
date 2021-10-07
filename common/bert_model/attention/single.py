import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
from ..utils.gelu import GELU
from ..utils.layer_norm import LayerNorm
use_score_layer = False

class ScoresLayer(nn.Module):
    def __init__(self, num_heads, t, dropout):
        super().__init__()
        
        channels = num_heads * t * 3
        self.relu = GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = 1
        self.expand_conv = nn.Linear(num_heads * t, channels)
        self.expand_bn = LayerNorm(channels)
        conv_layers = []
        bn_layers = []
        for i in range(self.num_layers):
            conv_layers.append(nn.Linear(channels, channels))
            bn_layers.append(LayerNorm(channels))
            conv_layers.append(nn.Linear(channels, channels))
            bn_layers.append(LayerNorm(channels))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.shrink = nn.Linear(channels, num_heads * t)
    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.bn_layers:
            bn.momentum = momentum
    def forward(self, scores):
        #scores (B, n_h, n_q, n_k)
        B , n_h, n_q, n_k = scores.shape
        scores = scores.permute(0, 2, 1, 3).contiguous()#(B, n_q, n_h, n_k)

        scores = scores.view(B, n_q, n_h * n_k)
        scores = self.dropout(self.relu(self.expand_bn(self.expand_conv(scores))))
        for i in range(self.num_layers):
            res = scores
            scores = self.dropout(self.relu(self.bn_layers[i * 2](self.conv_layers[i * 2](scores))))
            scores = self.dropout(self.relu(self.bn_layers[i * 2 + 1](self.conv_layers[i * 2 + 1](scores))))
            scores = scores + res
        scores = self.shrink(scores)#(B, n_q, n_h * n_k)
        
        scores = scores.view(B, n_q, n_h, n_k)
        scores = scores.permute(0, 2, 1, 3).contiguous()
        return scores
        
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self,num_heads, dropout, T):
        super().__init__()
        #self.g_1 = ScoresLayer(num_heads = num_heads, t = T, dropout = dropout)
        self.activation = GELU()
        self.T = T

    
    def set_bn_momentum(self, momentum):
        pass
        #self.g_1.set_bn_momentum(momentum)
    def forward(self, query, key, value, mask=None):
        B,_, T, C = query.shape
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        
        
        if self.training and mask is not None:
            #print(mask)
            #t = torch.sum(mask < 0.5)
#             if t > 1:
#                 scores = self.g_1(scores)

            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)


        return torch.matmul(p_attn, value)
