import torch.nn as nn
import torch
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .utils.gelu import *

class Head(nn.Module):
    def __init__(self, in_channels = 10, dropout=0.25, channels=2048, num_joints = 17):
        super().__init__()
        channels = in_channels
        
        self.shrink = nn.Conv1d(channels, num_joints * 3, 1, bias = True)

        self.num_joints = num_joints
    def set_bn_momentum(self, momentum):
        pass
    def forward(self, x):
        B, T, C = x.shape

        x = x[:,T // 2:(T // 2 + 1)]
        x = x.permute(0, 2, 1).contiguous()
        x = self.shrink(x).view(B, -1, self.num_joints, 3)
        return x

class BERT(nn.Module):
    def __init__(self, T = 9, inp_channels= 10, hidden=512, n_layers=12, attn_heads=8, dropout=0.1, num_joints = 17):
        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.num_joints = num_joints
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(inp_channels = inp_channels, embed_size=hidden, max_len = T)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, self.feed_forward_hidden, dropout, T = T) for i in range(n_layers)])
        self.shrink = Head(in_channels = hidden, num_joints = num_joints)
    def set_bn_momentum(self, momentum):
        self.shrink.set_bn_momentum(momentum)
        for t in self.transformer_blocks:
            t.set_bn_momentum(momentum)
    def forward(self, x):
        B, _, _, T, N = x.shape

        x = x.view(B, -1, T, N)#(B, C, T, N)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(B * N, T, -1)

        B, T, C = x.shape
        
        inp = x

        # embedding the indexed sequence to sequence of vectors
        inp, mask = self.embedding(inp)
        
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            inp = transformer.forward(inp, mask)
        #print('***********************************')

        x = self.shrink(inp)
        B, T, _, _ = x.shape
        x = x.view(-1, N, T, self.num_joints, 3)
        x = x.permute(0, 2, 3, 4, 1)
        return x
    


