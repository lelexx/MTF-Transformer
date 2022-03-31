import torch.nn as nn
import torch
from .transformer import TransformerBlock
from .embedding import BERTEmbedding
from .utils.gelu import *

class Head(nn.Module):
    def __init__(self, cfg, in_channels = 10, dropout=0.25, channels=2048, num_joints = 17):
        super().__init__()
        self.cfg = cfg
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
    def __init__(self, cfg, T = 7, dropout=0.1, num_joints = 17):
        super().__init__()
        self.cfg = cfg
        self.hidden = cfg.NETWORK.T_FORMER.NUM_CHANNELS
        self.n_layers = cfg.NETWORK.T_FORMER.NUM_LAYERS
        self.attn_heads = cfg.NETWORK.T_FORMER.NUM_HEADS
        self.num_joints = num_joints
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = self.hidden * 2

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(cfg, inp_channels = cfg.NETWORK.NUM_CHANNELS, embed_size=self.hidden, max_len = T)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(cfg, self.hidden, self.attn_heads, self.feed_forward_hidden, dropout, T = T, is_last = i == self.n_layers - 1) for i in range(self.n_layers)])
        self.shrink = Head(cfg, in_channels = self.hidden, num_joints = num_joints)
    def set_bn_momentum(self, momentum):
        self.shrink.set_bn_momentum(momentum)
        for t in self.transformer_blocks:
            t.set_bn_momentum(momentum)
    def forward(self, x):
        if len(x.shape) == 5:
            B, C1, C2, T, N = x.shape
            x = x.view(B, -1, T, N)
        B,C, T, N = x.shape#(B, C, T, N)

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
        x = x.permute(0, 2, 3, 4, 1) #(B, T, J, 3, N)
        return x

class MultiViewBert(nn.Module):
    def __init__(self, cfg, dropout = 0.1):
        super().__init__()
        self.cfg = cfg
        channels = cfg.NETWORK.NUM_CHANNELS
        self.num_view = len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS
        self.transformer_block = TransformerBlock(cfg, channels, 8, channels * 2, dropout = dropout, T = self.num_view)
    def set_bn_momentum(self, momentum):
        self.transformer_block.set_bn_momentum(momentum)
    def forward(self, x, mask = None):
        #x:B, N, C
        f = self.transformer_block.forward(x, mask)
        return f
    


