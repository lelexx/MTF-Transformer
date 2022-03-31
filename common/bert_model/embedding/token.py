import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, cfg, inp_channels = 10, embed_size=512):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Linear(inp_channels, embed_size)
    def forward(self, x):
        x = self.embedding(x)
        
        return x
    
