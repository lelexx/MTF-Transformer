import torch.nn as nn
import torch
import math
import numpy as np


class PositionalEmbedding(nn.Module):

    def __init__(self, cfg, d_model, max_len=512):
        super().__init__()
        self.cfg = cfg
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)
        self.max_len = max_len
        if self.cfg.NETWORK.TEMPORAL_MASK == []:
            self.t_choice = list(range(1, self.max_len + 1))[::2]
        else:
            self.t_choice = self.cfg.NETWORK.TEMPORAL_MASK
        print(self.t_choice)
        self.mask_len = max(2200, self.cfg.TRAIN.BATCH_SIZE * 2 * ((len(cfg.H36M_DATA.TRAIN_CAMERAS) + cfg.TRAIN.NUM_AUGMENT_VIEWS)))
        self.mask = torch.ones(self.mask_len, 1, 1, self.max_len)
        for i in range(self.mask_len):
            t = np.random.choice(self.t_choice, 1, p=list([1/len(self.t_choice)] * len(self.t_choice)))[0]
            pad = self.max_len // 2
            s = pad - t // 2 
            e = pad + t // 2  + 1
            self.mask[i,:,:,s:e] = 0
        self.mask = self.mask > 0.5
        
        #self.embedding = nn.Linear(d_model, d_model)

    def forward(self, x):
        B = x.shape[0]
        if self.training:
            mask = self.mask[:B]
            mask = mask.to(x.device)
        else:
            mask = None
        pad = self.pe.shape[1] // 2
        s = pad - x.shape[1] // 2
        e = pad + x.shape[1] // 2 + 1
        p_embedding = self.pe.to(x.device)[:, s:e]
        #p_embedding = self.embedding(p_embedding)
        
        return p_embedding, mask
