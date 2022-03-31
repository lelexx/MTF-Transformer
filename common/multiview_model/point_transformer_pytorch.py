import torch
from torch import nn, einsum
from einops import repeat

# helpers

def exists(val):
    return val is not None

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        cfg,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
    ):
        super().__init__()
        self.cfg = cfg

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(int(17 *2), pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )
    def set_bn_momentum(self, momentum):
        pass
    def forward(self, x, pos, mask = None):
        B, T, J, C, N = pos.shape
        pos = pos[:,:,:,:2,:].permute(0, 1, 4, 2, 3).contiguous() #(B, T, N, J, C)
        pos = pos.view(B * T, N, -1)
        
        n = x.shape[1]

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]


        # expand values
        v = repeat(v, 'b j d -> b i j d', i = n)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            sim.masked_fill_(mask, -1e9)

        # attention
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)

        return agg

if __name__ == '__main__':
    attn = PointTransformerLayer(
        dim = 128,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4
    )
    total_params = sum(p.numel() for p in attn.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in attn.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    feats = torch.randn(1, 16, 128)
    pos = torch.randn(1, 16, 3)
    mask = torch.ones(1, 16) > 0

    out = attn(feats, pos, mask = mask) # (1, 16, 128)
    print(out.shape)
    exit()