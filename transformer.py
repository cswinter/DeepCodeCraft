from typing import Optional, List
from torch import nn
import torch
import torch.nn.functional as F

from parametric_spatial_attn import MultiheadSpatialAttn


class SpatialTransformer(nn.Module):
    def __init__(self, embed_dim: int, kvdim: int, nhead: int, dff_ratio: int):
        super(SpatialTransformer, self).__init__()

        self.multihead_attention = MultiheadSpatialAttn(
            qdim=embed_dim,
            kvdim=kvdim,
            nhead=nhead,
        )
        self.linear1 = nn.Linear(embed_dim, embed_dim * dff_ratio)
        self.linear2 = nn.Linear(embed_dim * dff_ratio, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self,
                target: torch.Tensor,
                items: torch.Tensor,
                mask: torch.ByteTensor,
                modulators: Optional[List[torch.Tensor]]):
        x = self.multihead_attention(
            query=target,
            keyval=items,
            mask=mask,
            modulators=modulators,
        )
        x = self.norm1(x + target)
        x2 = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        return x
