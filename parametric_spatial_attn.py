from typing import Optional, List
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, constant_


class MultiheadSpatialAttn(nn.Module):
    def __init__(self, qdim: int, kvdim: int, nhead: int):
        super(MultiheadSpatialAttn, self).__init__()
        assert qdim % nhead == 0
        assert kvdim % nhead == 0

        self.wq = Parameter(torch.Tensor(qdim, qdim))
        self.wk = Parameter(torch.Tensor(kvdim, qdim))
        self.wv = Parameter(torch.Tensor(kvdim, qdim))
        self.bq = Parameter(torch.empty(qdim))
        self.bk = Parameter(torch.empty(qdim))
        self.bv = Parameter(torch.empty(qdim))
        self.out_proj = nn.Linear(qdim, qdim)

        self.qdim = qdim
        self.kvdim = kvdim
        self.nhead = nhead

        self._reset_parameters()

    def forward(self,
                query: torch.Tensor,
                keyval: torch.Tensor,
                mask: torch.Tensor,
                modulators: Optional[List[torch.Tensor]]) -> torch.Tensor:

        dbatch, dseq_q, dfeat_q = query.size()
        dbatch_kv, dseq_kv, dfeat_kv = keyval.size()

        assert dbatch == dbatch_kv
        assert dfeat_q == self.qdim
        assert dfeat_kv == self.kvdim
        assert list(mask.size()) == [dbatch, dseq_q, dseq_kv]

        queries = (query @ self.wq + self.bq)\
            .reshape(dbatch, dseq_q, self.nhead, self.qdim // self.nhead)\
            .transpose(1, 2)\
            .reshape(dbatch * self.nhead, dseq_q, self.qdim // self.nhead)
        keys = (keyval @ self.wk + self.bk)\
            .reshape(dbatch, dseq_kv, self.nhead, self.qdim // self.nhead)\
            .transpose(1, 2)\
            .reshape(dbatch * self.nhead, dseq_kv, self.qdim // self.nhead)
        values = (keyval @ self.wv + self.bv)\
            .reshape(dbatch, dseq_kv, self.nhead, self.qdim // self.nhead)\
            .transpose(1, 2)\
            .reshape(dbatch * self.nhead, dseq_kv, self.qdim // self.nhead)

        scale = (self.qdim / self.nhead) ** -0.5
        attention_weights = queries @ keys.transpose(1, 2) * scale

        attention_weights = attention_weights.view(dbatch, self.nhead, dseq_q, dseq_kv)
        if modulators:
            for modulator in modulators:
                attention_weights *= modulator

        attention_weights = attention_weights\
            .masked_fill(mask.view(dbatch, 1, dseq_q, dseq_kv), float('-inf'))\
            .view(dbatch * self.nhead, dseq_q, dseq_kv)

        attn = torch.softmax(attention_weights, dim=2)

        # dbatch * self.nhead x dseq x self.qdim // self.nhead
        x = (attn @ values)\
            .view(dbatch, self.nhead, dseq_q, self.qdim // self.nhead)\
            .transpose(1, 2)\
            .reshape(dbatch, dseq_q, self.qdim)
        return self.out_proj(x)

    def _reset_parameters(self):
        xavier_uniform_(self.wq)
        xavier_uniform_(self.wk)
        xavier_uniform_(self.wv)
        constant_(self.bq, 0.)
        constant_(self.bk, 0.)
        constant_(self.bv, 0.)
        constant_(self.out_proj.bias, 0.)


if __name__ == '__main__':
    mha = MultiheadSpatialAttn(8, 4, 2)
    query = torch.normal(mean=0, std=1, size=(7, 1, 8))
    kv = torch.normal(mean=0, std=1, size=(7, 3, 4))
    mask = torch.ByteTensor([
        [False, True, False],
        [False, True, False],
        [False, False, False],
        [False, True, False],
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ])
    output = mha.forward(query, kv, mask)
    print(output)

