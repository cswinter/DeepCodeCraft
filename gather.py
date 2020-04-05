import torch


def topk_by(values, vdim, keys, kdim, k):
    indices = keys.topk(k=k, dim=kdim, sorted=True).indices
    indices = indices.unsqueeze(-1).expand(indices.size() + values.size()[vdim+1:])
    values_topk = values.gather(dim=vdim, index=indices)
    return values_topk


def topk_and_index_by(values, vdim, keys, kdim, k):
    indices = keys.topk(k=k, dim=kdim, sorted=True).indices
    indices = indices.unsqueeze(-1).expand(indices.size() + values.size()[vdim+1:])
    values_topk = values.gather(dim=vdim, index=indices)
    return values_topk, indices


if __name__ == '__main__':
    a = torch.tensor([[[ 1.0, 0.0, 3.0 ], [0.0, 0.0,  2.0], [ 3.0,  3.0,  3.0], [4.0,   4.0,  4.0]],
                      [[-1.0, 0.3, 0.23], [1.0, 0.0, -2.0], [-3.0, -0.3, -3.0], [0.44, -0.44, 4.04]]])
    keys = a.sum(dim=2)
    topk_by(a, 1, keys, 1, 2)

