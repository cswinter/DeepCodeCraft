import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import uniform_


class GaussianAttention(nn.Module):
    def __init__(self, nhead: int, scale: float):
        super(GaussianAttention, self).__init__()
        self.nhead = nhead
        self.scale = scale

        self.mean = Parameter(torch.Tensor(nhead))
        self.logvariance = Parameter(torch.Tensor(nhead))
        self.weight = Parameter(torch.Tensor(nhead))

        self._reset_parameters()

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        dbatch, dseq1, dseq2 = distances.size()

        rescaled = distances * (1.0 / self.scale)
        variance = torch.exp(self.logvariance).view(1, self.nhead, 1, 1)
        centered = (rescaled.view(dbatch, 1, dseq1, dseq2) - self.mean.view(1, self.nhead, 1, 1)) / variance
        density = torch.exp(-centered.pow(2))
        return density

    def _reset_parameters(self):
        uniform_(self.mean, -0.1, 0.1)
        uniform_(self.logvariance, -1, 1)


if __name__ == '__main__':
    #__import__('ipdb').set_trace()
    ga = GaussianAttention(8, 1000.0)
    distances = torch.Tensor([
        [[0.0, 2500, 500, 20, 300]],
        [[0.0, 300, 0, 0, 0]],
        [[0.0, 2500, 2500, 2500, 2500]],
    ])
    result = ga(distances)
    print(result)
