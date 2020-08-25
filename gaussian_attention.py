from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.init import uniform_


class GaussianAttention(nn.Module):
    def __init__(self, nhead: int, scale: float, init_scale: float = 1.0, optional: bool = True):
        super(GaussianAttention, self).__init__()
        self.nhead = nhead
        self.scale = scale
        self.optional = optional

        self.mean = Parameter(torch.Tensor(nhead))
        self.logvariance = Parameter(torch.Tensor(nhead))
        self.weight = Parameter(torch.Tensor(nhead))

        self._reset_parameters(init_scale)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        dbatch, dseq1, dseq2 = distances.size()

        rescaled = distances * (1.0 / self.scale)
        variance = torch.exp(self.logvariance).view(1, self.nhead, 1, 1)
        centered = (rescaled.view(dbatch, 1, dseq1, dseq2) - self.mean.view(1, self.nhead, 1, 1)) / variance
        density = torch.exp(-centered.pow(2))
        if self.optional:
            weight = torch.sigmoid(self.weight).view(1, self.nhead, 1, 1)
        else:
            weight = 1.0
        return 1 + weight * (density - 1)

    def _reset_parameters(self, init_scale: float):
        uniform_(self.mean, -0.1 * init_scale, 0.1 * init_scale)
        uniform_(self.logvariance, -0.5 * init_scale, 0.5 * init_scale)
        uniform_(self.weight, -0.5 * init_scale, 0.5 * init_scale)


def plot_heatmap(
        mean: float,
        logvariance: float,
        weight: float,
        wangle: Optional[Tuple[float, float, float]] = None,
        scale: float = 1000.0,
        width: int = 1000,
        height: int = 1000):
    import matplotlib.pyplot as plt
    import numpy as np

    attn = np.zeros((width, height), dtype=np.float)
    weight = 1/(1 + np.exp(-weight))
    variance = np.exp(logvariance)
    if wangle is not None:
        meana, logvara, weighta = wangle
        weighta = 1/(1 + np.exp(-weighta))
        vara = np.exp(logvara)
    for i in range(width):
        for j in range(height):
            x = i - width // 2
            y = j - width // 2
            dist = (x ** 2 + y ** 2) ** 0.5 / scale
            attn[i][j] = 1 + weight * (np.exp(-((dist - mean) / variance) ** 2) - 1)
            if wangle is not None:
                angle = np.arctan2(y, x)
                attn[i][j] *= (1 + weighta * (np.exp(-((angle - meana) / vara) ** 2) - 1))

    plt.imshow(attn, cmap='viridis')
    plt.colorbar()
    plt.show()


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
