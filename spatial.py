import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


# N: Batch size
# L_s: number of controllable drones
# L: max number of visible objects
# C: number of channels/features on each object
def relative_positions(
        origin,     # (N, L_s, 2)
        direction,  # (N, L_s, 2)
        positions,  # (N, L, 2)
):  # (N, L_s, L, 2)
    n, ls, _ = origin.size()
    _, l, _ = positions.size()

    origin = origin.view(n, ls, 1, 2)
    direction = direction.view(n, ls, 1, 2)
    positions = positions.view(n, 1, l, 2)

    positions = positions - origin

    angle = -torch.atan2(direction[:, :, :, 1], direction[:, :, :, 0])
    rotation = torch.cat(
        [
            torch.cat(
                [angle.cos().view(n, ls, 1, 1, 1), angle.sin().view(n, ls, 1, 1, 1)],
                dim=3,
            ),
            torch.cat(
                [-angle.sin().view(n, ls, 1, 1, 1), angle.cos().view(n, ls, 1, 1, 1)],
                dim=3,
            ),
        ],
        dim=4,
    )

    positions_rotated = torch.matmul(rotation, positions.view(n, ls, l, 2, 1)).view(n, ls, l, 2)

    return positions_rotated


def polar_indices(
        positions,  # (N, L_s, L, 2)
        nray,
        nring,
        inner_radius
):  # (N, L_s, L), (N, L_s, L)
    distances = torch.sqrt(positions[:, :, :, 0] ** 2 + positions[:, :, :, 1] ** 2)
    distance_indices = torch.clamp_max(distances / inner_radius, nring - 1).floor().long()
    angles = torch.atan2(positions[:, :, :, 1], positions[:, :, :, 0]) + math.pi
    angular_indices = (angles / (2 * math.pi) * nray).floor().long()
    return distance_indices, angular_indices


def spatial_scatter(
        items,      # (N, L, C)
        positions,  # (N, L_s, L, 2)
        nray,
        nring,
        inner_radius,
        out=None
):  # (N, L_s, C, nring, nray)
    n, l, c = items.size()
    ls = positions.size(1)

    distance_index, angular_index = polar_indices(positions, nray, nring, inner_radius)
    #print("distance index", distance_index.size(), distance_index)
    #print("angular index", angular_index.size(), angular_index)
    index = distance_index * nray + angular_index
    index = index.unsqueeze(-1)
    items = items.view(n, 1, l, c)
    #print("items", items.size(), items)
    #print("index", index.size(), index)
    return scatter_add(items, index, dim=2, dim_size=nray * nring, out=out) \
        .permute(0, 1, 3, 2) \
        .reshape(n, ls, c, nring, nray)


class ZeroPaddedCylindricalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ZeroPaddedCylindricalConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.padding = kernel_size // 2

    # input should be of dims (N, C, H, W)
    # applies dimension-preserving conv2d by zero-padding H dimension and circularly padding W dimension
    def forward(self, input):
        input = F.pad(input, [0, 0, self.padding, self.padding], mode='circular')
        input = F.pad(input, [self.padding, self.padding, 0, 0], mode='constant')
        return self.conv(input)

