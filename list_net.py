import torch
import torch.nn as nn
import torch.nn.functional as F


class ListNet(nn.Module):
    def __init__(self, in_features, width, items, groups, resblocks=1):
        super(ListNet, self).__init__()

        self.in_features = in_features
        self.width = width
        self.items = items
        self.groups = groups

        self.layer0 = nn.Conv1d(in_channels=1, out_channels=width, kernel_size=in_features)
        self.net = nn.Sequential(
            *[ResBlock(width) for _ in range(resblocks)]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size * self.items * self.groups, 1, self.in_features)
        x = F.relu(self.layer0(x))
        x = self.net(x)
        x = x.view(batch_size, self.items, self.groups, self.width)
        x = x.permute(0, 2, 3, 1).reshape(batch_size * self.groups, self.width, self.items)
        x_max = F.max_pool1d(x, kernel_size=self.items)
        x_avg = F.avg_pool1d(x, kernel_size=self.items)
        x = torch.cat([x_max, x_avg], dim=1)
        return x.reshape(batch_size, self.groups, 2 * self.width, 1).permute(0, 2, 1, 3)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.ReLU(),
            # nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x)
        #return x +self.convs(x)

