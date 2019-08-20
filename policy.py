import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class Policy(nn.Module):
    def __init__(self, fc_layers, nhidden, conv):
        super(Policy, self).__init__()
        self.conv = conv
        if conv:
            self.fc_drone = nn.Linear(7, nhidden // 2)
            self.conv_minerals1 = nn.Conv2d(in_channels=1, out_channels=nhidden // 2, kernel_size=(1, 4))
            self.conv_minerals2 = nn.Conv2d(in_channels=nhidden // 2, out_channels=nhidden // 2, kernel_size=1)
            self.fc_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(fc_layers - 1)])
        else:
            self.fc_layers = nn.ModuleList([nn.Linear(47, nhidden)])
            for _ in range(fc_layers - 1):
                self.fc_layers.append(nn.Linear(nhidden, nhidden))

        self.policy_head = nn.Linear(nhidden, 8)
        self.value_head = nn.Linear(nhidden, 1)
        self.value_head.weight.data.fill_(0.0)
        self.value_head.bias.data.fill_(0.0)
        # TODO: init to 0?
        # self.dense_final.weight.data.fill_(0.0)

    def evaluate(self, observation):
        probs, v = self.forward(observation)
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()
        return actions, action_dist.log_prob(actions), entropy, v.detach().view(-1).cpu().numpy()

    def backprop(self, obs, actions, old_logprobs, returns, value_loss_scale, advantages):
        x = self.latents(obs)
        probs = F.softmax(self.policy_head(x), dim=1)
        logprobs = distributions.Categorical(probs).log_prob(actions)
        baseline = self.value_head(x)
        policy_loss = (advantages * torch.exp(old_logprobs - logprobs)).mean()
        value_loss = F.mse_loss(returns, baseline.view(-1)).mean()
        loss = policy_loss + value_loss_scale * value_loss
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist()

    def forward(self, x):
        x = self.latents(x)
        return F.softmax(self.policy_head(x), dim=1), self.value_head(x)

    def logits(self, x):
        x = self.latents(x)
        return self.policy_head(x)

    def latents(self, x):
        if self.conv:
            batch_size = x.size()[0]
            # x[0:7] is properties of drone 0
            xd = x[:, :7]
            xd = F.relu(self.fc_drone(xd))

            # x[7:47] are 10 x 4 properties concerning the closest minerals
            xm = x[:, 7:47].view(batch_size, 1, -1, 4)
            xm = F.relu(self.conv_minerals1(xm))
            xm = F.max_pool2d(F.relu(self.conv_minerals2(xm)), kernel_size=(10, 1))
            xm = xm.view(batch_size, -1)

            x = torch.cat((xd, xm), dim=1)

        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return x

