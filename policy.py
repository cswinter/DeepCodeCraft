import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class Policy(nn.Module):
    def __init__(self, fc_layers, nhidden, conv, small_init_pi, zero_init_vf, fp16):
        super(Policy, self).__init__()

        self.kwargs = dict(
            fc_layers=fc_layers,
            nhidden=nhidden,
            conv=conv,
            small_init_pi=small_init_pi,
            zero_init_vf=zero_init_vf,
            fp16=fp16)

        self.conv = conv
        self.fp16 = fp16
        if conv:
            self.fc_drone = nn.Linear(14, nhidden // 2)

            self.conv_minerals1 = nn.Conv2d(in_channels=1, out_channels=nhidden // 4, kernel_size=(1, 4))
            self.conv_minerals2 = nn.Conv2d(in_channels=nhidden // 4, out_channels=nhidden // 4, kernel_size=1)

            self.conv_enemies1 = nn.Conv2d(in_channels=1, out_channels=nhidden // 4, kernel_size=(1, 13))
            self.conv_enemies2 = nn.Conv2d(in_channels=nhidden // 4, out_channels=nhidden // 4, kernel_size=1)

            self.fc_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(fc_layers - 1)])
        else:
            self.fc_layers = nn.ModuleList([nn.Linear(184, nhidden)])
            for _ in range(fc_layers - 1):
                self.fc_layers.append(nn.Linear(nhidden, nhidden))

        self.policy_head = nn.Linear(nhidden, 8)
        if small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        self.value_head = nn.Linear(nhidden, 1)
        if zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.policy_head.bias.data.fill_(0.0)

    def evaluate(self, observation):
        probs, v = self.forward(observation)
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()
        return actions, action_dist.log_prob(actions), entropy, v.detach().view(-1).cpu().numpy()

    def backprop(self, hps, obs, actions, old_logprobs, returns, value_loss_scale, advantages, old_values):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()
        x = self.latents(obs)
        probs = F.softmax(self.policy_head(x), dim=1)

        logprobs = distributions.Categorical(probs).log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        vanilla_policy_loss = advantages * ratios
        clipped_policy_loss = advantages * torch.clamp(ratios, 1 - hps.cliprange, 1 + hps.cliprange)
        if hps.ppo:
            policy_loss = -torch.min(vanilla_policy_loss, clipped_policy_loss).mean()
        else:
            policy_loss = -vanilla_policy_loss.mean()

        approxkl = 0.5 * (old_logprobs - logprobs).pow(2).mean()
        clipfrac = ((ratios - 1.0).abs() > hps.cliprange).sum().type(torch.float32) / ratios.numel()

        values = self.value_head(x).view(-1)
        clipped_values = old_values + torch.clamp(values - old_values, -hps.cliprange, hps.cliprange)
        vanilla_value_loss = (values - returns) ** 2
        clipped_value_loss = (clipped_values - returns) ** 2
        if hps.clip_vf:
            value_loss = torch.max(vanilla_value_loss, clipped_value_loss).mean()
        else:
            value_loss = vanilla_value_loss.mean()

        loss = policy_loss + value_loss_scale * value_loss
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist(), approxkl.data.tolist(), clipfrac.data.tolist()

    def forward(self, x):
        x = self.latents(x)
        return F.softmax(self.policy_head(x), dim=1), self.value_head(x)

    def logits(self, x):
        x = self.latents(x)
        return self.policy_head(x)

    def latents(self, x):
        if self.fp16:
            x = x.half()
        if self.conv:
            batch_size = x.size()[0]
            # x[0:14] is properties of drone 0 and global features
            xd = x[:, :14]
            xd = F.relu(self.fc_drone(xd))

            # x[8:48] are 10 x 4 properties concerning the closest minerals
            xm = x[:, 14:54].view(batch_size, 1, -1, 4)
            xm = F.relu(self.conv_minerals1(xm))
            xm = F.max_pool2d(F.relu(self.conv_minerals2(xm)), kernel_size=(10, 1))
            xm = xm.view(batch_size, -1)

            # x[48:118] are 10 x 13 properties of the closest enemies
            xe = x[:, 54:184].view(batch_size, 1, -1, 13)
            xe = F.relu(self.conv_enemies1(xe))
            xe = F.max_pool2d(F.relu(self.conv_enemies2(xe)), kernel_size=(10, 1))
            xe = xe.view(batch_size, -1)

            x = torch.cat((xd, xm, xe), dim=1)

        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return x

