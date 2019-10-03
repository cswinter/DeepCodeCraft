import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.agents.pg.categorical import CategoricalPgAgent


class CodeCraftAgent(CategoricalPgAgent):
    def __init__(self, hps):
        super(CodeCraftAgent, self).__init__()
        self.hps = hps

    def make_env_to_model_kwargs(self, env_spaces):
        print(f'make_env_to_model_kwargs({env_spaces})')
        return {}

    def ModelCls(self, **kwargs):
        print(f'ModelCls({kwargs}')
        return Policy(self.hps.depth, self.hps.width, self.hps.conv, self.hps)


class Policy(nn.Module):
    def __init__(self, fc_layers, nhidden, conv, hps):
        super(Policy, self).__init__()
        self.conv = conv
        self.fp16 = hps.fp16
        if conv:
            self.fc_drone = nn.Linear(9, nhidden // 2)
            self.conv_minerals1 = nn.Conv2d(in_channels=1, out_channels=nhidden // 2, kernel_size=(1, 4))
            self.conv_minerals2 = nn.Conv2d(in_channels=nhidden // 2, out_channels=nhidden // 2, kernel_size=1)
            self.fc_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(fc_layers - 1)])
        else:
            self.fc_layers = nn.ModuleList([nn.Linear(49, nhidden)])
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

    def backprop(self, hps, obs, actions, old_logprobs, returns, value_loss_scale, advantages):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()
        x = self.latents(obs)
        probs = F.softmax(self.policy_head(x), dim=1)

        logprobs = distributions.Categorical(probs).log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        vanilla_policy_loss = advantages * ratios
        if hps.ppo:
            clipped_policy_loss = torch.clamp(ratios, 1 - hps.cliprange, 1 + hps.cliprange) * advantages
            policy_loss = -torch.min(vanilla_policy_loss, clipped_policy_loss).mean()
        else:
            policy_loss = -vanilla_policy_loss.mean()

        approxkl = 0.5 * (old_logprobs - logprobs).pow(2).mean()
        clipfrac = ((ratios - 1.0).abs() > hps.cliprange).sum().type(torch.float32) / ratios.numel()

        baseline = self.value_head(x)
        value_loss = F.mse_loss(returns, baseline.view(-1)).mean()

        loss = policy_loss + value_loss_scale * value_loss
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist(), approxkl.data.tolist(), clipfrac.data.tolist()

    def forward(self, x, prev_action, prev_rew):
        #print(x.size())
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(x, 1)

        x = self.latents(x, T, B)
        pi, val = F.softmax(self.policy_head(x), dim=-1), self.value_head(x)
        val = val.squeeze(-1)

        #fc_out = self.conv(img.view(T * B, *img_shape))
        #pi = F.softmax(self.pi(fc_out), dim=-1)
        #v = self.value(fc_out).squeeze(-1)
        #print(lead_dim, T, B, img_shape)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, val = restore_leading_dims((pi, val), lead_dim, T, B)

        #print(f'pi size {pi.size()} val size {val.size()}')
        return pi, val

    def logits(self, x):
        x = self.latents(x)
        return self.policy_head(x)

    def latents(self, x, T, B):
        if self.fp16:
            x = x.half()
        if self.conv:
            x = x.reshape(T*B, -1)
            batch_size = x.size()[0]
            # x[0:9] is properties of drone 0 and global features
            xd = x[:, :9]
            xd = F.relu(self.fc_drone(xd))

            # x[9:49] are 10 x 4 properties concerning the closest minerals
            xm = x[:, 9:].view(batch_size, 1, -1, 4)
            xm = F.relu(self.conv_minerals1(xm))
            xm = F.max_pool2d(F.relu(self.conv_minerals2(xm)), kernel_size=(10, 1))
            xm = xm.view(batch_size, -1)

            x = torch.cat((xd, xm), dim=1)

        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return x

