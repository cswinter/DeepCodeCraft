import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np


class Policy(nn.Module):
    def __init__(self, layers, nhidden):
        super(Policy, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(47, nhidden)])
        for layer in range(layers - 1):
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
        x = self.forward_shared(obs)
        probs = F.softmax(self.policy_head(x), dim=1)
        logprobs = distributions.Categorical(probs).log_prob(actions)
        baseline = self.value_head(x)
        policy_loss = torch.sum(advantages * torch.exp(old_logprobs - logprobs))
        value_loss = torch.sum(F.mse_loss(returns, baseline.view(-1)))
        loss = policy_loss + value_loss_scale * value_loss
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist()

    def forward(self, x):
        x = self.forward_shared(x)
        return F.softmax(self.policy_head(x), dim=1), self.value_head(x)

    def logits(self, x):
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.policy_head(x)

    def forward_shared(self, x):
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return x

