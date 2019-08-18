import torch
import torch.nn as nn
import torch.nn.functional as F
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
        actions = []
        ps = []
        probs.detach_()
        for i in range(probs.size()[0]):
            probs_np = probs[i].cpu().numpy()
            action = np.random.choice(8, 1, p=probs_np)[0]
            actions.append(action)
            ps.append(probs_np[action])
        return actions, ps, self.entropy(probs), v.detach().view(-1).cpu().numpy()

    def backprop(self, obs, actions, probs, returns, value_loss_scale, advantages):
        x = self.forward_shared(obs)
        logits = self.policy_head(x)
        baseline = self.value_head(x)
        policy_loss = torch.sum(advantages * F.cross_entropy(logits, actions) / torch.clamp_min(probs, 0.01))
        value_loss = torch.sum(F.mse_loss(returns, baseline.view(-1)))
        (policy_loss + value_loss_scale * value_loss).backward()
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

    def entropy(self, dist):
        logs = torch.log2(dist)
        logs[logs == float('inf')] = 0
        entropy = -torch.dot(dist.view(-1), logs.view(-1)) / dist.size()[0]
        return entropy


