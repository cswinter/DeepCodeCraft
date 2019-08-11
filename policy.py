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
        # TODO: init to 0
        self.dense_final = nn.Linear(nhidden, 8)

    def evaluate(self, observation):
        probs = self.forward(observation)
        actions = []
        ps = []
        probs.detach_()
        for i in range(probs.size()[0]):
            probs_np = probs[i].cpu().numpy()
            action = np.random.choice(8, 1, p=probs_np)[0]
            actions.append(action)
            ps.append(probs_np[action])
        return actions, ps, self.entropy(probs)

    def backprop(self, obs, actions, probs, returns):
        logits = self.logits(obs)
        loss = torch.sum(returns * F.cross_entropy(logits, actions) / torch.clamp_min(probs, 0.01))
        loss.backward()
        return loss.data.tolist()

    def forward(self, x):
        return F.softmax(self.logits(x), dim=1)

    def logits(self, x):
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.dense_final(x)

    def entropy(self, dist):
        logs = torch.log2(dist)
        logs[logs == float('inf')] = 0
        entropy = -torch.dot(dist.view(-1), logs.view(-1)) / dist.size()[0]
        return entropy


