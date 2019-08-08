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
        probs.detach_()
        for i in range(probs.size()[0]):
            actions.append(np.random.choice(8, 1, p=probs[i].cpu().numpy())[0])
        return actions, self.entropy(probs)

    def backprop(self, obs, actions, returns):
        logits = self.logits(obs)
        # TODO: should this use probability value at rollout time before policy updates?
        p = torch.clamp_min(F.softmax(logits.data, dim=1).gather(1, actions.view(-1, 1)), 1).view(-1)
        loss = torch.sum(returns * F.cross_entropy(logits, actions) / p)
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


