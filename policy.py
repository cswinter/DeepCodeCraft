import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from gym_codecraft.envs.codecraft_vec_env import DEFAULT_OBS_CONFIG, GLOBAL_FEATURES, MSTRIDE, DSTRIDE


class Policy(nn.Module):
    def __init__(self,
                 fc_layers,
                 nhidden,
                 conv,
                 small_init_pi,
                 zero_init_vf,
                 fp16,
                 obs_config=DEFAULT_OBS_CONFIG):
        super(Policy, self).__init__()
        self.version = 'v2'

        self.kwargs = dict(
            fc_layers=fc_layers,
            nhidden=nhidden,
            conv=conv,
            small_init_pi=small_init_pi,
            zero_init_vf=zero_init_vf,
            fp16=fp16,
            obs_config=obs_config)

        self.allies = obs_config.allies
        self.drones = obs_config.drones
        self.minerals = obs_config.minerals

        self.width = nhidden

        self.conv = conv
        self.fp16 = fp16
        if conv:
            # TODO: use Conv1d
            self.conv_drone = nn.Conv2d(
                in_channels=1,
                out_channels=nhidden // 2,
                kernel_size=(1, GLOBAL_FEATURES + DSTRIDE))

            self.conv_minerals1 = nn.Conv2d(
                in_channels=1,
                out_channels=nhidden // 8,
                kernel_size=(1, MSTRIDE))
            self.conv_minerals2 = nn.Conv2d(
                in_channels=nhidden // 4,
                out_channels=nhidden // 8,
                kernel_size=1)

            self.conv_enemies1 = nn.Conv2d(
                in_channels=1,
                out_channels=nhidden // 8,
                kernel_size=(1, DSTRIDE))
            self.conv_enemies2 = nn.Conv2d(
                in_channels=nhidden // 4,
                out_channels=nhidden // 8,
                kernel_size=1)

            # self.fc_layers = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(fc_layers - 1)])
            self.final_convs = nn.ModuleList([nn.Conv2d(in_channels=nhidden,
                                                        out_channels=nhidden,
                                                        kernel_size=1) for _ in range(fc_layers - 1)])

            self.policy_head = nn.Conv2d(in_channels=nhidden, out_channels=8, kernel_size=1)
            if small_init_pi:
                self.policy_head.weight.data *= 0.01
                self.policy_head.bias.data.fill_(0.0)

            self.value_head = nn.Linear(nhidden, 1)
            if zero_init_vf:
                self.value_head.weight.data.fill_(0.0)
                self.value_head.bias.data.fill_(0.0)
        else:
            self.fc_layers = nn.ModuleList([nn.Linear(195, nhidden)])
            for _ in range(fc_layers - 1):
                self.fc_layers.append(nn.Linear(nhidden, nhidden))

            self.policy_head = nn.Linear(nhidden, 8)
            if small_init_pi:
                self.policy_head.weight.data *= 0.01
                self.policy_head.bias.data.fill_(0.0)

            self.value_head = nn.Linear(nhidden, 1)
            if zero_init_vf:
                self.value_head.weight.data.fill_(0.0)
                self.value_head.bias.data.fill_(0.0)

    def evaluate(self, observation, action_masks):
        if self.fp16:
            action_masks = action_masks.half()
        probs, v = self.forward(observation)
        probs = probs * action_masks + 1e-8  # Add small value to prevent crash when no action is possible
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        # TODO: correct for action masking?
        entropy = action_dist.entropy().mean(dim=1)
        return actions, action_dist.log_prob(actions), entropy, v.detach().view(-1).cpu().numpy(), probs.detach().cpu().numpy()

    def backprop(self, hps, obs, actions, old_logprobs, returns, value_loss_scale, advantages, old_values, action_masks, old_probs):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()

        batch_size = obs.size()[0]

        x = self.latents(obs)
        probs = F.softmax(self.policy_head(x), dim=1).view(batch_size, 8, self.allies).permute(0, 2, 1)
        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        probs = probs * action_masks + 1e-8

        logprobs = distributions.Categorical(probs).log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = advantages.view(-1, 1)
        vanilla_policy_loss = advantages * ratios
        clipped_policy_loss = advantages * torch.clamp(ratios, 1 - hps.cliprange, 1 + hps.cliprange)
        if hps.ppo:
            policy_loss = -torch.min(vanilla_policy_loss, clipped_policy_loss).mean()
        else:
            policy_loss = -vanilla_policy_loss.mean()

        # TODO: do over full distribution, not just selected actions?
        approxkl = 0.5 * (old_logprobs - logprobs).pow(2).mean()
        clipfrac = ((ratios - 1.0).abs() > hps.cliprange).sum().type(torch.float32) / ratios.numel()

        pooled = F.avg_pool2d(x, kernel_size=(self.allies, 1))
        values = self.value_head(pooled.view(batch_size, -1)).view(batch_size)
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
        batch_size = x.size()[0]
        x = self.latents(x)

        pooled = F.avg_pool2d(x, kernel_size=(self.allies, 1))
        values = self.value_head(pooled.view(batch_size, -1))

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=1)

        return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values

    def logits(self, x):
        x = self.latents(x)
        return self.policy_head(x)

    def latents(self, x):
        if self.fp16:
            x = x.half()
        if self.conv:
            endallies = (DSTRIDE + GLOBAL_FEATURES) * self.allies
            endmins = endallies + MSTRIDE * self.minerals * self.allies
            enddrones = endmins + DSTRIDE * self.drones * self.allies

            batch_size = x.size()[0]
            # properties global features of selected allied drones
            xd = x[:, :endallies].view(batch_size, 1, self.allies, DSTRIDE + GLOBAL_FEATURES)
            xd = F.relu(self.conv_drone(xd))

            # properties of closest minerals
            xm = x[:, endallies:endmins].view(batch_size, 1, self.minerals * self.allies, MSTRIDE)
            xm = F.relu(self.conv_minerals1(xm))
            pooled = F.avg_pool2d(xm, kernel_size=(self.minerals, 1))
            xm = xm.view(batch_size, -1, self.minerals, self.allies, 1)
            pooled = pooled.view(batch_size, -1, 1, self.allies, 1)
            pooled_expanded = torch.cat(self.minerals * [pooled], dim=2)
            xm = torch.cat([xm, pooled_expanded], dim=1)
            xm = xm.view(batch_size, -1, self.minerals * self.allies, 1)

            xm = F.relu(self.conv_minerals2(xm))
            xm_avg = F.avg_pool2d(xm, kernel_size=(self.minerals, 1))
            xm_max = F.max_pool2d(xm, kernel_size=(self.minerals, 1))

            # properties of the closest drones
            xe = x[:, endmins:enddrones].view(batch_size, 1, self.drones * self.allies, DSTRIDE)
            xe = F.relu(self.conv_enemies1(xe))
            pooled = F.avg_pool2d(xe, kernel_size=(self.drones, 1))
            xe = xe.view(batch_size, -1, self.drones, self.allies, 1)
            pooled = pooled.view(batch_size, -1, 1, self.allies, 1)
            pooled_expanded = torch.cat(self.drones * [pooled], dim=2)
            xe = torch.cat([xe, pooled_expanded], dim=1)
            xe = xe.view(batch_size, -1, self.drones * self.allies, 1)

            xe = F.relu(self.conv_minerals2(xe))
            xe_avg = F.avg_pool2d(xe, kernel_size=(self.drones, 1))
            xe_max = F.max_pool2d(xe, kernel_size=(self.drones, 1))

            x = torch.cat((xd, xm_avg, xm_max, xe_avg, xe_max), dim=1)

            for conv in self.final_convs:
                x = F.relu(conv(x))

        else:
            for fc in self.fc_layers:
                x = F.relu(fc(x))

        return x

