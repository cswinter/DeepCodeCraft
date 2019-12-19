import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from gym_codecraft.envs.codecraft_vec_env import DEFAULT_OBS_CONFIG, GLOBAL_FEATURES, MSTRIDE, DSTRIDE
from list_net import ListNet


class Policy(nn.Module):
    def __init__(self,
                 fc_layers,
                 nhidden,
                 small_init_pi,
                 zero_init_vf,
                 fp16,
                 mpooling,
                 dpooling,
                 norm,
                 obs_config=DEFAULT_OBS_CONFIG,
                 use_privileged=False,
                 resblocks=1):
        super(Policy, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'

        self.version = 'v3'

        self.kwargs = dict(
            fc_layers=fc_layers,
            nhidden=nhidden,
            small_init_pi=small_init_pi,
            zero_init_vf=zero_init_vf,
            fp16=fp16,
            use_privileged=use_privileged,
            norm=norm,
            obs_config=obs_config,
            mpooling=mpooling,
            dpooling=dpooling,
            resblocks=resblocks,
        )

        self.obs_config = obs_config
        self.allies = obs_config.allies
        self.drones = obs_config.drones
        self.minerals = obs_config.minerals
        if hasattr(obs_config, 'global_drones'):
            self.global_drones = obs_config.global_drones
        else:
            self.global_drones = 0

        self.width = nhidden

        self.fp16 = fp16
        self.use_privileged = use_privileged

        self.self_net = self.drone_net = ListNet(
            in_features=DSTRIDE + GLOBAL_FEATURES,
            width=nhidden // 2,
            items=1,
            groups=self.allies,
            pooling=dpooling,
            norm=norm,
            resblocks=resblocks,
        )

        if self.minerals > 0:
            self.mineral_net = ListNet(
                in_features=MSTRIDE,
                width=nhidden // 4 if self.drones > 0 else nhidden // 2,
                items=self.minerals,
                groups=self.allies,
                pooling=mpooling,
                norm=norm,
                resblocks=resblocks,
            )

        if self.drones > 0:
            self.drone_net = ListNet(
                in_features=DSTRIDE,
                width=nhidden // 4 if self.minerals > 0 else nhidden // 2,
                items=self.drones,
                groups=self.allies,
                pooling=dpooling,
                norm=norm,
                resblocks=resblocks,
            )

        if use_privileged:
            self.privileged_net = ListNet(
                in_features=DSTRIDE,
                width=nhidden,
                items=self.global_drones,
                groups=1,
                pooling='both',
                norm=norm,
                resblocks=resblocks,
            )

        layers = []
        norm_layers = []
        for i in range(fc_layers - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nhidden,
                    out_channels=nhidden,
                    kernel_size=1
                )
            )
            layers.append(nn.ReLU())
            if norm == 'none' or i == fc_layers - 2:
                norm_layers.append(nn.Sequential())
            elif norm == 'batchnorm':
                layers.append(nn.BatchNorm2d(nhidden))
            elif norm == 'layernorm':
                layers.append(nn.Sequential(nn.LayerNorm([nhidden, 1, 1])))
            else:
                raise Exception(f'Unexpected normalization layer {norm}')
        self.fc_layers = nn.ModuleList(layers)
        self.fc_norm_layers = nn.ModuleList(norm_layers)

        self.policy_head = nn.Conv2d(in_channels=nhidden, out_channels=8, kernel_size=1)
        if small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        self.value_head = nn.Linear(2 * nhidden if use_privileged else nhidden, 1)
        if zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

    def evaluate(self, observation, action_masks, privileged_obs):
        if self.fp16:
            action_masks = action_masks.half()
        probs, v = self.forward(observation, privileged_obs)
        probs = probs * action_masks + 1e-8  # Add small value to prevent crash when no action is possible
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()[action_masks.sum(2) != 0]
        return actions, action_dist.log_prob(actions), entropy, v.detach().view(-1).cpu().numpy(), probs.detach().cpu().numpy()

    def backprop(self,
                 hps,
                 obs,
                 actions,
                 old_logprobs,
                 returns,
                 value_loss_scale,
                 advantages,
                 old_values,
                 action_masks,
                 old_probs,
                 privileged_obs):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()

        batch_size = obs.size()[0]

        x, x_privileged = self.latents(obs, privileged_obs)
        probs = F.softmax(self.policy_head(x), dim=1).view(batch_size, 8, self.allies).permute(0, 2, 1)
        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        probs = probs * action_masks + 1e-8

        dist = distributions.Categorical(probs)
        entropy = dist.entropy()
        logprobs = dist.log_prob(actions)
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
        if self.use_privileged:
            vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        else:
            vin = pooled.view(batch_size, -1)
        values = self.value_head(vin).view(batch_size)
        clipped_values = old_values + torch.clamp(values - old_values, -hps.cliprange, hps.cliprange)
        vanilla_value_loss = (values - returns) ** 2
        clipped_value_loss = (clipped_values - returns) ** 2
        if hps.clip_vf:
            value_loss = torch.max(vanilla_value_loss, clipped_value_loss).mean()
        else:
            value_loss = vanilla_value_loss.mean()

        entropy_loss = hps.entropy_bonus * entropy.mean()

        loss = policy_loss + value_loss_scale * value_loss + entropy_loss
        loss /= hps.batches_per_update
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist(), approxkl.data.tolist(), clipfrac.data.tolist()

    def forward(self, x, x_privileged):
        batch_size = x.size()[0]
        x, x_privileged = self.latents(x, x_privileged)

        pooled = F.avg_pool2d(x, kernel_size=(self.allies, 1))
        if self.use_privileged:
            vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        else:
            vin = pooled.view(batch_size, -1)
        values = self.value_head(vin).view(batch_size, -1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=1)

        return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values

    def logits(self, x, x_privileged):
        x, x_privileged = self.latents(x, x_privileged)
        return self.policy_head(x)

    def latents(self, x, x_privileged):
        if self.fp16:
            x = x.half()
            x_privileged = x_privileged.half()

        endallies = (DSTRIDE + GLOBAL_FEATURES) * self.allies
        endmins = endallies + MSTRIDE * self.minerals * self.allies
        enddrones = endmins + DSTRIDE * self.drones * self.allies

        batch_size = x.size()[0]
        # global features and properties of the drone controlled by this network
        xd = x[:, :endallies]
        xd = self.self_net(xd)

        if self.minerals > 0:
            # properties of closest minerals
            xm = x[:, endallies:endmins]
            xm = self.mineral_net(xm)

        if self.drones > 0:
            # properties of the closest drones
            xe = x[:, endmins:enddrones]
            xe = self.drone_net(xe)

        # properties of global drones
        if self.use_privileged:
            x_privileged = self.privileged_net(x_privileged)
        else:
            x_privileged = None

        if self.minerals == 0:
            x = torch.cat((xd, xe), dim=1)
        elif self.drones == 0:
            x = torch.cat((xd, xm), dim=1)
        else:
            x = torch.cat((xd, xe, xm), dim=1)

        for fc, fc_norm in zip(self.fc_layers, self.fc_norm_layers):
            x = F.relu(fc(x))
            x = x.view(batch_size * self.allies, self.width, 1)
            x = fc_norm(x)
            x = x.view(batch_size, self.width, self.allies, 1)

        return x, x_privileged

    def param_groups(self):
        group0 = [
            *self.self_net.parameters(),
            *(self.mineral_net.parameters() if self.minerals > 0 else []),
            *(self.drone_net.parameters() if self.drones > 0 else []),
            *(self.privileged_net.parameters() if self.use_privileged else []),
        ]
        group1 = [
            *self.fc_layers.parameters(),
        ]
        group2 = [
            *self.value_head.parameters(),
            *self.policy_head.parameters(),
        ]
        grouped_params = len(group0) + len(group1) + len(group2)
        actual_params = len(list(self.parameters()))
        assert grouped_params == actual_params,\
                f'Missing parameters in group: {grouped_params} != {actual_params}'
        return group0, group1, group2

