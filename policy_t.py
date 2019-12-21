import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from gym_codecraft.envs.codecraft_vec_env import DEFAULT_OBS_CONFIG, GLOBAL_FEATURES, MSTRIDE, DSTRIDE
from list_net import ListNet


class TransformerPolicy(nn.Module):
    def __init__(self,
                 transformer_layers,
                 d_model,
                 nhead,
                 dim_feedforward_ratio,
                 dropout,
                 disable_transformer,
                 small_init_pi,
                 zero_init_vf,
                 fp16,
                 norm,
                 obs_config=DEFAULT_OBS_CONFIG,
                 use_privileged=False):
        super(TransformerPolicy, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'

        self.version = 'transformer_v1'

        self.kwargs = dict(
            transformer_layers=transformer_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward_ratio=dim_feedforward_ratio,
            dropout=dropout,
            small_init_pi=small_init_pi,
            zero_init_vf=zero_init_vf,
            fp16=fp16,
            use_privileged=use_privileged,
            norm=norm,
            obs_config=obs_config,
        )

        self.obs_config = obs_config
        self.allies = obs_config.allies
        self.drones = obs_config.drones
        self.minerals = obs_config.minerals
        if hasattr(obs_config, 'global_drones'):
            self.global_drones = obs_config.global_drones
        else:
            self.global_drones = 0

        self.transformer_layers = transformer_layers
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward_ratio = dim_feedforward_ratio
        self.dropout = dropout
        self.disable_transformer = disable_transformer

        self.fp16 = fp16
        self.use_privileged = use_privileged

        if norm == 'none':
            norm_fn = lambda x: None
        elif norm == 'batchnorm':
            norm_fn = lambda n: nn.BatchNorm2d(n)
        elif norm == 'layernorm':
            norm_fn = lambda n: nn.LayerNorm(n)
        else:
            raise Exception(f'Unexpected normalization layer {norm}')

        self.drone_embedding = ItemEmbedding(
            DSTRIDE + GLOBAL_FEATURES,
            d_model,
            d_model * dim_feedforward_ratio,
            norm_fn,
        )
        # TODO: other drones

        if self.minerals > 0:
            self.mineral_embedding = ItemEmbedding(
                MSTRIDE,
                d_model,
                d_model * dim_feedforward_ratio,
                norm_fn,
            )

        if use_privileged:
            # TODO
            pass

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * dim_feedforward_ratio,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers, norm=None)

        # TODO: just input final drone item?
        self.policy_head = nn.Linear(d_model * (self.minerals + self.allies), 8)
        if small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        self.value_head = nn.Linear(d_model * (self.minerals + self.allies), 1)
        if zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

    def evaluate(self, observation, action_masks, privileged_obs):
        if self.fp16:
            action_masks = action_masks.half()
        probs, v = self.forward(observation, privileged_obs)
        probs = probs.view(-1, 1, 8)
        probs = probs * action_masks + 1e-8  # Add small value to prevent crash when no action is possible
        print(f"probs: {probs.size()}")
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        print(f"es: {action_dist.entropy().size()} am: {action_masks.size()} ams: {action_masks.sum(1).size()}")
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

        print(obs.size())
        batch_size = obs.size()[0]

        x, x_privileged = self.latents(obs, privileged_obs)
        x = x.view(batch_size, (self.allies + self.minerals) * self.d_model)
        print(x.size())
        values = self.value_head(x).view(-1)
        # TODO
        #if self.use_privileged:
        #    vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        #else:
        #    vin = pooled.view(batch_size, -1)
        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=1)
        probs = probs.view(-1, 1, 8)

        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        # TODO: mask actions by setting logits to -inf?
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

        clipped_values = old_values + torch.clamp(values - old_values, -hps.cliprange, hps.cliprange)
        vanilla_value_loss = (values - returns) ** 2
        clipped_value_loss = (clipped_values - returns) ** 2
        if hps.clip_vf:
            value_loss = torch.max(vanilla_value_loss, clipped_value_loss).mean()
        else:
            value_loss = vanilla_value_loss.mean()

        entropy_loss = -hps.entropy_bonus * entropy.mean()

        loss = policy_loss + value_loss_scale * value_loss + entropy_loss
        loss /= hps.batches_per_update
        loss.backward()
        return policy_loss.data.tolist(), value_loss.data.tolist(), approxkl.data.tolist(), clipfrac.data.tolist()

    def forward(self, x, x_privileged):
        batch_size = x.size()[0]
        x, x_privileged = self.latents(x, x_privileged)
        x = x.view(batch_size, (self.allies + self.minerals) * self.d_model)
        # TODO
        #if self.use_privileged:
        #    vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        #else:
        #    vin = pooled.view(batch_size, -1)
        values = self.value_head(x).view(-1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=1)
        print(f"x: {x.size()} logits: {logits.size()} probs: {probs.size()}")

        # return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values
        return probs, values

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

        # Dimensions: batch, items, properties

        batch_size = x.size()[0]
        # global features and properties of the drone controlled by this network
        xd = x[:, :endallies].view(batch_size, self.allies, DSTRIDE + GLOBAL_FEATURES)
        xd = F.relu(self.drone_embedding(xd))

        if self.minerals > 0:
            # properties of closest minerals
            xm = x[:, endallies:endmins].view(batch_size, self.minerals, MSTRIDE)
            xm = self.mineral_embedding(xm)
            x = torch.cat((xd, xm), dim=1)

        # TODO: self.use_privileged
        if self.use_privileged:
            x_privileged = self.privileged_net(x_privileged)
        else:
            x_privileged = None

        if not self.disable_transformer:
            x = self.transformer(x)

        return x, x_privileged

    def param_groups(self):
        # TODO?
        pass


class ItemEmbedding(nn.Module):
    def __init__(self, d_in, d_model, d_ff, norm_fn):
        super(ItemEmbedding, self).__init__()

        self.linear_0 = nn.Linear(d_in, d_model)
        self.norm_0 = norm_fn(d_model)

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm_2 = norm_fn(d_model)

        # self.linear_2.weight.data.fill_(0.0)
        # self.linear_2.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.norm_0(F.relu(self.linear_0(x)))

        x2 = F.relu(self.linear_1(x))
        x = x + F.relu(self.linear_2(x2))
        x = self.norm_2(x)
        return x
