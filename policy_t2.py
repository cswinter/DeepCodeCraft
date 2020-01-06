import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from gym_codecraft.envs.codecraft_vec_env import DEFAULT_OBS_CONFIG, GLOBAL_FEATURES, MSTRIDE, DSTRIDE


class TransformerPolicy2(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward_ratio,
                 dropout,
                 small_init_pi,
                 zero_init_vf,
                 fp16,
                 norm,
                 obs_config=DEFAULT_OBS_CONFIG,
                 use_privileged=False):
        super(TransformerPolicy2, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'

        self.version = 'transformer_v2'

        self.kwargs = dict(
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

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward_ratio = dim_feedforward_ratio
        self.dropout = dropout

        self.fp16 = fp16
        self.use_privileged = use_privileged

        if norm == 'none':
            norm_fn = lambda x: nn.Sequential()
        elif norm == 'batchnorm':
            norm_fn = lambda n: nn.BatchNorm2d(n)
        elif norm == 'layernorm':
            norm_fn = lambda n: nn.LayerNorm(n)
        else:
            raise Exception(f'Unexpected normalization layer {norm}')

        self.self_embedding = InputEmbedding(DSTRIDE + GLOBAL_FEATURES, d_model, norm_fn)
        self.self_resblock = FFResblock(d_model, d_model * dim_feedforward_ratio, norm_fn)
        # TODO: same embedding for self and other drones?
        self.drone_embedding = InputEmbedding(DSTRIDE, d_model, norm_fn)
        self.drone_resblock = FFResblock(d_model, d_model * dim_feedforward_ratio, norm_fn)

        if self.minerals > 0:
            self.mineral_embedding = InputEmbedding(MSTRIDE, d_model, norm_fn)
            self.mineral_resblock = FFResblock(d_model, d_model * dim_feedforward_ratio, norm_fn)

        if use_privileged:
            # TODO
            pass

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.downscale = nn.Linear(d_model, d_model // 64)

        self.final_layer = nn.Sequential(
            FFResblock(d_model, d_model * dim_feedforward_ratio, norm_fn),
            nn.Linear(d_model, d_model * dim_feedforward_ratio),
            nn.ReLU(),
        )


        # TODO: just input final drone item?
        self.policy_head = nn.Linear(d_model * dim_feedforward_ratio, 8)
        if small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        self.value_head = nn.Linear(d_model * dim_feedforward_ratio, 1)
        if zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

        self.epsilon = 1e-4 if fp16 else 1e-8

    def evaluate(self, observation, action_masks, privileged_obs):
        if self.fp16:
            action_masks = action_masks.half()
        probs, v = self.forward(observation, privileged_obs)
        probs = probs.view(-1, self.allies, 8)
        probs = probs * action_masks + self.epsilon  # Add small value to prevent crash when no action is possible
        # We get device-side assert when using fp16 here (needs more investigation)
        # print(probs)
        action_dist = distributions.Categorical(probs.float() if self.fp16 else probs)
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
            action_masks = action_masks.half()
            old_logprobs = old_logprobs.half()

        x, x_privileged = self.latents(obs, privileged_obs)
        vin = F.max_pool2d(x, kernel_size=(self.allies, 1))
        values = self.value_head(vin).view(-1)
        # TODO
        #if self.use_privileged:
        #    vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        #else:
        #    vin = pooled.view(batch_size, -1)
        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=2)
        probs = probs.view(-1, self.allies, 8)

        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        # TODO: mask actions by setting logits to -inf?
        probs = probs * action_masks + self.epsilon

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
        # TODO
        #if self.use_privileged:
        #    vin = torch.cat([pooled.view(batch_size, -1), x_privileged.view(batch_size, -1)], dim=1)
        #else:
        #    vin = pooled.view(batch_size, -1)
        vin = F.max_pool2d(x, kernel_size=(self.allies, 1))
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=2)

        # return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values
        return probs, values

    def logits(self, x, x_privileged):
        x, x_privileged = self.latents(x, x_privileged)
        return self.policy_head(x)

    def latents(self, x, x_privileged):
        if self.fp16:
            # Normalization layers perform fp16 conversion for x after normalization
            x_privileged = x_privileged.half()

        endallies = (DSTRIDE + GLOBAL_FEATURES) * self.allies
        endmins = endallies + MSTRIDE * self.minerals * self.allies
        enddrones = endmins + DSTRIDE * self.drones * self.allies

        # Dimensions: batch, items, properties

        batch_size = x.size()[0]
        # global features and properties of the drone controlled by this network
        xs = x[:, :endallies].view(batch_size, self.allies, 1, DSTRIDE + GLOBAL_FEATURES)
        # Ensures that at least one mask element is present, otherwise we get NaN in attention softmax
        # If the ally is nonexistant, it's output will be ignored anyway
        # Derive from original tensor to keep on device
        mask = (xs[:, :, :, GLOBAL_FEATURES + 7] * 0 != 0).view(batch_size, self.allies, 1) # Position 7 is hitpoints
        actual_mask = (xs[:, :, :, GLOBAL_FEATURES + 7] == 0).view(batch_size, self.allies, 1)
        x_emb_self = self.self_resblock(self.self_embedding(xs, ~actual_mask))
        x_emb = x_emb_self

        # origin = xs[:, :, :, GLOBAL_FEATURES:GLOBAL_FEATURES+2].view(batch_size, self.allies, 2)
        # direction = xs[:, :, :, GLOBAL_FEATURES+2:GLOBAL_FEATURES+4].view(batch_size, self.allies, 2)

        if self.minerals > 0:
            # properties of closest minerals
            xm = x[:, endallies:endmins].view(batch_size, self.allies, self.minerals, MSTRIDE)

            # xm_pos = xm[:, :, :, 0:2].view(batch_size, self.minerals, 2)
            # xm_relpos = relative_positions(origin, direction, xm_pos)
            # xm[:, :, :, 0:2] = xm_relpos.view(batch_size, self.allies, self.minerals, 2)

            mineral_mask = (xm[:, :, :, 3] == 0).view(batch_size, self.allies, self.minerals)
            xm = self.mineral_resblock(self.mineral_embedding(xm, ~mineral_mask))
            mask = torch.cat((mask, mineral_mask), dim=2)
            x_emb = torch.cat((x_emb, xm), dim=2)

        if self.drones > 0:
            # properties of closest minerals
            xd = x[:, endmins:enddrones].view(batch_size, self.allies, self.drones, DSTRIDE)
            drone_mask = (xd[:, :, :, 7] == 0).view(batch_size, self.allies, self.drones) # Position 7 is hitpoints
            xd = self.drone_resblock(self.drone_embedding(xd, ~drone_mask))
            mask = torch.cat((mask, drone_mask), dim=2)
            x_emb = torch.cat((x_emb, xd), dim=2)

        mask = mask.view(batch_size * self.allies, 1 + self.minerals + self.drones)

        # TODO: self.use_privileged
        if self.use_privileged:
            x_privileged = self.privileged_net(x_privileged)
        else:
            x_privileged = None

        # Transformer input dimensions are: Sequence length, Batch size, Embedding size
        source = x_emb.view(batch_size * self.allies, 1 + self.minerals + self.drones, self.d_model).permute(1, 0, 2)
        target = x_emb_self.view(batch_size * self.allies, 1, self.d_model).permute(1, 0, 2)
        x, attn_weights = self.multihead_attention(
            query=target,
            key=source,
            value=source,
            key_padding_mask=mask
        )
        x = self.norm1(x + target)
        x = x.permute(1, 0, 2)
        # mins = xm.view(batch_size, self.minerals, self.d_model) * (1 - mineral_mask.float()).view(batch_size, self.minerals, 1)
        # mins = F.relu(self.downscale(mins))
        # nearby_map = spatial_scatter(
        #    items=mins,
        #    positions=xm_relpos,
        #    nray=8,
        #    nring=8,
        #    inner_radius=50 / 2000,
        #)
        #nearby_map = nearby_map.reshape(batch_size, self.allies, self.d_model)
        # x = torch.cat([x, nearby_map], dim=2)

        x = self.final_layer(x)
        x = x.view(batch_size, self.allies, self.d_model * self.dim_feedforward_ratio)

        return x, x_privileged

    def param_groups(self):
        # TODO?
        pass


# Computes a running mean/variance of input features and performs normalization.
# https://www.johndcook.com/blog/standard_deviation/
class InputNorm(nn.Module):
    def __init__(self, num_features, cliprange=5):
        super(InputNorm, self).__init__()

        self.cliprange = cliprange
        self.register_buffer('count', torch.tensor(0))
        self.register_buffer('mean', torch.zeros(num_features))
        self.register_buffer('squares_sum', torch.zeros(num_features))
        self.fp16 = False

    def update(self, input, mask):
        features = input.size()[-1]
        if mask is not None:
            input = input[mask, :]
        else:
            input = input.reshape(-1, features)
        count = input.numel() / features
        if count == 0:
            return
        mean = input.mean(dim=0)
        if self.count == 0:
            self.count += count
            self.mean = mean
            self.squares_sum = ((input - mean) * (input - mean)).sum(dim=0)
        else:
            self.count += count
            new_mean = self.mean + (mean - self.mean) * count / self.count
            # This is probably not quite right because it applies multiple updates simultaneously.
            self.squares_sum = self.squares_sum + ((input - self.mean) * (input - new_mean)).sum(dim=0)
            self.mean = new_mean

    def forward(self, input, mask=None):
        with torch.no_grad():
            if self.training:
                self.update(input, mask=mask)
            if self.count > 1:
                input = (input - self.mean) / self.stddev()
            input = torch.clamp(input, -self.cliprange, self.cliprange)
            if (input == float('-inf')).sum() > 0 \
                    or (input == float('inf')).sum() > 0 \
                    or (input != input).sum() > 0:
                print(input)
                print(self.squares_sum)
                print(self.stddev())
                print(input)
                raise Exception("OVER/UNDERFLOW DETECTED!")

        return input.half() if self.fp16 else input

    def enable_fp16(self):
        # Convert buffers back to fp32, fp16 has insufficient precision and runs into overflow on squares_sum
        self.float()
        self.fp16 = True

    def stddev(self):
        sd = torch.sqrt(self.squares_sum / (self.count - 1))
        sd[sd == 0] = 1
        return sd


class InputEmbedding(nn.Module):
    def __init__(self, d_in, d_model, norm_fn):
        super(InputEmbedding, self).__init__()

        self.normalize = InputNorm(d_in)
        self.linear = nn.Linear(d_in, d_model)
        self.norm = norm_fn(d_model)

    def forward(self, x, mask=None):
        x = self.normalize(x, mask)
        x = F.relu(self.linear(x))
        x = self.norm(x)
        return x


class FFResblock(nn.Module):
    def __init__(self, d_model, d_ff, norm_fn):
        super(FFResblock, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.norm = norm_fn(d_model)

        # self.linear_2.weight.data.fill_(0.0)
        # self.linear_2.bias.data.fill_(0.0)

    def forward(self, x, mask=None):
        x2 = F.relu(self.linear_1(x))
        x = x + F.relu(self.linear_2(x2))
        x = self.norm(x)
        return x

