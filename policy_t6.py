import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch_scatter import scatter_add

from gather import topk_by
from multihead_attention import MultiheadAttention
import spatial


class TransformerPolicy6(nn.Module):
    def __init__(self, hps, obs_config):
        super(TransformerPolicy6, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'
        assert obs_config.drones >= obs_config.allies
        assert not hps.use_privileged or (hps.nmineral > 0 and hps.nally > 0 and (hps.nenemy > 0 or hps.ally_enemy_same))

        self.version = 'transformer_v6'

        self.kwargs = dict(
            hps=hps,
            obs_config=obs_config
        )

        self.hps = hps
        self.obs_config = obs_config
        self.agents = hps.agents
        self.nally = hps.nally
        self.nenemy = hps.nenemy
        self.nmineral = hps.nmineral
        self.nconstant = hps.nconstant
        self.ntile = hps.ntile
        self.nitem = hps.nally + hps.nenemy + hps.nmineral + hps.nconstant + hps.ntile
        self.fp16 = hps.fp16
        self.d_agent = hps.d_agent
        self.d_item = hps.d_item
        self.naction = hps.objective.naction() + obs_config.extra_actions()

        if hasattr(obs_config, 'global_drones'):
            self.global_drones = obs_config.global_drones
        else:
            self.global_drones = 0

        if hps.norm == 'none':
            norm_fn = lambda x: nn.Sequential()
        elif hps.norm == 'batchnorm':
            norm_fn = lambda n: nn.BatchNorm2d(n)
        elif hps.norm == 'layernorm':
            norm_fn = lambda n: nn.LayerNorm(n)
        else:
            raise Exception(f'Unexpected normalization layer {hps.norm}')

        self.agent_embedding = ItemBlock(
            obs_config.dstride() + obs_config.global_features(),
            hps.d_agent, hps.d_agent * hps.dff_ratio, norm_fn, True,
            keep_abspos=True,
            mask_feature=7,  # Feature 7 is hitpoints
            relpos=False,
        )
        if hps.ally_enemy_same:
            self.drone_net = ItemBlock(
                obs_config.dstride(),
                hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                keep_abspos=hps.obs_keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=hps.nally+hps.nenemy,
            )
        else:
            self.ally_net = ItemBlock(
                obs_config.dstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                keep_abspos=hps.obs_keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=hps.nally,
            )
            self.enemy_net = ItemBlock(
                obs_config.dstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                keep_abspos=hps.obs_keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=hps.nenemy,
            )
        self.mineral_net = ItemBlock(
            obs_config.mstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
            keep_abspos=hps.obs_keep_abspos,
            mask_feature=2,  # Feature 2 is size
            topk=hps.nmineral,
        )
        if hps.ntile > 0:
            self.tile_net = ItemBlock(
                obs_config.tstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                keep_abspos=hps.obs_keep_abspos,
                mask_feature=2,  # Feature is elapsed since last visited time
                topk=hps.ntile,
            )
        if hps.nconstant > 0:
            self.constant_items = nn.Parameter(torch.normal(0, 1, (hps.nconstant, hps.d_item)))

        if hps.use_privileged:
            self.pmineral_net = ItemBlock(
                obs_config.mstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                keep_abspos=True, relpos=False, mask_feature=2,
            )
            if hps.ally_enemy_same:
                self.pdrone_net = ItemBlock(
                    obs_config.dstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )
            else:
                self.pally_net = ItemBlock(
                    obs_config.dstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )
                self.penemy_net = ItemBlock(
                    obs_config.dstride(), hps.d_item, hps.d_item * hps.dff_ratio, norm_fn, hps.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )

        if hps.item_item_attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=hps.d_item, nhead=8)
            self.item_item_attn = nn.TransformerEncoder(encoder_layer, num_layers=hps.item_item_attn_layers)
        else:
            self.item_item_attn = None

        self.multihead_attention = MultiheadAttention(
            embed_dim=hps.d_agent,
            kdim=hps.d_item,
            vdim=hps.d_item,
            num_heads=hps.nhead,
            dropout=hps.dropout,
        )
        self.linear1 = nn.Linear(hps.d_agent, hps.d_agent * hps.dff_ratio)
        self.linear2 = nn.Linear(hps.d_agent * hps.dff_ratio, hps.d_agent)
        self.norm1 = nn.LayerNorm(hps.d_agent)
        self.norm2 = nn.LayerNorm(hps.d_agent)

        self.map_channels = hps.d_agent // (hps.nm_nrings * hps.nm_nrays)
        map_item_channels = self.map_channels - 2 if self.hps.map_embed_offset else self.map_channels
        self.downscale = nn.Linear(hps.d_item, map_item_channels)
        self.norm_map = norm_fn(map_item_channels)
        self.conv1 = spatial.ZeroPaddedCylindricalConv2d(
            self.map_channels, hps.dff_ratio * self.map_channels, kernel_size=3)
        self.conv2 = spatial.ZeroPaddedCylindricalConv2d(
            hps.dff_ratio * self.map_channels, self.map_channels, kernel_size=3)
        self.norm_conv = norm_fn(self.map_channels)

        final_width = hps.d_agent
        if hps.nearby_map:
            final_width += hps.d_agent
        self.final_layer = nn.Sequential(
            nn.Linear(final_width, hps.d_agent * hps.dff_ratio),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hps.d_agent * hps.dff_ratio, self.naction)
        if hps.small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        if hps.use_privileged:
            self.value_head = nn.Linear(hps.d_agent * hps.dff_ratio + 2 * hps.d_item, 1)
        else:
            self.value_head = nn.Linear(hps.d_agent * hps.dff_ratio, 1)
        if hps.zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

        self.epsilon = 1e-4 if hps.fp16 else 1e-8

    def evaluate(self, observation, action_masks, privileged_obs):
        if self.fp16:
            action_masks = action_masks.half()
        action_masks = action_masks[:, :self.agents, :]
        probs, v = self.forward(observation, privileged_obs, action_masks)
        probs = probs.view(-1, self.agents, self.naction)
        if action_masks.size(2) != self.naction:
            nbatch, nagent, naction = action_masks.size()
            zeros = torch.zeros(nbatch, nagent, self.naction - naction).to(observation.device)
            action_masks = torch.cat([action_masks, zeros], dim=2)
        probs = probs * action_masks + self.epsilon  # Add small value to prevent crash when no action is possible
        # We get device-side assert when using fp16 here (needs more investigation)
        action_dist = distributions.Categorical(probs.float() if self.fp16 else probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()[action_masks.sum(2) > 1]
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
                 privileged_obs,
                 split_reward):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()
            action_masks = action_masks.half()
            old_logprobs = old_logprobs.half()

        action_masks = action_masks[:, :self.agents, :]
        x, (pitems, pmask) = self.latents(obs, privileged_obs, action_masks)
        batch_size = x.size()[0]

        vin = x.max(dim=1).values.view(batch_size, self.d_agent * self.hps.dff_ratio)
        if self.hps.use_privileged:
            pitems_max = pitems.max(dim=1).values
            pitems_avg = pitems.sum(dim=1) / torch.clamp_min((~pmask).float().sum(dim=1), min=1).unsqueeze(-1)
            vin = torch.cat([vin, pitems_max, pitems_avg], dim=1)
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=2)
        probs = probs.view(-1, self.agents, self.naction)

        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        # TODO: mask actions by setting logits to -inf?
        probs = probs * action_masks + self.epsilon

        active_agents = torch.clamp_min((action_masks.sum(dim=2) > 0).float().sum(dim=1), min=1)

        dist = distributions.Categorical(probs)
        entropy = dist.entropy()
        logprobs = dist.log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = advantages.view(-1, 1)
        if split_reward:
            advantages = advantages / active_agents.view(-1, 1)
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

    def forward(self, x, x_privileged, action_masks):
        batch_size = x.size()[0]
        x, (pitems, pmask) = self.latents(x, x_privileged, action_masks)

        vin = x.max(dim=1).values.view(batch_size, self.d_agent * self.hps.dff_ratio)
        if self.hps.use_privileged:
            pitems_max = pitems.max(dim=1).values
            pitems_avg = pitems.sum(dim=1) / torch.clamp_min((~pmask).float().sum(dim=1), min=1).unsqueeze(-1)
            vin = torch.cat([vin, pitems_max, pitems_avg], dim=1)
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=2)

        # return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values
        return probs, values

    def logits(self, x, x_privileged, action_masks):
        x, x_privileged = self.latents(x, x_privileged, action_masks)
        return self.policy_head(x)

    def latents(self, x, x_privileged, action_masks):
        if self.fp16:
            # Normalization layers perform fp16 conversion for x after normalization
            x_privileged = x_privileged.half()

        batch_size = x.size()[0]

        endglobals = self.obs_config.endglobals()
        endallies = self.obs_config.endallies()
        endenemies = self.obs_config.endenemies()
        endmins = self.obs_config.endmins()
        endtiles = self.obs_config.endtiles()
        endallenemies = self.obs_config.endallenemies()

        globals = x[:, :endglobals]

        # properties of the drone controlled by this network
        xagent = x[:, endglobals:endallies]\
            .view(batch_size, self.obs_config.allies, self.obs_config.dstride())[:, :self.agents, :]
        globals = globals.view(batch_size, 1, self.obs_config.global_features()) \
            .expand(batch_size, self.agents, self.obs_config.global_features())
        xagent = torch.cat([xagent, globals], dim=2)
        agent_active = action_masks.sum(2) > 1
        active_agent_groups = []
        active_agent_indices = []
        for b in range(0, batch_size):
            for a in range(0, xagent.size(1)):
                if agent_active[b, a] == 1:
                    active_agent_groups.append(b)
                    active_agent_indices.append(b * xagent.size(1) + a)
        xagent = xagent[agent_active]
        agents, _, mask_agent = self.agent_embedding(xagent)

        origin = xagent[:, 0:2].clone()
        direction = xagent[:, 2:4].clone()

        if self.hps.ally_enemy_same:
            xdrone = x[:, endglobals:endenemies].view(batch_size, self.obs_config.drones, self.obs_config.dstride())
            items, relpos, mask = self.drone_net(xdrone, active_agent_groups, origin, direction)
        else:
            xally = x[:, endglobals:endallies].view(batch_size, self.obs_config.allies, self.obs_config.dstride())
            items, relpos, mask = self.ally_net(xally, active_agent_groups, origin, direction)
        # Ensure that at least one item is not masked out to prevent NaN in transformer softmax
        mask[:, 0] = 0

        if self.nenemy > 0 and not self.hps.ally_enemy_same:
            eobs = self.obs_config.drones - self.obs_config.allies
            xe = x[:, endallies:endenemies].view(batch_size, eobs, self.obs_config.dstride())

            items_e, relpos_e, mask_e = self.enemy_net(xe, active_agent_groups, origin, direction)
            items = torch.cat([items, items_e], dim=1)
            mask = torch.cat([mask, mask_e], dim=1)
            relpos = torch.cat([relpos, relpos_e], dim=1)

        if self.nmineral > 0:
            xm = x[:, endenemies:endmins].view(batch_size, self.obs_config.minerals, self.obs_config.mstride())

            items_m, relpos_m, mask_m = self.mineral_net(xm, active_agent_groups, origin, direction)
            items = torch.cat([items, items_m], dim=1)
            mask = torch.cat([mask, mask_m], dim=1)
            relpos = torch.cat([relpos, relpos_m], dim=1)

        if self.ntile > 0:
            xt = x[:, endmins:endtiles].view(batch_size, self.obs_config.tiles, self.obs_config.tstride())

            items_t, relpos_t, mask_t = self.tile_net(xt, active_agent_groups, origin, direction)
            items = torch.cat([items, items_t], dim=1)
            mask = torch.cat([mask, mask_t], dim=1)
            relpos = torch.cat([relpos, relpos_t], dim=1)

        if self.nconstant > 0:
            # TODO: filtered to active agents
            items_c = self.constant_items\
                .view(1, 1, self.nconstant, self.hps.d_item)\
                .repeat((batch_size, self.agents, 1, 1))
            mask_c = torch.zeros(batch_size, self.agents, self.nconstant).bool().to(x.device)
            items = torch.cat([items, items_c], dim=1)
            mask = torch.cat([mask, mask_c], dim=1)

        if self.hps.use_privileged:
            xally = x[:, endglobals:endallies].view(batch_size, self.obs_config.allies, self.obs_config.dstride())
            eobs = self.obs_config.drones - self.obs_config.allies
            xenemy = x[:, endtiles:endallenemies].view(batch_size, eobs, self.obs_config.dstride())
            if self.hps.ally_enemy_same:
                xdrone = torch.cat([xally, xenemy], dim=1)
                pitems, _, pmask = self.pdrone_net(xdrone)
            else:
                pitems, _, pmask = self.pally_net(xally)
                pitems_e, _, pmask_e = self.penemy_net(xenemy)
                pitems = torch.cat([pitems, pitems_e], dim=1)
                pmask = torch.cat([pmask, pmask_e], dim=1)
            xm = x[:, endenemies:endmins].view(batch_size, self.obs_config.minerals, self.obs_config.mstride())
            pitems_m, _, pmask_m = self.pmineral_net(xm)
            pitems = torch.cat([pitems, pitems_m], dim=1)
            pmask = torch.cat([pmask, pmask_m], dim=1)
            if self.item_item_attn:
                pmask_nonzero = pmask.clone()
                pmask_nonzero[:, 0] = False
                pitems = self.item_item_attn(
                    pitems.permute(1, 0, 2),
                    src_key_padding_mask=pmask_nonzero,
                ).permute(1, 0, 2)
                if (pitems != pitems).sum() > 0:
                    print(pmask)
                    print(pitems)
                    raise Exception("NaN!")
        else:
            pitems = None
            pmask = None

        # Transformer input dimensions are: Sequence length, Batch size, Embedding size
        # source = items.view(batch_size * self.agents, self.nitem, self.d_item).permute(1, 0, 2)
        # target = agents.view(1, batch_size * self.agents, self.d_agent)
        source = items.permute(1, 0, 2)
        target = agents.view(1, -1, self.d_agent)
        x, attn_weights = self.multihead_attention(
            query=target,
            key=source,
            value=source,
            key_padding_mask=mask,
        )
        x = self.norm1(x + target)
        x2 = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        # x = x.view(batch_size, self.agents, self.d_agent)

        if self.hps.nearby_map:
            items = self.norm_map(F.relu(self.downscale(items)))
            items = items * (1 - mask.float().unsqueeze(-1))
            nearby_map = spatial.spatial_scatter(
                items=items[:, :, :(self.nitem - self.nconstant - self.ntile), :],
                positions=relpos[:, :, :self.nitem - self.nconstant - self.ntile],
                nray=self.hps.nm_nrays,
                nring=self.hps.nm_nrings,
                inner_radius=self.hps.nm_ring_width,
                embed_offsets=self.hps.map_embed_offset,
            ).view(batch_size * self.agents, self.map_channels, self.hps.nm_nrings, self.hps.nm_nrays)
            if self.hps.map_conv:
                nearby_map2 = self.conv2(F.relu(self.conv1(nearby_map)))
                nearby_map2 = nearby_map2.permute(0, 3, 2, 1)
                nearby_map = nearby_map.permute(0, 3, 2, 1)
                nearby_map = self.norm_conv(nearby_map + nearby_map2)
            nearby_map = nearby_map.reshape(batch_size, self.agents, self.d_agent)
            x = torch.cat([x, nearby_map], dim=2)

        x = self.final_layer(x).squeeze(0)
        output = torch.zeros(batch_size * self.agents, self.d_agent * self.hps.dff_ratio).to(x.device)
        indices = torch.tensor(active_agent_indices).to(x.device)
        scatter_add(x, index=indices, dim=0, out=output)
        x = output.view(batch_size, self.agents, self.d_agent * self.hps.dff_ratio)
        #x = x * (~mask_agent).float().unsqueeze(-1)

        return x, (pitems, pmask)

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
        if mask is not None:
            if len(input.size()) == 3:
                batch_size, nitem, features = input.size()
                assert (batch_size, nitem) == mask.size()
            elif len(input.size()) == 2:
                batch_size, features = input.size()
                assert (batch_size,) == mask.size()
            else:
                raise Exception(f'Expecting 3 or 4 dimensions, actual: {len(input.size())}')
            input = input[mask, :]
        else:
            features = input.size()[-1]
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


class ItemBlock(nn.Module):
    def __init__(self, d_in, d_model, d_ff, norm_fn, resblock, keep_abspos, mask_feature, relpos=True, topk=None):
        super(ItemBlock, self).__init__()

        if relpos:
            if keep_abspos:
                d_in += 3
            else:
                d_in += 1
        self.embedding = InputEmbedding(d_in, d_model, norm_fn)
        self.mask_feature = mask_feature
        self.keep_abspos = keep_abspos
        self.topk = topk
        if resblock:
            self.resblock = FFResblock(d_model, d_ff, norm_fn)

    def forward(self, x, indices=None, origin=None, direction=None):
        if origin is not None:
            batch_agents, _ = origin.size()

            x = x[indices]
            pos = x[:, :, 0:2]
            relpos = spatial.unbatched_relative_positions(origin, direction, pos)
            dist = relpos.norm(p=2, dim=2)
            direction = relpos / (dist.unsqueeze(-1) + 1e-8)

            if self.keep_abspos:
                x = torch.cat([x, direction, torch.sqrt(dist.unsqueeze(-1))], dim=2)
            else:
                x = torch.cat([direction, x[:, :, 2:], torch.sqrt(dist.unsqueeze(-1))], dim=2)

            if self.topk is not None:
                empty = (x[:, :, self.mask_feature] == 0).float()
                key = -dist - empty * 1e8
                x = topk_by(values=x, vdim=1, keys=key, kdim=1, k=self.topk)
                relpos = topk_by(values=relpos, vdim=1, keys=key, kdim=1, k=self.topk)

            mask = x[:, :, self.mask_feature] == 0
        else:
            relpos = None
            if x.dim() == 3:
                mask = x[:, :, self.mask_feature] == 0
            else:
                mask = x[:, self.mask_feature] == 0

        x = self.embedding(x, ~mask)
        if self.resblock is not None:
            x = self.resblock(x)
        x = x * (~mask).unsqueeze(-1).float()

        return x, relpos, mask

