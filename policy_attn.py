import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch_scatter import scatter_add, scatter_max
from gaussian_attention import GaussianAttention

from transformer import SpatialTransformer
import spatial


class SpatialAttnPolicy(nn.Module):
    def __init__(self, hps, obs_config):
        super(SpatialAttnPolicy, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'
        assert obs_config.drones >= obs_config.allies
        assert not hps.use_privileged or (hps.nmineral > 0 and hps.nally > 0 and (hps.nenemy > 0 or hps.ally_enemy_same))

        assert hps.nally == obs_config.allies
        assert hps.nenemy == obs_config.drones - obs_config.allies
        assert hps.nmineral == obs_config.minerals
        assert hps.ntile == obs_config.tiles

        self.version = 'transformer_attn'

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

        endglobals = self.obs_config.endglobals()
        endallies = self.obs_config.endallies()
        endenemies = self.obs_config.endenemies()
        endmins = self.obs_config.endmins()
        endtiles = self.obs_config.endtiles()
        endallenemies = self.obs_config.endallenemies()

        self.agent_embedding = ItemBlock(
            obs_config.dstride() + obs_config.global_features(),
            hps.d_agent, hps.d_agent * hps.dff_ratio, norm_fn, True,
            mask_feature=7,  # Feature 7 is hitpoints
        )
        self.relpos_net = ItemBlock(
            3, hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff
        )

        self.item_nets = nn.ModuleList()
        if hps.ally_enemy_same:
            self.item_nets.append(PosItemBlock(
                obs_config.dstride(),
                hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff,
                mask_feature=7,  # Feature 7 is hitpoints
                count=obs_config.drones,
                start=endglobals,
                end=endenemies,
            ))
        else:
            if self.nally > 0:
                self.item_nets.append(PosItemBlock(
                    obs_config.dstride(), hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff,
                    mask_feature=7,  # Feature 7 is hitpoints
                    count=obs_config.allies,
                    start=endglobals,
                    end=endallies,
                ))
            if self.nenemy > 0:
                self.item_nets.append(PosItemBlock(
                    obs_config.dstride(), hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff,
                    mask_feature=7,  # Feature 7 is hitpoints
                    count=obs_config.drones - self.obs_config.allies,
                    start=endallies,
                    end=endenemies,
                    start_privileged=endtiles if hps.use_privileged else None,
                    end_privileged=endallenemies if hps.use_privileged else None,
                ))
        if hps.nmineral > 0:
            self.item_nets.append(PosItemBlock(
                obs_config.mstride(), hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff,
                mask_feature=2,  # Feature 2 is size
                count=obs_config.minerals,
                start=endenemies,
                end=endmins,
            ))
        if hps.ntile > 0:
            self.item_nets.append(PosItemBlock(
                obs_config.tstride(), hps.d_item // 2, hps.d_item // 2 * hps.dff_ratio, norm_fn, hps.item_ff,
                mask_feature=2,  # Feature is elapsed since last visited time
                count=obs_config.tiles,
                start=endmins,
                end=endtiles,
            ))
        if hps.nconstant > 0:
            self.constant_items = nn.Parameter(torch.normal(0, 1, (hps.nconstant, hps.d_item)))

        if hps.item_item_attn_layers > 0:
            self.item_item_attn = SpatialTransformer(
                embed_dim=hps.d_item,
                kvdim=hps.d_item,
                nhead=hps.nhead,
                dff_ratio=hps.dff_ratio
            )
        else:
            self.item_item_attn = None

        if hps.spatial_attn:
            self.gattn = GaussianAttention(hps.nhead, scale=1000.0)
        self.transformer = SpatialTransformer(
            embed_dim=hps.d_agent,
            kvdim=hps.d_item,
            nhead=hps.nhead,
            dff_ratio=hps.dff_ratio,
        )

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
            self.value_head = nn.Linear(hps.d_agent * hps.dff_ratio + hps.d_item, 1)
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
        actions = actions[:, :self.agents]
        old_logprobs = old_logprobs[:, :self.agents]

        probs, values = self.forward(obs, privileged_obs, action_masks)
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
            policy_loss = -torch.min(vanilla_policy_loss, clipped_policy_loss).mean(dim=0).sum()
        else:
            policy_loss = -vanilla_policy_loss.mean(dim=0).sum()
        # TODO remove
        # Adjustement loss magnitude to keep parity with previous averaging scheme
        policy_loss /= 8

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
        x, indices, groups, (pitems, pmask) = self.latents(x, x_privileged, action_masks)

        if x.is_cuda:
            vin = torch.cuda.FloatTensor(batch_size, self.d_agent * self.hps.dff_ratio).fill_(0)
        else:
            vin = torch.zeros(batch_size, self.d_agent * self.hps.dff_ratio)
        scatter_max(x, index=groups, dim=0, out=vin)
        if self.hps.use_privileged:
            mask1k = 1000.0 * pmask.float().unsqueeze(-1)
            pitems_max = (pitems - mask1k).max(dim=1).values
            pitems_max[pitems_max == -1000.0] = 0.0
            pitems_avg = pitems.sum(dim=1) / torch.clamp_min((~pmask).float().sum(dim=1), min=1).unsqueeze(-1)
            vin = torch.cat([vin, pitems_max, pitems_avg], dim=1)
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)

        if x.is_cuda:
            padded_logits = torch.cuda.FloatTensor(batch_size * self.agents, self.naction).fill_(0)
        else:
            padded_logits = torch.zeros(batch_size * self.agents, self.naction)
        scatter_add(logits, index=indices, dim=0, out=padded_logits)
        padded_logits = padded_logits.view(batch_size, self.agents, self.naction)
        probs = F.softmax(padded_logits, dim=2)

        return probs, values

    def logits(self, x, x_privileged, action_masks):
        x, x_privileged = self.latents(x, x_privileged, action_masks)
        return self.policy_head(x)

    def latents(self, x, x_privileged, action_masks):
        batch_size = x.size()[0]

        endglobals = self.obs_config.endglobals()
        endallies = self.obs_config.endallies()

        globals = x[:, :endglobals]

        # properties of the drone controlled by this network
        xagent = x[:, endglobals:endallies]\
            .view(batch_size, self.obs_config.allies, self.obs_config.dstride())[:, :self.agents, :]
        globals = globals.view(batch_size, 1, self.obs_config.global_features()) \
            .expand(batch_size, self.agents, self.obs_config.global_features())
        xagent = torch.cat([xagent, globals], dim=2)

        nagents = xagent.size(1)

        agent_active = action_masks.sum(2) > 0
        # Ensure at least one agent because code doesn't work with empty tensors.
        # Returning immediately with (empty?) result would be more efficient but probably doesn't matter.
        if agent_active.float().sum() == 0:
            agent_active[0][0] = True
        flat_agent_active = agent_active.flatten()
        agent_group = torch.arange(0, batch_size).to(x.device).repeat_interleave(nagents)
        agent_index = torch.arange(0, batch_size * nagents).to(x.device)
        active_agent_groups = agent_group[flat_agent_active]
        active_agent_indices = agent_index[flat_agent_active]
        xagent = xagent[agent_active]
        agents, mask_agent = self.agent_embedding(xagent)

        origin = xagent[:, 0:2].clone()
        direction = xagent[:, 2:4].clone()

        pemb_list = []
        pmask_list = []
        emb_list = []
        relpos_list = []
        mask_list = []
        for item_net in self.item_nets:
            emb, mask = item_net(x)
            emb_list.append(emb[active_agent_groups])
            mask_list.append(mask[active_agent_groups])
            relpos = item_net.relpos(x, active_agent_groups, origin, direction)
            relpos_list.append(relpos)
            if item_net.start_privileged is not None:
                pemb, pmask = item_net(x, privileged=True)
                pemb_list.append(pemb)
                pmask_list.append(pmask)
            else:
                pemb_list.append(emb)
                pmask_list.append(mask)
        relpos = torch.cat(relpos_list, dim=1)
        relpos_embed, _ = self.relpos_net(relpos)
        embed = torch.cat(emb_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        # Ensure that at least one item is not masked out to prevent NaN in transformer softmax
        mask[:, 0] = 0

        items = torch.cat([relpos_embed, embed], dim=2)

        pitems = torch.cat(pemb_list, dim=1)
        pmask = torch.cat(pmask_list, dim=1)

        # TODO: constant item?
        """
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
                """

        if self.hps.spatial_attn:
            distances = relpos[:, :, 2].pow(2).view(-1, 1, self.nitem)
            spatial_attn = self.gattn(distances)
            modulators = [spatial_attn]
        else:
            modulators = None
        # Multihead attention input dimensions are: batch size, sequence length, embedding features
        target = agents.view(-1, 1, self.d_agent)
        x = self.transformer(target, items, mask, modulators)
        x = x.view(-1, self.d_agent)

        if self.hps.nearby_map:
            items = self.norm_map(F.relu(self.downscale(items)))
            items = items * (1 - mask.float().unsqueeze(-1))
            nearby_map = spatial.single_batch_dim_spatial_scatter(
                items=items[:, :(self.nitem - self.nconstant - self.ntile), :],
                positions=relpos[:, :self.nitem - self.nconstant - self.ntile, :2],
                nray=self.hps.nm_nrays,
                nring=self.hps.nm_nrings,
                inner_radius=self.hps.nm_ring_width,
                embed_offsets=self.hps.map_embed_offset,
            ).view(-1, self.map_channels, self.hps.nm_nrings, self.hps.nm_nrays)
            if self.hps.map_conv:
                nearby_map2 = self.conv2(F.relu(self.conv1(nearby_map)))
                nearby_map2 = nearby_map2.permute(0, 3, 2, 1)
                nearby_map = nearby_map.permute(0, 3, 2, 1)
                nearby_map = self.norm_conv(nearby_map + nearby_map2)
            nearby_map = nearby_map.reshape(-1, self.d_agent)
            x = torch.cat([x, nearby_map], dim=1)

        x = self.final_layer(x).squeeze(0)
        # TODO: remove
        x = x * (~mask_agent).float().unsqueeze(-1)
        return x, active_agent_indices, active_agent_groups, (pitems, pmask)


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
        self._stddev = None
        self._dirty = True

    def update(self, input, mask):
        self._dirty = True
        if mask is not None:
            if len(input.size()) == 3:
                batch_size, nitem, features = input.size()
                assert (batch_size, nitem) == mask.size()
            elif len(input.size()) == 2:
                batch_size, features = input.size()
                assert (batch_size,) == mask.size()
            else:
                raise Exception(f'Expecting 2 or 3 dimensions, actual: {len(input.size())}')
            #count = mask.float().sum().item()
            #mask = mask.view(-1).unsqueeze(-1).float()
            input = input[mask]
        else:
            features = input.size()[-1]
            input = input.reshape(-1, features)

        count = input.numel() / features
        if count == 0:
            return
        #mean = input.sum(dim=0) / count
        mean = input.mean(dim=0)
        if self.count == 0:
            self.count += count
            self.mean = mean
            self.squares_sum = ((input - mean) * (input - mean)).sum(dim=0)
            #if mask is not None:
            #    self.squares_sum = ((input - mean) * (input - mean) * mask).sum(dim=0)
            #else:
            #    self.squares_sum = ((input - mean) * (input - mean)).sum(dim=0)
        else:
            self.count += count
            new_mean = self.mean + (mean - self.mean) * count / self.count
            # This is probably not quite right because it applies multiple updates simultaneously.
            self.squares_sum = self.squares_sum + ((input - self.mean) * (input - new_mean)).sum(dim=0)
            #if mask is not None:
            #    self.squares_sum = self.squares_sum + ((input - self.mean) * (input - new_mean) * mask).sum(dim=0)
            #else:
            #    self.squares_sum = self.squares_sum + ((input - self.mean) * (input - new_mean)).sum(dim=0)
            self.mean = new_mean

    def forward(self, input, mask=None):
        with torch.no_grad():
            if self.training:
                self.update(input, mask=mask)
            if self.count > 1:
                input = (input - self.mean) / self.stddev()
            input = torch.clamp(input, -self.cliprange, self.cliprange)

        return input.half() if self.fp16 else input

    def enable_fp16(self):
        # Convert buffers back to fp32, fp16 has insufficient precision and runs into overflow on squares_sum
        self.float()
        self.fp16 = True

    def stddev(self):
        if self._dirty:
            sd = torch.sqrt(self.squares_sum / (self.count - 1))
            sd[sd == 0] = 1
            self._stddev = sd
            self._dirty = False
        return self._stddev


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


class PosItemBlock(nn.Module):
    def __init__(self,
                 d_in,
                 d_model,
                 d_ff,
                 norm_fn,
                 resblock,
                 mask_feature,
                 count,
                 start,
                 end,
                 start_privileged=None,
                 end_privileged=None):
        super(PosItemBlock, self).__init__()

        self.d_in = d_in
        self.embedding = InputEmbedding(d_in, d_model, norm_fn)
        self.mask_feature = mask_feature
        if resblock:
            self.resblock = FFResblock(d_model, d_ff, norm_fn)
        self.count = count
        self.start = start
        self.end = end
        self.start_privileged = start_privileged
        self.end_privileged = end_privileged

    def forward(self, x, privileged=False):
        if privileged:
            x = x[:, self.start_privileged:self.end_privileged].view(-1, self.count, self.d_in)
        else:
            x = x[:, self.start:self.end].view(-1, self.count, self.d_in)

        mask = x[:, :, self.mask_feature] == 0

        x = self.embedding(x, ~mask)
        if self.resblock is not None:
            x = self.resblock(x)
        x = x * (~mask).unsqueeze(-1).float()

        return x, mask

    def relpos(self, x, indices, origin, direction):
        batch_agents, _ = origin.size()
        x = x[:, self.start:self.end].view(-1, self.count, self.d_in)
        pos = x[indices, :, 0:2]
        relpos = spatial.unbatched_relative_positions(origin, direction, pos)
        dist = relpos.norm(p=2, dim=2)
        direction = relpos / (dist.unsqueeze(-1) + 1e-8)
        return torch.cat([direction, torch.sqrt(dist.unsqueeze(-1))], dim=2)


class ItemBlock(nn.Module):
    def __init__(self, d_in, d_model, d_ff, norm_fn, resblock, mask_feature=None):
        super(ItemBlock, self).__init__()

        self.embedding = InputEmbedding(d_in, d_model, norm_fn)
        self.mask_feature = mask_feature
        if resblock:
            self.resblock = FFResblock(d_model, d_ff, norm_fn)

    def forward(self, x):
        if self.mask_feature:
            if x.dim() == 3:
                mask = x[:, :, self.mask_feature] == 0
            else:
                mask = x[:, self.mask_feature] == 0
        else:
            if x.dim() == 3:
                mask = torch.ones_like(x[:, :, 0]).to(x.device) == 0
            else:
                mask = torch.ones_like(x[:, 0]).to(x.device) == 0

        x = self.embedding(x, ~mask)
        if self.resblock is not None:
            x = self.resblock(x)
        x = x * (~mask).unsqueeze(-1).float()
        return x, mask