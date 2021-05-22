import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch_scatter import scatter_add, scatter_max

from hyperstate import PolicyConfig, ObsConfig, Config

import spatial


class TransformerPolicy8HS(nn.Module):
    def __init__(self, config: PolicyConfig, obs_config: ObsConfig, naction: int):
        super(TransformerPolicy8HS, self).__init__()
        assert (
            obs_config.drones > 0 or obs_config.minerals > 0
        ), "Must have at least one mineral or drones observation"
        assert obs_config.drones >= obs_config.allies
        assert not obs_config.use_privileged or (
            config.nmineral > 0
            and config.nally > 0
            and (config.nenemy > 0 or config.ally_enemy_same)
        )

        assert config.nally == obs_config.allies
        assert config.nenemy == obs_config.drones - obs_config.allies
        assert config.nmineral == obs_config.obs_minerals
        assert config.ntile == obs_config.obs_map_tiles
        assert config.nconstant == 0

        self.version = "transformer_v8"

        self.hps = config
        self.obs_config = obs_config
        self.agents = config.agents
        self.nally = config.nally
        self.nenemy = config.nenemy
        self.nmineral = config.nmineral
        self.nconstant = config.nconstant
        self.ntile = config.ntile
        self.nitem = config.nally + config.nenemy + config.nmineral + config.ntile
        self.d_agent = config.d_agent
        self.d_item = config.d_item
        self.naction = naction

        if hasattr(obs_config, "global_drones"):
            self.global_drones = obs_config.global_drones
        else:
            self.global_drones = 0

        if config.norm == "none":
            norm_fn = lambda x: nn.Sequential()
        elif config.norm == "batchnorm":
            norm_fn = lambda n: nn.BatchNorm2d(n)
        elif config.norm == "layernorm":
            norm_fn = lambda n: nn.LayerNorm(n)
        else:
            raise Exception(f"Unexpected normalization layer {config.norm}")

        endglobals = self.obs_config.endglobals()
        endallies = self.obs_config.endallies()
        endenemies = self.obs_config.endenemies()
        endmins = self.obs_config.endmins()
        endtiles = self.obs_config.endtiles()
        endallenemies = self.obs_config.endallenemies()

        self.agent_embedding = ItemBlock(
            obs_config.dstride() + obs_config.global_features(),
            config.d_agent,
            config.d_agent * config.dff_ratio,
            norm_fn,
            True,
        )
        self.relpos_net = ItemBlock(
            3,
            config.d_item // 2,
            config.d_item // 2 * config.dff_ratio,
            norm_fn,
            config.item_ff,
        )

        self.item_nets = nn.ModuleList()
        if config.ally_enemy_same:
            self.item_nets.append(
                PosItemBlock(
                    obs_config.dstride(),
                    config.d_item // 2,
                    config.d_item // 2 * config.dff_ratio,
                    norm_fn,
                    config.item_ff,
                    mask_feature=7,  # Feature 7 is hitpoints
                    count=obs_config.drones,
                    start=endglobals,
                    end=endenemies,
                )
            )
        else:
            if self.nally > 0:
                self.item_nets.append(
                    PosItemBlock(
                        obs_config.dstride(),
                        config.d_item // 2,
                        config.d_item // 2 * config.dff_ratio,
                        norm_fn,
                        config.item_ff,
                        mask_feature=7,  # Feature 7 is hitpoints
                        count=obs_config.allies,
                        start=endglobals,
                        end=endallies,
                    )
                )
            if self.nenemy > 0:
                self.item_nets.append(
                    PosItemBlock(
                        obs_config.dstride(),
                        config.d_item // 2,
                        config.d_item // 2 * config.dff_ratio,
                        norm_fn,
                        config.item_ff,
                        mask_feature=7,  # Feature 7 is hitpoints
                        count=obs_config.drones - self.obs_config.allies,
                        start=endallies,
                        end=endenemies,
                        start_privileged=endtiles
                        if obs_config.use_privileged
                        else None,
                        end_privileged=endallenemies
                        if obs_config.use_privileged
                        else None,
                    )
                )
        if config.nmineral > 0:
            self.item_nets.append(
                PosItemBlock(
                    obs_config.mstride(),
                    config.d_item // 2,
                    config.d_item // 2 * config.dff_ratio,
                    norm_fn,
                    config.item_ff,
                    mask_feature=2,  # Feature 2 is size
                    count=obs_config.obs_minerals,
                    start=endenemies,
                    end=endmins,
                )
            )
        if config.ntile > 0:
            self.item_nets.append(
                PosItemBlock(
                    obs_config.tstride(),
                    config.d_item // 2,
                    config.d_item // 2 * config.dff_ratio,
                    norm_fn,
                    config.item_ff,
                    mask_feature=2,  # Feature is elapsed since last visited time
                    count=obs_config.obs_map_tiles,
                    start=endmins,
                    end=endtiles,
                )
            )
        if config.nconstant > 0:
            self.constant_items = nn.Parameter(
                torch.normal(0, 1, (config.nconstant, config.d_item))
            )

        if config.item_item_attn_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_item, nhead=8)
            self.item_item_attn = nn.TransformerEncoder(
                encoder_layer, num_layers=config.item_item_attn_layers
            )
        else:
            self.item_item_attn = None

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config.d_agent,
            kdim=config.d_item,
            vdim=config.d_item,
            num_heads=config.nhead,
            dropout=config.dropout,
        )
        self.linear1 = nn.Linear(config.d_agent, config.d_agent * config.dff_ratio)
        self.linear2 = nn.Linear(config.d_agent * config.dff_ratio, config.d_agent)
        self.norm1 = nn.LayerNorm(config.d_agent)
        self.norm2 = nn.LayerNorm(config.d_agent)

        self.map_channels = config.d_agent // (config.nm_nrings * config.nm_nrays)
        map_item_channels = (
            self.map_channels - 2 if self.hps.map_embed_offset else self.map_channels
        )
        self.downscale = nn.Linear(config.d_item, map_item_channels)
        self.norm_map = norm_fn(map_item_channels)
        self.conv1 = spatial.ZeroPaddedCylindricalConv2d(
            self.map_channels, config.dff_ratio * self.map_channels, kernel_size=3
        )
        self.conv2 = spatial.ZeroPaddedCylindricalConv2d(
            config.dff_ratio * self.map_channels, self.map_channels, kernel_size=3
        )
        self.norm_conv = norm_fn(self.map_channels)

        final_width = config.d_agent
        if config.nearby_map:
            final_width += config.d_agent
        self.final_layer = nn.Sequential(
            nn.Linear(final_width, config.d_agent * config.dff_ratio), nn.ReLU(),
        )

        self.policy_head = nn.Linear(config.d_agent * config.dff_ratio, self.naction)
        if config.small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        if obs_config.use_privileged:
            self.value_head = nn.Linear(
                config.d_agent * config.dff_ratio + config.d_item, 1
            )
        else:
            self.value_head = nn.Linear(config.d_agent * config.dff_ratio, 1)
        if config.zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

        self.epsilon = 1e-8

    def evaluate(self, observation, action_masks, privileged_obs):
        action_masks = action_masks[:, : self.agents, :]
        probs, v = self.forward(observation, privileged_obs, action_masks)
        probs = probs.view(-1, self.agents, self.naction)
        if action_masks.size(2) != self.naction:
            nbatch, nagent, naction = action_masks.size()
            zeros = torch.zeros(nbatch, nagent, self.naction - naction).to(
                observation.device
            )
            action_masks = torch.cat([action_masks, zeros], dim=2)
        # Add small value to prevent crash when no action is possible
        probs = probs * action_masks + self.epsilon
        action_dist = distributions.Categorical(probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()[action_masks.sum(2) > 1]
        return (
            actions,
            action_dist.log_prob(actions),
            entropy,
            v.detach().view(-1).cpu().numpy(),
            probs.detach().cpu().numpy(),
        )

    def backprop(
        self,
        hps: Config,
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
        split_reward,
    ):

        action_masks = action_masks[:, : self.agents, :]
        actions = actions[:, : self.agents]
        old_logprobs = old_logprobs[:, : self.agents]

        probs, values = self.forward(obs, privileged_obs, action_masks)
        probs = probs.view(-1, self.agents, self.naction)

        # add small value to prevent degenerate probability distribution when no action is possible
        # gradients still get blocked by the action mask
        probs = probs * action_masks + self.epsilon

        agent_masks = (action_masks.sum(dim=2) > 1).float()
        active_agents = torch.clamp_min(agent_masks.sum(dim=1), min=1)

        dist = distributions.Categorical(probs)
        entropy = dist.entropy()
        logprobs = dist.log_prob(actions)
        ratios = torch.exp(logprobs - old_logprobs)
        advantages = advantages.view(-1, 1)
        if split_reward:
            advantages = advantages / active_agents.view(-1, 1)
        vanilla_policy_loss = advantages * ratios * agent_masks
        clipped_policy_loss = (
            advantages
            * torch.clamp(ratios, 1 - hps.ppo.cliprange, 1 + hps.ppo.cliprange)
            * agent_masks
        )
        if hps.ppo.ppo:
            policy_loss = (
                -torch.min(vanilla_policy_loss, clipped_policy_loss).mean(dim=0).sum()
            )
        else:
            policy_loss = -vanilla_policy_loss.mean(dim=0).sum()

        approxkl = (old_probs * torch.log(old_probs / probs)).sum(dim=2).mean()
        clipfrac = ((ratios - 1.0).abs() > hps.ppo.cliprange).sum().type(
            torch.float32
        ) / ratios.numel()

        clipped_values = old_values + torch.clamp(
            values - old_values, -hps.ppo.cliprange, hps.ppo.cliprange
        )
        vanilla_value_loss = (values - returns) ** 2
        clipped_value_loss = (clipped_values - returns) ** 2
        if hps.ppo.clip_vf:
            value_loss = torch.max(vanilla_value_loss, clipped_value_loss).mean()
        else:
            value_loss = vanilla_value_loss.mean()

        entropy_loss = -hps.optimizer.entropy_bonus * entropy.mean()

        loss = policy_loss + value_loss_scale * value_loss + entropy_loss
        loss /= hps.optimizer.batches_per_update
        loss.backward()
        return (
            policy_loss.data.tolist(),
            value_loss.data.tolist(),
            -entropy_loss.data.tolist(),
            approxkl.data.tolist(),
            clipfrac.data.tolist(),
        )

    def forward(self, x, x_privileged, action_masks):
        batch_size = x.size()[0]
        x, active_agents, (pitems, pmask) = self.latents(x, action_masks)

        if x.is_cuda:
            vin = torch.cuda.FloatTensor(
                batch_size, self.d_agent * self.hps.dff_ratio
            ).fill_(0)
        else:
            vin = torch.zeros(batch_size, self.d_agent * self.hps.dff_ratio)
        scatter_max(x, index=active_agents.batch_index, dim=0, out=vin)
        if self.obs_config.use_privileged:
            mask1k = 1000.0 * pmask.float().unsqueeze(-1)
            pitems_max = (pitems - mask1k).max(dim=1).values
            pitems_max[pitems_max == -1000.0] = 0.0
            pitems_avg = pitems.sum(dim=1) / torch.clamp_min(
                (~pmask).float().sum(dim=1), min=1
            ).unsqueeze(-1)
            vin = torch.cat([vin, pitems_max, pitems_avg], dim=1)
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)
        logits = logits.masked_fill(
            action_masks.reshape(-1, self.naction)[active_agents.flat_index] == 0,
            float("-inf"),
        )
        probs = F.softmax(logits, dim=1)
        probs = active_agents.pad(probs)
        return probs, values

    def latents(self, x, action_masks):
        batch_size = x.size()[0]

        endglobals = self.obs_config.endglobals()
        endallies = self.obs_config.endallies()

        globals = x[:, :endglobals]

        # properties of the drone controlled by this network
        xagent = x[:, endglobals:endallies].view(
            batch_size, self.obs_config.allies, self.obs_config.dstride()
        )[:, : self.agents, :]
        globals = globals.view(batch_size, 1, self.obs_config.global_features()).expand(
            batch_size, self.agents, self.obs_config.global_features()
        )
        xagent = torch.cat([xagent, globals], dim=2)

        agent_active = action_masks.sum(2) > 0
        # Ensure at least one agent is selected because code doesn't work with empty tensors.
        if agent_active.float().sum() == 0:
            agent_active[0][0] = True
            action_masks[0][0] = 1.0
        active_agents = SparseSequence.from_mask(agent_active)
        xagent = xagent[agent_active]
        agents = self.agent_embedding(xagent)

        origin = xagent[:, 0:2].clone()
        direction = xagent[:, 2:4].clone()

        pemb_list = []
        pmask_list = []
        emb_list = []
        relpos_list = []
        sparse_relpos_list = []
        relpos_sparsity_list = []
        mask_list = []
        for item_net in self.item_nets:
            emb, mask = item_net(x)
            emb_list.append(emb[active_agents.batch_index])
            mask_list.append(mask[active_agents.batch_index])

            relpos, sparse_relpos, relpos_sparsity = item_net.relpos(
                x, active_agents.batch_index, origin, direction
            )
            relpos_list.append(relpos)
            sparse_relpos_list.append(sparse_relpos)
            relpos_sparsity_list.append(relpos_sparsity)

            if item_net.start_privileged is not None:
                pemb, pmask = item_net(x, privileged=True)
                pemb_list.append(pemb)
                pmask_list.append(pmask)
            else:
                pemb_list.append(emb)
                pmask_list.append(mask)

        relpos = torch.cat(relpos_list, dim=1)
        sparse_relpos = torch.cat(sparse_relpos_list, dim=0)
        sparse_relpos_embed = self.relpos_net(sparse_relpos)
        relpos_embed_list = []
        offset = 0
        for sparsity in relpos_sparsity_list:
            if sparsity.sparse_count > 0:
                relpos_embed_list.append(
                    sparsity.pad(
                        sparse_relpos_embed[offset : offset + sparsity.sparse_count]
                    )
                )
                offset += sparsity.sparse_count
            else:
                relpos_embed_list.append(
                    torch.zeros(
                        active_agents.sparse_count,
                        sparsity.dseq,
                        self.d_item // 2,
                        device=relpos.device,
                    )
                )
        relpos_embed = torch.cat(relpos_embed_list, dim=1)

        embed = torch.cat(emb_list, dim=1)
        mask = torch.cat(mask_list, dim=1)
        # Ensure that at least one item is not masked out to prevent NaN in transformer softmax
        mask[:, 0] = 0

        items = torch.cat([relpos_embed, embed], dim=2)

        pitems = torch.cat(pemb_list, dim=1)
        pmask = torch.cat(pmask_list, dim=1)

        # Transformer input dimensions are: Sequence length, Batch size, Embedding size
        source = items.permute(1, 0, 2)
        target = agents.view(1, -1, self.d_agent)
        x, attn_weights = self.multihead_attention(
            query=target, key=source, value=source, key_padding_mask=mask,
        )
        x = self.norm1(x + target)
        x2 = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        x = x.view(-1, self.d_agent)

        if self.hps.nearby_map:
            items = self.norm_map(F.relu(self.downscale(items)))
            items = items * (1 - mask.float().unsqueeze(-1))
            nearby_map = spatial.single_batch_dim_spatial_scatter(
                items=items[:, : (self.nitem - self.nconstant - self.ntile), :],
                positions=relpos[:, : self.nitem - self.nconstant - self.ntile, :2],
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
        return x, active_agents, (pitems, pmask)


# Computes a running mean/variance of input features and performs normalization.
# https://www.johndcook.com/blog/standard_deviation/
class InputNorm(nn.Module):
    def __init__(self, num_features, cliprange=5):
        super(InputNorm, self).__init__()

        self.cliprange = cliprange
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("squares_sum", torch.zeros(num_features))
        self._stddev = None
        self._dirty = True

    def update(self, input):
        self._dirty = True
        dbatch, dfeat = input.size()

        count = input.numel() / dfeat
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
            self.squares_sum = self.squares_sum + (
                (input - self.mean) * (input - new_mean)
            ).sum(dim=0)
            self.mean = new_mean

    def forward(self, input, mask=None):
        with torch.no_grad():
            if self.training:
                self.update(input)
            if self.count > 1:
                input = (input - self.mean) / self.stddev()
            input = torch.clamp(input, -self.cliprange, self.cliprange)

        return input

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

    def forward(self, x):
        x = self.normalize(x)
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
        x2 = self.linear_2(F.relu(self.linear_1(x)))
        x = self.norm(x + x2)
        return x


class PosItemBlock(nn.Module):
    def __init__(
        self,
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
        end_privileged=None,
    ):
        super(PosItemBlock, self).__init__()

        self.d_in = d_in
        self.d_model = d_model
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
            x = x[:, self.start_privileged : self.end_privileged].view(
                -1, self.count, self.d_in
            )
        else:
            x = x[:, self.start : self.end].view(-1, self.count, self.d_in)

        select = x[:, :, self.mask_feature] != 0

        active = SparseSequence.from_mask(select)
        x_sparse = x[select]
        mask = select == False

        if x_sparse.numel() > 0:
            x_sparse = self.embedding(x_sparse)
            if self.resblock is not None:
                x_sparse = self.resblock(x_sparse)
            return active.pad(x_sparse), mask
        elif x.is_cuda:
            return (
                torch.cuda.FloatTensor(x.size()[0], x.size()[1], self.d_model).fill_(0),
                mask,
            )
        else:
            return torch.zeros(x.size()[0], x.size()[1], self.d_model), mask

    def relpos(self, x, indices, origin, direction):
        batch_agents, _ = origin.size()
        x = x[:, self.start : self.end].view(-1, self.count, self.d_in)
        mask = (x[:, :, self.mask_feature] != 0)[indices]
        pos = x[indices, :, 0:2]
        relpos = spatial.unbatched_relative_positions(origin, direction, pos)
        dist = relpos.norm(p=2, dim=2)
        direction = relpos / (dist.unsqueeze(-1) + 1e-8)
        x = torch.cat([direction, torch.sqrt(dist.unsqueeze(-1))], dim=2)
        sparse_x = x[mask]
        return x, sparse_x, SparseSequence.from_mask(mask)


class ItemBlock(nn.Module):
    def __init__(self, d_in, d_model, d_ff, norm_fn, resblock):
        super(ItemBlock, self).__init__()

        self.embedding = InputEmbedding(d_in, d_model, norm_fn)
        if resblock:
            self.resblock = FFResblock(d_model, d_ff, norm_fn)

    def forward(self, x):
        x = self.embedding(x)
        if self.resblock is not None:
            x = self.resblock(x)
        return x


ARANGE_CACHED = None
ARANGE_MAX = 0


def arange(count, device):
    global ARANGE_CACHED, ARANGE_MAX
    if count > ARANGE_MAX or ARANGE_CACHED is None:
        ARANGE_CACHED = torch.arange(0, count, device=device)
        ARANGE_MAX = count
    return ARANGE_CACHED[:count]


class SparseSequence:
    def __init__(self, dbatch: int, dseq: int, select: torch.ByteTensor):
        self.dbatch = dbatch
        self.dseq = dseq
        self.count = dbatch * dseq
        self.select = select
        self.flat_select = select.flatten()
        self.sparse_count = select.sum().item()
        self.device = select.device

        self._batch_index = None
        self._seq_index = None
        self._flat_index = None

    @property
    def batch_index(self):
        if self._batch_index is None:
            self._batch_index = arange(self.dbatch, self.device).repeat_interleave(
                self.dseq
            )[self.flat_select]
        return self._batch_index

    @property
    def seq_index(self):
        if self._seq_index is None:
            self._seq_index = arange(self.dseq, self.device).repeat(self.dbatch)[
                self.flat_select
            ]
        return self._seq_index

    @property
    def flat_index(self):
        if self._flat_index is None:
            self._flat_index = arange(self.dseq * self.dbatch, self.device)[
                self.flat_select
            ]
        return self._flat_index

    def pad(self, x: torch.Tensor):
        count, dfeat = x.size()
        assert count == self.sparse_count
        if x.is_cuda:
            x_padded = torch.cuda.FloatTensor(self.count, dfeat).fill_(0)
        else:
            x_padded = torch.zeros(self.count, dfeat)
        scatter_add(x, index=self.flat_index, dim=0, out=x_padded)
        return x_padded.view(self.dbatch, self.dseq, dfeat)

    @staticmethod
    def from_mask(select: torch.ByteTensor) -> "SparseSequence":
        dbatch, dseq = select.size()
        return SparseSequence(dbatch, dseq, select)
