import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

from gather import topk_by
from multihead_attention import MultiheadAttention
import spatial
from gym_codecraft.envs.codecraft_vec_env import DEFAULT_OBS_CONFIG, GLOBAL_FEATURES_V2, MSTRIDE_V2, DSTRIDE_V2


class PolicyTMem(nn.Module):
    def __init__(self,
                 d_agent,
                 d_item,
                 dff_ratio,
                 nhead,
                 dropout,
                 small_init_pi,
                 zero_init_vf,
                 fp16,
                 norm,
                 agents,
                 nally,
                 nenemy,
                 nmineral,
                 obs_config=DEFAULT_OBS_CONFIG,
                 use_privileged=False,
                 nearby_map=False,
                 ring_width=40,
                 nrays=8,
                 nrings=8,
                 map_conv=False,
                 map_conv_kernel_size=3,
                 map_embed_offset=False,
                 item_ff=True,
                 keep_abspos=False,
                 ally_enemy_same=False,
                 naction=8,
                 memory=True,
                 obs_last_action=False,
                 ):
        super(PolicyTMem, self).__init__()
        assert obs_config.drones > 0 or obs_config.minerals > 0,\
            'Must have at least one mineral or drones observation'
        assert obs_config.drones >= obs_config.allies
        assert not use_privileged or (nmineral > 0 and nally > 0 and (nenemy > 0 or ally_enemy_same))

        self.version = 'transformer_lstm'

        self.kwargs = dict(
            d_agent=d_agent,
            d_item=d_item,
            dff_ratio=dff_ratio,
            nhead=nhead,
            dropout=dropout,
            small_init_pi=small_init_pi,
            zero_init_vf=zero_init_vf,
            fp16=fp16,
            use_privileged=use_privileged,
            norm=norm,
            obs_config=obs_config,
            agents=agents,
            nally=nally,
            nenemy=nenemy,
            nmineral=nmineral,
            nearby_map=nearby_map,

            ring_width=ring_width,
            nrays=nrays,
            nrings=nrings,
            map_conv=map_conv,
            map_conv_kernel_size=map_conv_kernel_size,
            map_embed_offset=map_embed_offset,
            item_ff=item_ff,
            keep_abspos=keep_abspos,
            ally_enemy_same=ally_enemy_same,
            naction=naction,
            memory=memory,
            obs_last_action=obs_last_action,
        )

        self.obs_config = obs_config
        self.agents = agents
        self.nally = nally
        self.nenemy = nenemy
        self.nmineral = nmineral
        self.nitem = nally + nenemy + nmineral
        if hasattr(obs_config, 'global_drones'):
            self.global_drones = obs_config.global_drones
        else:
            self.global_drones = 0

        self.d_agent = d_agent
        self.d_item = d_item
        self.dff_ratio = dff_ratio
        self.nhead = nhead
        self.dropout = dropout
        self.nearby_map = nearby_map
        self.ring_width = ring_width
        self.nrays = nrays
        self.nrings = nrings
        self.map_conv = map_conv
        self.map_conv_kernel_size = map_conv_kernel_size
        self.map_embed_offset = map_embed_offset
        self.item_ff = item_ff
        self.naction = naction

        self.fp16 = fp16
        self.use_privileged = use_privileged
        self.ally_enemy_same = ally_enemy_same

        if norm == 'none':
            norm_fn = lambda x: nn.Sequential()
        elif norm == 'batchnorm':
            norm_fn = lambda n: nn.BatchNorm2d(n)
        elif norm == 'layernorm':
            norm_fn = lambda n: nn.LayerNorm(n)
        else:
            raise Exception(f'Unexpected normalization layer {norm}')

        self.agent_embedding = ItemBlock(
            DSTRIDE_V2 + GLOBAL_FEATURES_V2 + (8 if obs_last_action else 0),
            d_agent, d_agent * dff_ratio, norm_fn, True,
            keep_abspos=True,
            mask_feature=7,  # Feature 7 is hitpoints
            relpos=False,
        )
        if ally_enemy_same:
            self.drone_net = ItemBlock(
                DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                keep_abspos=keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=nally+nenemy,
            )
        else:
            self.ally_net = ItemBlock(
                DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                keep_abspos=keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=nally,
            )
            self.enemy_net = ItemBlock(
                DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                keep_abspos=keep_abspos,
                mask_feature=7,  # Feature 7 is hitpoints
                topk=nenemy,
            )
        self.mineral_net = ItemBlock(
            MSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
            keep_abspos=keep_abspos,
            mask_feature=2,  # Feature 2 is size
            topk=nmineral,
        )

        if use_privileged:
            self.pmineral_net = ItemBlock(
                MSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                keep_abspos=True, relpos=False, mask_feature=2,
            )
            if ally_enemy_same:
                self.pdrone_net = ItemBlock(
                    DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )
            else:
                self.pally_net = ItemBlock(
                    DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )
                self.penemy_net = ItemBlock(
                    DSTRIDE_V2, d_item, d_item * dff_ratio, norm_fn, self.item_ff,
                    keep_abspos=True, relpos=False, mask_feature=7,
                )

        self.multihead_attention = MultiheadAttention(
            embed_dim=d_agent,
            kdim=d_item,
            vdim=d_item,
            num_heads=nhead,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_agent, d_agent * dff_ratio)
        self.linear2 = nn.Linear(d_agent * dff_ratio, d_agent)
        self.norm1 = nn.LayerNorm(d_agent)
        self.norm2 = nn.LayerNorm(d_agent)

        self.map_channels = d_agent // (nrings * nrays)
        map_item_channels = self.map_channels - 2 if self.map_embed_offset else self.map_channels
        self.downscale = nn.Linear(d_item, map_item_channels)
        self.norm_map = norm_fn(map_item_channels)
        self.conv1 = spatial.ZeroPaddedCylindricalConv2d(
            self.map_channels, dff_ratio * self.map_channels, kernel_size=map_conv_kernel_size)
        self.conv2 = spatial.ZeroPaddedCylindricalConv2d(
            dff_ratio * self.map_channels, self.map_channels, kernel_size=map_conv_kernel_size)
        self.norm_conv = norm_fn(self.map_channels)

        self.final_width = d_agent
        if nearby_map:
            self.final_width += d_agent

        self.lstm = None
        self.final_layer = None
        if memory:
            self.lstm = nn.LSTM(input_size=self.final_width, hidden_size=d_agent)
            self.final_layer = nn.Sequential(
                nn.Linear(d_agent, d_agent),
                nn.ReLU(),
            )
        else:
            self.final_layer = nn.Sequential(
                nn.Linear(self.final_width, d_agent),
                nn.ReLU(),
            )

        self.policy_head = nn.Linear(d_agent, naction)
        if small_init_pi:
            self.policy_head.weight.data *= 0.01
            self.policy_head.bias.data.fill_(0.0)

        if self.use_privileged:
            self.value_head = nn.Linear(d_agent + 2 * d_item, 1)
        else:
            self.value_head = nn.Linear(d_agent, 1)
        if zero_init_vf:
            self.value_head.weight.data.fill_(0.0)
            self.value_head.bias.data.fill_(0.0)

        self.epsilon = 1e-4 if fp16 else 1e-8

    def evaluate(self, observation, action_masks, hidden_state):
        if self.fp16:
            action_masks = action_masks.half()
        action_masks = action_masks[:, :self.agents, :]
        probs, v, hidden_state = self.forward(observation, hidden_state)
        probs = probs.view(-1, self.agents, self.naction)
        probs = probs * action_masks + self.epsilon  # Add small value to prevent crash when no action is possible
        # We get device-side assert when using fp16 here (needs more investigation)
        action_dist = distributions.Categorical(probs.float() if self.fp16 else probs)
        actions = action_dist.sample()
        entropy = action_dist.entropy()[action_masks.sum(2) != 0]
        return actions,\
               action_dist.log_prob(actions).detach(),\
               entropy.detach(),\
               v.detach().view(-1),\
               probs.detach(),\
               hidden_state

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
                 split_reward,
                 hidden_state):
        if self.fp16:
            advantages = advantages.half()
            returns = returns.half()
            action_masks = action_masks.half()
            old_logprobs = old_logprobs.half()

        x, (pitems, pmask), _ = self.latents(obs, hidden_state)
        batch_size = x.size()[0]

        vin = x.max(dim=1).values.view(batch_size, self.d_agent)
        if self.use_privileged:
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
        actions = actions.view(-1, self.agents)
        logprobs = dist.log_prob(actions).view(-1)
        assert logprobs.size() == old_logprobs.size()
        ratios = torch.exp(logprobs - old_logprobs)
        if split_reward:
            advantages = advantages / active_agents.view(-1)
        advantages = advantages.view(-1, 1).repeat(1, self.agents).view(-1)
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

    def forward(self, x, hidden_state):
        batch_size = x.size()[0]
        x, (pitems, pmask), hidden_state = self.latents(x, hidden_state)

        vin = x.max(dim=1).values.view(batch_size, self.d_agent)
        if self.use_privileged:
            pitems_max = pitems.max(dim=1).values
            pitems_avg = pitems.sum(dim=1) / torch.clamp_min((~pmask).float().sum(dim=1), min=1).unsqueeze(-1)
            vin = torch.cat([vin, pitems_max, pitems_avg], dim=1)
        values = self.value_head(vin).view(-1)

        logits = self.policy_head(x)
        probs = F.softmax(logits, dim=2)

        # return probs.view(batch_size, 8, self.allies).permute(0, 2, 1), values
        return probs, values, hidden_state

    def logits(self, x, hidden_state):
        x, _, _ = self.latents(x, hidden_state)
        return self.policy_head(x)

    def latents(self, x, hidden_state):
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        #x.fill_(0.0)
        timesteps, sequences, _ = x.size()
        batch_size = timesteps * sequences
        #print(x.size())
        # print('INPUT', x[:, :, 0:5])

        ASTRIDE = DSTRIDE_V2 + (8 if self.obs_config.obs_last_action else 0)
        obs_config_enemies = self.obs_config.drones - self.obs_config.allies
        endglobals = GLOBAL_FEATURES_V2
        endallies = GLOBAL_FEATURES_V2 + ASTRIDE * self.obs_config.allies
        endenemies = endallies + DSTRIDE_V2 * obs_config_enemies
        endmins = endenemies + MSTRIDE_V2 * self.obs_config.minerals
        endallenemies = endmins + DSTRIDE_V2 * (self.obs_config.drones - self.obs_config.allies)

        globals = x[:, :, :endglobals]

        # properties of the drone controlled by this network
        xagent = x[:, :, endglobals:endallies].reshape(batch_size, self.obs_config.allies, ASTRIDE)[:, :self.agents, :]
        globals = globals.reshape(batch_size, 1, GLOBAL_FEATURES_V2) \
            .expand(batch_size, self.agents, GLOBAL_FEATURES_V2)
        xagent = torch.cat([xagent, globals], dim=2)
        agents, _, mask_agent = self.agent_embedding(xagent)
        # print('AGENTS', agents.view(timesteps, sequences, self.obs_config.allies, self.d_agent)[:, :, :, 0:5])

        origin = xagent[:, :, 0:2].clone()
        direction = xagent[:, :, 2:4].clone()

        xally = x[:, :, endglobals:endallies].reshape(batch_size, self.obs_config.allies, ASTRIDE)[:, :, :DSTRIDE_V2]
        eobs = self.obs_config.drones - self.obs_config.allies
        if self.ally_enemy_same:
            xe = x[:, :, endallies:endenemies].reshape(batch_size, eobs, DSTRIDE_V2)
            xdrone = torch.cat([xally, xe], dim=1)
            items, relpos, mask = self.drone_net(xdrone, origin, direction)
        else:
            items, relpos, mask = self.ally_net(xally, origin, direction)
        # Ensure that at least one item is not masked out to prevent NaN in transformer softmax
        mask[:, :, 0] = 0

        if self.nenemy > 0 and not self.ally_enemy_same:
            xe = x[:, :, endallies:endenemies].reshape(batch_size, eobs, DSTRIDE_V2)

            items_e, relpos_e, mask_e = self.enemy_net(xe, origin, direction)
            items = torch.cat([items, items_e], dim=2)
            mask = torch.cat([mask, mask_e], dim=2)
            relpos = torch.cat([relpos, relpos_e], dim=2)

        if self.nmineral > 0:
            xm = x[:, :, endenemies:endmins].reshape(batch_size, self.obs_config.minerals, MSTRIDE_V2)

            items_m, relpos_m, mask_m = self.mineral_net(xm, origin, direction)
            items = torch.cat([items, items_m], dim=2)
            mask = torch.cat([mask, mask_m], dim=2)
            relpos = torch.cat([relpos, relpos_m], dim=2)

        if self.use_privileged:
            eobs = self.obs_config.drones - self.obs_config.allies
            xenemy = x[:, :, endmins:endallenemies].reshape(batch_size, eobs, DSTRIDE_V2)
            if self.ally_enemy_same:
                xdrone = torch.cat([xally, xenemy], dim=1)
                pitems, _, pmask = self.pdrone_net(xdrone)
            else:
                pitems, _, pmask = self.pally_net(xally)
                pitems_e, _, pmask_e = self.penemy_net(xenemy)
                pitems = torch.cat([pitems, pitems_e], dim=1)
                pmask = torch.cat([pmask, pmask_e], dim=1)
            xm = x[:, :, endenemies:endmins].reshape(batch_size, self.obs_config.minerals, MSTRIDE_V2)
            pitems_m, _, pmask_m = self.pmineral_net(xm)
            pitems = torch.cat([pitems, pitems_m], dim=1)
            pmask = torch.cat([pmask, pmask_m], dim=1)
        else:
            pitems = None
            pmask = None

        # Transformer input dimensions are: Sequence length, Batch size, Embedding size
        source = items.view(batch_size * self.agents, self.nitem, self.d_item).permute(1, 0, 2)
        target = agents.view(1, batch_size * self.agents, self.d_agent)
        x, attn_weights = self.multihead_attention(
            query=target,
            key=source,
            value=source,
            key_padding_mask=mask.view(batch_size * self.agents, self.nitem),
        )
        x = self.norm1(x + target)
        x2 = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + x2)
        x = x.view(batch_size, self.agents, self.d_agent)

        if self.nearby_map:
            items = self.norm_map(F.relu(self.downscale(items)))
            items = items * (1 - mask.float().unsqueeze(-1))
            nearby_map = spatial.spatial_scatter(
                items=items,
                positions=relpos,
                nray=self.nrays,
                nring=self.nrings,
                inner_radius=self.ring_width,
                embed_offsets=self.map_embed_offset,
            ).view(batch_size * self.agents, self.map_channels, self.nrings, self.nrays)
            if self.map_conv:
                nearby_map2 = self.conv2(F.relu(self.conv1(nearby_map)))
                nearby_map2 = nearby_map2.permute(0, 3, 2, 1)
                nearby_map = nearby_map.permute(0, 3, 2, 1)
                nearby_map = self.norm_conv(nearby_map + nearby_map2)
            nearby_map = nearby_map.reshape(batch_size, self.agents, self.d_agent)
            x = torch.cat([x, nearby_map], dim=2)

        if self.lstm:
            x = x.view(timesteps, sequences * self.agents, self.final_width)
            hidden_state_mask = (~mask_agent).float().view(timesteps, sequences * self.agents)
            x2 = []
            for t in range(timesteps):
                # TODO: need stable slot assignment and way to detect when drone is replaced
                # print(f'HIDDEN{t}', hidden_state[0][0, 0, 0].item(), hidden_state[1][0, 0, 0].item())
                out, (hidden, cell) = self.lstm(x[t].unsqueeze(0), hidden_state)
                hidden = hidden * hidden_state_mask[t].view(1, -1, 1)
                cell = cell * hidden_state_mask[t].view(1, -1, 1)
                hidden_state = (hidden, cell)
                # print(f'HIDDEN{t+1}', hidden_state[0][0, :, 0:3], hidden_state[1][0, :, 0:3])
                x2.append(hidden)
                #print(t, hidden_state[0][0, 0, 0:2], out[0, 0, 0:2])
            if len(x2) == 1:
                x = x2[0]
            else:
                x = torch.cat(x2, dim=0)
        x = self.final_layer(x)
        x = x.view(batch_size, self.agents, self.d_agent)
        #x = x * (~mask_agent).float().unsqueeze(-1)

        return x, (pitems, pmask), hidden_state

    def param_groups(self):
        # TODO?
        pass

    def initial_hidden_state(self, dbatch, device):
        if self.lstm is None:
            return None
        else:
            return (
                torch.zeros((1, dbatch * self.agents, self.d_agent)).to(device),
                torch.zeros((1, dbatch * self.agents, self.d_agent)).to(device),
            )


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
            if len(input.size()) == 4:
                batch_size, nally, nitem, features = input.size()
                assert (batch_size, nally, nitem) == mask.size()
            elif len(input.size()) == 3:
                batch_size, nally, features = input.size()
                assert (batch_size, nally) == mask.size()
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
            if self.training and False:
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

    def forward(self, x, origin=None, direction=None):
        batch_size, items, features = x.size()

        if origin is not None:
            _, agents, _ = origin.size()

            pos = x[:, :, 0:2]
            relpos = spatial.relative_positions(origin, direction, pos)
            dist = relpos.norm(p=2, dim=3)
            direction = relpos / (dist.unsqueeze(-1) + 1e-8)

            x = x.view(batch_size, 1, items, features)\
                .expand(batch_size, agents, items, features)
            if self.keep_abspos:
                x = torch.cat([x, direction, torch.sqrt(dist.unsqueeze(-1))], dim=3)
            else:
                x = torch.cat([direction, x[:, :, :, 2:], torch.sqrt(dist.unsqueeze(-1))], dim=3)

            if self.topk is not None:
                empty = (x[:, :, :, self.mask_feature] == 0).float()
                key = -dist - empty * 1e8
                x = topk_by(values=x, vdim=2, keys=key, kdim=2, k=self.topk)
                relpos = topk_by(values=relpos, vdim=2, keys=key, kdim=2, k=self.topk)

            mask = x[:, :, :, self.mask_feature] == 0
        else:
            relpos = None
            mask = x[:, :, self.mask_feature] == 0

        x = self.embedding(x, ~mask)
        if self.resblock is not None:
            x = self.resblock(x)
        x = x * (~mask).unsqueeze(-1).float()

        return x, relpos, mask

