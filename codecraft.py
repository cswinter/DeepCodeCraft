import requests
import logging
import time

import orjson
import numpy as np

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


RETRIES = 100


@dataclass
class ObsConfig:
    allies: int
    drones: int
    minerals: int
    tiles: int
    num_builds: int
    global_drones: int = 0
    relative_positions: bool = True
    feat_last_seen: bool = False
    feat_map_size: bool = False
    feat_is_visible: bool = False
    feat_abstime: bool = False
    v2: bool = False
    feat_rule_msdm: bool = False
    feat_rule_costs: bool = False
    feat_mineral_claims: bool = False
    harvest_action: bool = False
    lock_build_action: bool = False
    feat_dist_to_wall: bool = False
    unit_count: bool = False

    def global_features(self):
        gf = 2
        if self.feat_map_size:
            gf += 2
        if self.feat_abstime:
            gf += 2
        if self.feat_rule_msdm:
            gf += 1
        if self.feat_rule_costs:
            gf += self.num_builds
        if self.unit_count:
            gf += 1
        return gf

    def dstride(self):
        ds = 15
        if self.feat_last_seen:
            ds += 2
        if self.feat_is_visible:
            ds += 1
        if self.lock_build_action:
            ds += 1
        if self.feat_dist_to_wall:
            ds += 5
        return ds

    def mstride(self):
        return 4 if self.feat_mineral_claims else 3

    def tstride(self):
        return 4

    def nonobs_features(self):
        return 5

    def enemies(self):
        return self.drones - self.allies

    def total_drones(self):
        return 2 * self.drones - self.allies

    def stride(self):
        return self.global_features() \
               + self.total_drones() * self.dstride() \
               + self.minerals * self.mstride() \
               + self.tiles * self.tstride()

    def endglobals(self):
        return self.global_features()

    def endallies(self):
        return self.global_features() + self.dstride() * self.allies

    def endenemies(self):
        return self.global_features() + self.dstride() * self.drones

    def endmins(self):
        return self.endenemies() + self.mstride() * self.minerals

    def endtiles(self):
        return self.endmins() + self.tstride() * self.tiles

    def endallenemies(self):
        return self.endtiles() + self.dstride() * self.enemies()

    def extra_actions(self):
        if self.lock_build_action:
            return 2
        else:
            return 0


@dataclass
class Rules:
    mothership_damage_multiplier: float
    cost_modifiers: Dict[Tuple[int, int, int, int, int], float]


def create_game(game_length: int = None,
                action_delay: int = 0,
                self_play: bool = False,
                custom_map=None,
                scripted_opponent: str = 'none',
                rules=None,
                allowHarvesting: bool = True,
                forceHarvesting: bool = True,
                randomizeIdle: bool = True) -> int:
    assert rules is not None
    json = {
        'map': [] if custom_map is None else [custom_map],
        'costModifiers': list(rules.cost_modifiers.items()),
    }
    try:
        if game_length:
            response = requests.post(f'http://localhost:9000/start-game'
                                     f'?maxTicks={game_length}'
                                     f'&actionDelay={action_delay}'
                                     f'&scriptedOpponent={scripted_opponent}'
                                     f'&mothershipDamageMultiplier={rules.mothership_damage_multiplier}'
                                     f'&allowHarvesting={scalabool(allowHarvesting)}'
                                     f'&forceHarvesting={scalabool(forceHarvesting)}'
                                     f'&randomizeIdle={scalabool(randomizeIdle)}' ,
                                     json=json).json()
        else:
            response = requests.post(f'http://localhost:9000/start-game?actionDelay={action_delay}').json()
        return int(response['id'])
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on create_game, retrying")
        time.sleep(1)
        return create_game(game_length, action_delay, self_play)


def act(game_id: int, action):
    retries = 100
    while retries > 0:
        try:
            requests.post(f'http://localhost:9000/act?gameID={game_id}&playerID=0', json=action).raise_for_status()
            return
        except requests.exceptions.ConnectionError:
            # For some reason, a small percentage of requests fails with
            # "connection error (errno 98, address already in use)"
            # Just retry
            retries -= 1
            logging.info(f"Connection error on act({game_id}), retrying")
            time.sleep(1)


def act_batch(actions):
    payload = {}
    for game_id, player_id, player_actions in actions:
        player_actions_json = []
        for move, turn, buildSpec, harvest, lockBuildAction, unlockBuildAction in player_actions:
            player_actions_json.append({
                "buildDrone": buildSpec,
                "move": move,
                "harvest": harvest,
                "transfer": False,
                "turn": turn,
                "lockBuildAction": lockBuildAction,
                "unlockBuildAction": unlockBuildAction
            })
        payload[f'{game_id}.{player_id}'] = player_actions_json

    retries = 100
    while retries > 0:
        try:
            requests.post(
                f'http://localhost:9000/batch-act',
                data=orjson.dumps(payload),
                headers={'Content-Type': 'application/json'},
            ).raise_for_status()
            return
        except requests.exceptions.ConnectionError:
            # For some reason, a small percentage of requests fails with
            # "connection error (errno 98, address already in use)"
            # Just retry
            retries -= 1
            logging.info(f"Connection error on act_batch(), retrying")
            time.sleep(1)


def observe(game_id: int, player_id: int = 0):
    try:
        return requests.get(f'http://localhost:9000/observation?gameID={game_id}&playerID={player_id}').json()
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on observe({game_id}.{player_id}), retrying")
        time.sleep(1)
        return observe(game_id, player_id)


def observe_batch(game_ids):
    retries = RETRIES
    while retries > 0:
        try:
            return requests.get(f'http://localhost:9000/batch-observation', json=[game_ids, []]).json()
        except requests.exceptions.ConnectionError:
            retries -= 1
            logging.info(f"Connection error on observe_batch(), retrying")
            time.sleep(10)


def scalabool(b: bool) -> str:
    return 'true' if b else 'false'


def observe_batch_raw(obs_config: ObsConfig,
                      game_ids: List[Tuple[int, int]],
                      allies: int,
                      drones: int,
                      minerals: int,
                      global_drones: int,
                      tiles: int,
                      relative_positions: bool,
                      v2: bool,
                      extra_build_actions: List[List[int]],
                      map_size: bool = False,
                      last_seen: bool = False,
                      is_visible: bool = False,
                      abstime: bool = False,
                      rule_msdm: bool = False,
                      rule_costs: bool = False,
                      enforce_unit_cap: bool = False,
                      unit_cap_override: int = 0) -> object:
    retries = RETRIES
    ebcstr = ''
    url = f'http://localhost:9000/batch-observation?' \
        f'json=false&' \
        f'allies={allies}&' \
        f'drones={drones}&' \
        f'minerals={minerals}&' \
        f'tiles={tiles}&' \
        f'globalDrones={global_drones}&' \
        f'relativePositions={"true" if relative_positions else "false"}&' \
        f'lastSeen={"true" if last_seen else "false"}&' \
        f'isVisible={"true" if is_visible else "false"}&' \
        f'abstime={"true" if abstime else "false"}&' \
        f'mapSize={"true" if map_size else "false"}&' \
        f'v2={"true" if v2 else "false"}&' \
        f'mineralClaims={scalabool(obs_config.feat_mineral_claims)}&' \
        f'harvestAction={scalabool(obs_config.harvest_action)}&' \
        f'ruleMsdm={scalabool(rule_msdm)}&' \
        f'ruleCosts={scalabool(rule_costs)}&' \
        f'lockBuildAction={scalabool(obs_config.lock_build_action)}&' \
        f'distanceToWall={scalabool(obs_config.feat_dist_to_wall)}&' \
        f'unitCount={scalabool(obs_config.unit_count)}&' \
        f'enforceUnitCap={scalabool(enforce_unit_cap)}&' \
        f'unitCapOverride={unit_cap_override}' + ebcstr
    while retries > 0:
        json = [game_ids, extra_build_actions]
        try:
            response = requests.get(url,
                                    json=json,
                                    stream=True)
            response.raise_for_status()
            response_bytes = response.content
            return np.frombuffer(response_bytes, dtype=np.float32)
        except requests.exceptions.ConnectionError as e:
            retries -= 1
            logging.info(f"Connection error on {url} with json={json}, retrying: {e}")
            time.sleep(10)


def one_hot_to_action(action):
    # 0-5: turn/movement (4 is no turn, no movement)
    # 6: build [0,1,0,0,0] drone (if minerals > 5)
    # 7: harvest
    move = False
    harvest = False
    turn = 0
    build = []
    if action == 0 or action == 1 or action == 2:
        move = True
    if action == 0 or action == 3:
        turn = -1
    if action == 2 or action == 5:
        turn = 1
    if action == 6:
        build = [[0, 1, 0, 0, 0]]
    if action == 7:
        harvest = True

    return {
        "buildDrone": build,
        "move": move,
        "harvest": harvest,
        "transfer": False,
        "turn": turn,
    }


def observation_to_np(observation):
    o = []
    x = float(observation['alliedDrones'][0]['xPos'])
    y = float(observation['alliedDrones'][0]['yPos'])
    o.append(x / 1000.0)
    o.append(y / 1000.0)
    o.append(np.sin(float(observation['alliedDrones'][0]['orientation'])))
    o.append(np.cos(float(observation['alliedDrones'][0]['orientation'])))
    o.append(float(observation['alliedDrones'][0]['storedResources']) / 50.0)
    o.append(1.0 if observation['alliedDrones'][0]['isConstructing'] else -1.0)
    o.append(1.0 if observation['alliedDrones'][0]['isHarvesting'] else -1.0)
    minerals = sorted(observation['minerals'], key=lambda m: dist2(m['xPos'], m['yPos'], x, y))
    for m in range(0, 10):
        if m < len(minerals):
            mx = float(minerals[m]['xPos'])
            my = float(minerals[m]['yPos'])
            o.append((mx - x) / 1000.0)
            o.append((my - y) / 1000.0)
            o.append(dist(mx, my, x, y) / 1000.0)
            o.append(float(minerals[m]['size'] / 100.0))
        else:
            o.extend([0.0, 0.0, 0.0, 0.0])
    return np.array(o, dtype=np.float32)


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy
