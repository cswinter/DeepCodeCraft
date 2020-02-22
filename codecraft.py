import requests
import logging
import time

import numpy as np


RETRIES = 100


def create_game(game_length: int = None, action_delay: int = 0, self_play: bool = False, custom_map=None, strong_scripted_opponent=False) -> int:
    if custom_map is None:
        custom_map = ''
    try:
        scripted_opponent = 'false' if self_play else 'true'
        idle_opponent = 'false' if strong_scripted_opponent else 'true'
        if game_length:
            response = requests.post(f'http://localhost:9000/start-game'
                                     f'?maxTicks={game_length}'
                                     f'&actionDelay={action_delay}'
                                     f'&scriptedOpponent={scripted_opponent}'
                                     f'&idleOpponent={idle_opponent}',
                                     json=custom_map).json()
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


def act_batch(actions, disable_harvest: bool = False):
    payload = {}
    for game_id, player_id, player_actions in actions:
        player_actions_json = []
        for move, turn, buildSpec, harvest in player_actions:
            player_actions_json.append({
                "buildDrone": buildSpec,
                "move": move,
                "harvest": not disable_harvest,#harvest,
                "transfer": False,
                "turn": turn,
            })
        payload[f'{game_id}.{player_id}'] = player_actions_json

    retries = 100
    while retries > 0:
        try:
            requests.post(f'http://localhost:9000/batch-act', json=payload).raise_for_status()
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
            return requests.get(f'http://localhost:9000/batch-observation', json=game_ids).json()
        except requests.exceptions.ConnectionError:
            retries -= 1
            logging.info(f"Connection error on observe_batch(), retrying")
            time.sleep(10)


def observe_batch_raw(game_ids, allies, drones, minerals, global_drones, relative_positions, v2, extra_build_costs = []):
    retries = RETRIES
    ebcstr = ''
    if len(extra_build_costs) > 0:
        ebcstr = ''.join([f'&actions={c}' for c in extra_build_costs])
    url = f'http://localhost:9000/batch-observation?' \
        f'json=false&' \
        f'allies={allies}&' \
        f'drones={drones}&' \
        f'minerals={minerals}&' \
        f'globalDrones={global_drones}&' \
        f'relativePositions={"true" if relative_positions else "false"}&' \
        f'v2={"true" if v2 else "false"}' + ebcstr
    while retries > 0:
        try:
            response = requests.get(url,
                                    json=game_ids,
                                    stream=True)
            response_bytes = response.content
            return np.frombuffer(response_bytes, dtype=np.float32)
        except requests.exceptions.ConnectionError as e:
            retries -= 1
            logging.info(f"Connection error on {url} with json={game_ids}, retrying: {e}")
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
        "harvest": True,
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
