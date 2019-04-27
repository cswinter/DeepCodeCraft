import requests
import json
import logging
import time

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


RETRIES = 100

def create_game(game_length = None) -> int:
    try:
        if game_length:
            response = requests.post(f'http://localhost:9000/start-game?maxTicks={game_length}').json()
        else:
            response = requests.post('http://localhost:9000/start-game').json()
        return int(response['id'])
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on create_game, retrying")
        time.sleep(1)
        return create_game()


def act(game_id: int, move, turn):
    retries = 100
    while retries > 0:
        try:
            # buildDrone: Option[Seq[Int]],
            # move: Boolean,
            # harvest: Boolean,
            # transfer: Boolean,
            # turn: Int /* -1, 0, 1 */
            action = {
                "buildDrone": [],
                "move": move,
                "harvest": False,
                "transfer": False,
                "turn": turn,
            }
            requests.post(f'http://localhost:9000/act?gameID={game_id}&playerID=0', json=action)
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
    for (game_id, move, turn, buildSpec, harvest) in actions:
        action = {
            "buildDrone": buildSpec,
            "move": move,
            "harvest": True,#harvest,
            "transfer": False,
            "turn": turn,
        }
        payload[game_id] = action

    retries = 100
    while retries > 0:
        try:
            requests.post(f'http://localhost:9000/batch-act', json=payload)
            return
        except requests.exceptions.ConnectionError:
            # For some reason, a small percentage of requests fails with
            # "connection error (errno 98, address already in use)"
            # Just retry
            retries -= 1
            logging.info(f"Connection error on act_batch(), retrying")
            time.sleep(1)

def observe(game_id: int):
    try:
        return requests.get(f'http://localhost:9000/observation?gameID={game_id}&playerID=0').json()
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on observe({game_id}), retrying")
        time.sleep(1)
        return observe(game_id)

def observe_batch(game_ids):
    retries = RETRIES
    while retries > 0:
        try:
            return requests.get(f'http://localhost:9000/batch-observation', json=game_ids).json()
        except requests.exceptions.ConnectionError:
            retries -= 1
            logging.info(f"Connection error on observe_batch(), retrying")
            time.sleep(10)
