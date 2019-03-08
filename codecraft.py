import requests
import json
import logging
import time

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def create_game() -> int:
    try:
        response = requests.post('http://localhost:9000/start-game').json()
        return response['id']
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on observe({game_id}, retrying")
        time.sleep(1)
        return create_game()


def act(game_id: int):
    try:
        # buildDrone: Option[Seq[Int]],
        # move: Boolean,
        # harvest: Boolean,
        # transfer: Boolean,
        # turn: Int /* -1, 0, 1 */
        action = {
            "buildDrone": [[0,1,0,0,0]],
            "move": True,
            "harvest": False,
            "transfer": False,
            "turn": -1,
        }
        requests.post(f'http://localhost:9000/act?gameID={game_id}&playerID=0', json=action)
    except requests.exceptions.ConnectionError:
        # For some reason, a small percentage of requests fails with
        # "connection error (errno 98, address already in use)"
        # Just retry
        logging.info(f"Connection error on act({game_id}), retrying")
        time.sleep(1)
        act(game_id)

def observe(game_id: int):
    try:
        return requests.get(f'http://localhost:9000/observation?gameID={game_id}&playerID=0').json()
    except requests.exceptions.ConnectionError:
        logging.info(f"Connection error on observe({game_id}), retrying")
        time.sleep(1)
        return observe(game_id)

