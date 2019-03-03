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
    requests.post(f'http://localhost:9000/act?gameID={game_id}&playerID=0')
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


def main():
  logging.basicConfig(level=logging.INFO)

  games = []
  for i in range(5):
    game_id = create_game()
    print("Starting game:", game_id)
    games.append(game_id)

  log_interval = 5
  frames = 0
  last_time = time.time()

  while True:
    elapsed = time.time() - last_time
    if elapsed > log_interval:
      logging.info(f"{frames/elapsed}fps")
      frames = 0
      last_time = time.time()

    for i in range(len(games)):
      game_id = games[i]
      observation = observe(game_id)
      if len(observation['winner']) > 0:
        print(f'Game {game_id} won by {observation["winner"][0]}')
        game_id = create_game()
        print("Starting game:", game_id)
        games[i] = game_id
      else:
        act(game_id)
      frames += 1


if __name__== "__main__":
  main()

