import time
import logging

import codecraft
import gym_codecraft


def run_codecraft():
  games = []
  for i in range(5):
    game_id = codecraft.create_game()
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
      observation = codecraft.observe(game_id)
      if len(observation['winner']) > 0:
        print(f'Game {game_id} won by {observation["winner"][0]}')
        game_id = codecraft.create_game()
        print("Starting game:", game_id)
        games[i] = game_id
      else:
        codecraft.act(game_id)
      frames += 1


def main():
  logging.basicConfig(level=logging.INFO)


if __name__== "__main__":
  main()

