import gym
from gym import error, spaces, utils
from gym.utils import seeding

import codecraft


class CodeCraftEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        self.game_id = codecraft.create_game()
        self.done = False

    def step(self, action):
        codecraft.act(self.game_id)

    def reset(self):
        if self.done:
            self.game_id = codecraft.create_game()
            self.done = False

    def render(self, mode='human'):
        raise Exception("Can't render CodeCraft")

    def close(self):
        pass
