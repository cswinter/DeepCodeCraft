import click
import torch
import numpy as np

from main import load_policy
from gym_codecraft import envs


@click.command()
@click.argument('model1_path', nargs=1)
@click.argument('model2_path', nargs=1)
@click.option('--task', default='ARENA_TINY_2V2')
@click.option('--randomize/--no-randomize', default=False)
def showmatch(model1_path, model2_path, task, randomize):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    nenv = 128

    policy1 = load_policy(model1_path).to(device)
    policy2 = load_policy(model2_path).to(device)

    env = envs.CodeCraftVecEnv(nenv,
                               nenv // 2,
                               envs.Objective(task),
                               action_delay=0,
                               randomize=randomize,
                               stagger=True,
                               fair=True)

    returns = []
    lengths = []
    obs = env.reset()
    evens = list([2 * i for i in range(nenv // 2)])
    odds = list([2 * i + 1 for i in range(nenv // 2)])
    policy1_envs = evens
    policy2_envs = odds

    try:
        while True:
            obs_tensor = torch.tensor(obs).to(device)
            obs_policy1 = obs_tensor[policy1_envs]
            obs_policy2 = obs_tensor[policy2_envs]
            actions1, _, _, _ = policy1.evaluate(obs_policy1)
            actions2, _, _, _ = policy2.evaluate(obs_policy2)

            actions = np.zeros((nenv, policy1.allies), dtype=np.int)
            actions[policy1_envs] = actions1.cpu()
            actions[policy2_envs] = actions2.cpu()

            obs, rews, dones, infos = env.step(actions)

            for info in infos:
                index = info['episode']['index']
                if index in policy1_envs:
                    ret = info['episode']['r']
                    length = info['episode']['l']
                    returns.append(ret)
                    lengths.append(length)

                    if len(returns) % 50 == 0:
                        print(np.array(returns).mean())
    except KeyboardInterrupt:
        print('exiting')

    env.close()


if __name__ == "__main__":
    showmatch()
