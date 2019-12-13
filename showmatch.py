import click
import torch
import numpy as np

from main import load_policy, eval
from gym_codecraft import envs


@click.command()
@click.argument('model1_path', nargs=1)
@click.argument('model2_path', nargs=1)
@click.option('--task', default='ARENA_TINY_2V2')
@click.option('--randomize/--no-randomize', default=False)
@click.option('--hardness', default=0)
@click.option('--num_envs', default=4)
def showmatch(model1_path, model2_path, task, randomize, hardness, num_envs):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    objective = envs.Objective(task)
    policy1, _, _ = load_policy(model1_path, device)
    eval(
        policy=policy1,
        num_envs=num_envs,
        device=device,
        objective=objective,
        eval_steps=int(1e20),
        opponents={'player2': {'model_file': model2_path}},
        printerval=100,
        randomize=randomize,
        hardness=hardness,
    )


if __name__ == "__main__":
    showmatch()
