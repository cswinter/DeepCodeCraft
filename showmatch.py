import click
import torch
import numpy as np

from main import load_policy, eval
from gym_codecraft import envs


@click.command()
@click.argument('model_paths', nargs=-1)
@click.option('--task', default='ARENA_TINY_2V2')
@click.option('--randomize/--no-randomize', default=False)
@click.option('--hardness', default=0)
@click.option('--num_envs', default=4)
@click.option('--symmetric/--no-symmetric', default=True)
@click.option('--random_rules', default=0.0)
def showmatch(model_paths, task, randomize, hardness, num_envs, symmetric, random_rules):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    if len(model_paths) == 1:
        opponents = None
    elif len(model_paths) == 2:
        opponents = {'player2': {'model_file': model_paths[1]}}
    else:
        raise Exception("Invalid args")
    objective = envs.Objective(task)
    policy1, _, _, _ = load_policy(model_paths[0], device)
    eval(
        policy=policy1,
        num_envs=num_envs,
        device=device,
        objective=objective,
        eval_steps=int(1e20),
        opponents=opponents,
        printerval=500,
        randomize=randomize,
        hardness=hardness,
        symmetric=symmetric,
        random_rules=random_rules,
    )


if __name__ == "__main__":
    showmatch()
