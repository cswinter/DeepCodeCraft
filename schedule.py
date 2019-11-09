import os
import pathlib
from shutil import copyfile
import subprocess
import sys
import tempfile
import time

import click
import yaml


QUEUE_DIR = "192.168.0.102:/home/clemens/xprun/queue"


@click.command()
@click.option("--repo-path", default="git@github.com:cswinter/DeepCodeCraft.git", help="Path to git code repository to execute.")
@click.option("--revision", default="HEAD", help="Git revision to execute.")
@click.option("--params-file", default=None, help="Path to parameter file.")
@click.option("--hps", default=None, help="List of hyperparameters in format name1:value1,name2:value2")
def main(repo_path, revision, params_file, hps):
    # pathlib.Path(QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", revision]).decode("UTF-8")[:-1]

    if params_file:
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
    elif hps:
        params = [{}]
        for param in hps.split(","):
            key, value = param.split(":")
            params[0][key] = value
    else:
        params = [{}]

    job = {
        "repo-path": repo_path,
        "revision": commit,
        "params": params,
    }

    fd, path = tempfile.mkstemp()
    with open(fd, 'w') as f:
        f.write(yaml.dump(job))
    subprocess.check_call(["rsync", path, os.path.join(QUEUE_DIR, f"{int(time.time())}.yaml")])


if __name__ == "__main__":
    main()

