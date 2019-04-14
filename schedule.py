import os
import pathlib
from shutil import copyfile
import subprocess
import sys
import tempfile
import time

import click
import yaml


QUEUE_DIR = "/home/clemens/xprun/queue"


@click.command()
@click.option("--repo-path", default=".", help="Path to git code repository to execute.")
@click.option("--revision", default="HEAD", help="Git revision to execute.")
@click.option("--params-file", default="params.yaml", help="Path to parameter file.")
def main(repo_path, revision, params_file):
    pathlib.Path(QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", revision]).decode("UTF-8")[:-1]
    repo_path = os.path.abspath(repo_path)
    
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    job = {
        "repo-path": repo_path,
        "revision": commit,
        "params": params,
    }

    fd, path = tempfile.mkstemp()
    with open(fd, 'w') as f:
        f.write(yaml.dump(job))
        os.rename(path, os.path.join(QUEUE_DIR, f"{int(time.time())}.yaml"))


if __name__ == "__main__":
    main()

