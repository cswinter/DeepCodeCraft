import copy
import logging
import os
import pathlib
import queue
import subprocess
import threading
import tempfile
import time
import yaml

import click


OUT_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft'

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


class JobQueue:
    def __init__(self, queue_dir, concurrency, devices):
        self.queue_dir = queue_dir
        self.devices = devices
        self.concurrency = concurrency
        self.known_jobs = {}
        self.queue = queue.Queue()
        self.active_jobs = 0
        self.active_jobs_per_device = {device: 0 for device in range(devices)}
        self.lock = threading.Lock()

    
    def run(self):
        logging.info(f"Watching {self.queue_dir} for new jobs...")

        while True:
            for job_file in os.listdir(self.queue_dir):
                if job_file not in self.known_jobs:
                    if job_file.startswith("."):
                        logging.info(f"Ignoring hidden file {job_file}")
                        continue
                    logging.info(f"Found new job file {job_file}")
                    self.process_job_file(job_file)
            
            self.lock.acquire()
            while self.queue.qsize() > 0 and self.active_jobs < self.concurrency:
                job = self.queue.get()

                min_load = self.concurrency
                min_device = -1
                for device, load in self.active_jobs_per_device.items():
                    if load < min_load:
                        min_load = load
                        min_device = device
                job.set_device(min_device)
                threading.Thread(target=self.run_job, args=(job,)).start()
                logging.info(f"In queue: {self.queue.qsize()}  Running: {self.active_jobs_per_device}")
                time.sleep(0.1)
            self.lock.release()

            time.sleep(0.1)

    
    def run_job(self, job):
        with tempfile.TemporaryDirectory() as dir:

            def git(args, workdir=dir):
                FNULL = open(os.devnull, 'w')
                cmd = ["git"]
                if workdir is not None:
                    cmd.extend(["-C", dir])
                cmd.extend(args)
                subprocess.check_call(cmd, stdout=FNULL, stderr=subprocess.STDOUT)

            git(["clone", job.repo_path, dir], workdir=None)
            git(["reset", "--hard", "HEAD"])
            git(["clean", "-fd"])
            try:
                git(["checkout", job.revision])
            except subprocess.CalledProcessError:
                logging.error(f"Failed to checkout revision {job.revision}! Aborting.")
                return

            revision = subprocess.check_output(
                    ["git", "-C", dir, "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")[:-1]

            out_dir = os.path.join(OUT_ROOT_DIR, f'{time.strftime("%Y-%m-%d~%H:%M:%S")}-{revision}')
            for name, value in job.params.items():
                out_dir += f"-{name}{value}"
            pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

            job_desc = f"{job.repo_path} at {job.revision} with {job.params}"
            args = []
            for name, value in job.params.items():
                if isinstance(value, bool):
                    if value:
                        args.append(f'--{name}')
                    else:
                        args.append(f'--no-{name}')
                else:
                    args.append(f"--{name}={value}")
            args.append(f"--descriptor={job.descriptor}")

            logpath = os.path.join(out_dir, "out.txt")

            logging.info(f"Running {job_desc}")
            logging.info(f"Output in {logpath}")

            self.active_jobs += 1
            self.active_jobs_per_device[job.device] += 1

            with open(logpath, "w+") as outfile:
                retcode = subprocess.call(["python3", "main.py", "--out-dir", out_dir] + args,
                                          stdout=outfile, stderr=outfile, cwd=dir)
            if retcode != 0:
                logging.warning(f"Command {job_desc} returned non-zero exit status {retcode}. Logs: {logpath}")
            else:
                logging.info(f"Success: {job_desc}")

        self.lock.acquire()
        self.active_jobs -= 1
        self.known_jobs[job.handle] -= 1
        self.active_jobs_per_device[job.device] -= 1
        if self.known_jobs[job.handle] == 0:
            del self.known_jobs[job.handle]
        self.lock.release()


    def process_job_file(self, job_file):
        filepath = os.path.join(self.queue_dir, job_file)
        job = yaml.safe_load(open(filepath, "r"))
        param_sets = []
        for param_set in job["params"]:
            param_sets.extend(self.all_combinations(param_set))

        logging.info(f"Enqueuing {len(param_sets)} jobs")
        self.known_jobs[job_file] = len(param_sets)

        for param_set in param_sets:
            self.queue.put(Job(job["repo-path"], job["revision"], param_set, job_file))
        os.remove(filepath)


    def all_combinations(self, params_dict):
        param_sets = [{}]
        if 'repeat' in params_dict:
            repetitions = params_dict['repeat']
            del(params_dict['repeat'])
        else:
            repetitions = 1

        for name, values in params_dict.items():
            if type(values) is list:
                new_sets = []
                for value in values:
                    for param_set in param_sets:
                        ps = copy.deepcopy(param_set)
                        ps[name] = value
                        new_sets.append(ps)
                param_sets = new_sets
            else:
                for param_set in param_sets:
                    param_set[name] = values

        return param_sets * repetitions


class Job:
    def __init__(self, repo_path, revision, params, handle):
        self.repo_path = repo_path
        self.revision = revision
        self.params = params
        self.handle = handle
        self.device = None
        self.descriptor = "-".join([revision[:6]] + [f'{k}{v}' for k, v in params.items()])

    def set_device(self, device):
        self.device = device
        self.params['device'] = device


@click.command()
@click.option("--concurrency", default=8, help="Maximum number of jobs running at the same time.")
def main(concurrency):
    gpus = len(subprocess.check_output(["nvidia-smi", "-L"]).decode("UTF-8").split("\n")) - 1
    job_queue = JobQueue("/home/clemens/xprun/queue", concurrency, gpus)
    job_queue.run()


if __name__ == "__main__":
    main()

