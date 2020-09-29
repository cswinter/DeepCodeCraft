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
        self.port_offset = 0
        for device in os.environ.get("GPU_DENYLIST", default="").split(","):
            if device != '':
                self.active_jobs_per_device.pop(int(device))
                devices -= 1

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

            while self.queue.qsize() > 0:
                job = self.queue.get()
                required_devices = min(self.devices, job.parallelism)
                if job.parallelism > self.devices and job.parallelism % self.devices != 0:
                    logging.error(f"Can't evenly distribute {job.parallelism} processes across {self.devices} GPUs, dropping job.")
                    continue
                required_slots_per_device = job.parallelism // self.devices if job.parallelism > self.devices else 1
                while True:
                    selected_devices = []
                    with self.lock:
                        min_load = self.concurrency + 1
                        for device, load in self.active_jobs_per_device.items():
                            if load + required_slots_per_device <= self.concurrency // self.devices:
                                if load < min_load:
                                    selected_devices = [device]
                                    min_load = load
                                elif load == min_load:
                                    selected_devices.append(device)
                        if len(selected_devices) >= required_devices:
                            rank = 0
                            for device in selected_devices[:required_devices]:
                                for _ in range(required_slots_per_device):
                                    job_copy = copy.deepcopy(job)
                                    job_copy.set_device(device, rank, 29000 + self.port_offset)
                                    self.active_jobs_per_device[job_copy.device] += 1
                                    threading.Thread(target=self.run_job, args=(job_copy,)).start()
                                    rank += 1
                            self.active_jobs += 1
                            self.port_offset = (self.port_offset + 1) % 1000
                            logging.info(f"In queue: {self.queue.qsize()}  Running: {self.active_jobs_per_device}")
                            break
                    time.sleep(0.1)

                time.sleep(0.1)

            time.sleep(0.1)

    def run_job(self, job):
        try:
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

                logpath = os.path.join(out_dir, f"out{job.rank}.txt")

                logging.info(f"Running {job_desc}")
                logging.info(f"Output in {logpath}")

                with open(logpath, "w+") as outfile:
                    retcode = subprocess.call(
                        ["python3", "main.py", "--out-dir", out_dir] + args,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(job.device)),
                        stdout=outfile, stderr=outfile, cwd=dir
                    )
                if retcode != 0:
                    logging.warning(f"Command {job_desc} returned non-zero exit status {retcode}. Logs: {logpath}")
                else:
                    logging.info(f"Success: {job_desc}")
        finally:
            with self.lock:
                if job.rank == 0:
                    self.active_jobs -= 1
                    self.known_jobs[job.handle] -= 1
                    if self.known_jobs[job.handle] == 0:
                        del self.known_jobs[job.handle]
                self.active_jobs_per_device[job.device] -= 1

    def process_job_file(self, job_file):
        filepath = os.path.join(self.queue_dir, job_file)
        job = yaml.safe_load(open(filepath, "r"))
        param_sets = []
        for param_set in job["params"]:
            param_sets.extend(self.all_combinations(param_set))

        logging.info(f"Enqueuing {len(param_sets)} jobs")
        self.known_jobs[job_file] = len(param_sets)

        for param_set in param_sets:
            self.queue.put(Job(job["repo-path"], job["revision"], param_set, job_file, param_set.get("parallelism", 1)))
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

        result = []
        for param_set in param_sets:
            for _ in range(repetitions):
                result.append(param_set.copy())
        return result


class Job:
    def __init__(self, repo_path, revision, params, handle, parallelism):
        self.repo_path = repo_path
        self.revision = revision
        self.params = params
        self.handle = handle
        self.device = None
        self.parallelism = parallelism
        self.descriptor = "-".join([revision[:6]] + [f'{k}{v}' for k, v in params.items()])
        self.rank = 0

    def set_device(self, device, rank, discovery_port):
        self.device = device
        self.rank = rank
        self.params['device'] = device
        self.params['rank'] = rank
        self.params['discovery_port'] = discovery_port


@click.command()
@click.option("--concurrency", default=8, help="Maximum number of jobs running at the same time.")
def main(concurrency):
    gpus = len(subprocess.check_output(["nvidia-smi", "-L"]).decode("UTF-8").split("\n")) - 1
    job_queue = JobQueue("/home/clemens/xprun/queue", concurrency, gpus)
    job_queue.run()


if __name__ == "__main__":
    main()

