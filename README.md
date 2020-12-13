# Deep CodeCraft

Hacky research code that trains policies for the [CodeCraft](http://codecraftgame.org/) real-time strategy game with proximal policy optimization.

## Requirements

- Python >= 3.7, pip
- [CodeCraft Server](https://github.com/cswinter/CodeCraftServer/)

## Setup

Install dependencies with

```
pip install -r requirements.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.6.0+${CUDA}.html
```

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101` or `cu102` depending on your PyTorch installation.

If you want the training code to record metrics to [Weights & Biases](https://www.wandb.com/), run `wandb login`.

## Usage

The first step is to setup and run [CodeCraft Server](https://github.com/cswinter/CodeCraftServer/).

### Training

To train a policy with the default set of hyperparameters, run:

```
EVAL_MODELS_PATH=/path/to/golden-models python main.py --hpset=standard --out-dir=${OUT_DIR}`
```

Logs and model checkpoints will be written to the `${OUT_DIR}` directory.
If you want policies to be evaluted against a set of fixed opponents during training, download the required checkpoints [available here](https://www.dropbox.com/sh/h0f4faf7cbubn3t/AACfYYYY9kwPwNjm_TeCahxAa/golden-models?dl=0) to the right subfolder in the folder specified by `EVAL_MODEL_PATH`.
For evaluations with the standard config, you need `standard/curious-galaxy-40M.pt` and `standard/graceful-frog-100M.pt`.
To disable evaluation of the policy during training, set `--eval_envs=0`.
To see additional options, run `python main.py --help` and consult [hyperparams.py](https://github.com/cswinter/DeepCodeCraft/blob/master/hyper_params.py).

### Showmatch

To run games with already trained policies, run:

```
python showmatch.py /path/to/policy1.pt /path/to/policy2.pt --task=STANDARD --num_envs=64
```

You can then watch the games at <http://localhost:9000/observe?autorestart=true&autozoom=true>.

### Job Runner

The job runner allows you to schedule and execute many runs in parallel.
The command

```
python runner.py --jobfile-dir=${JOB_DIR} --out-dir=${OUT_DIR} --concurrency=${CONCURRENCY}
```

starts a job runner that watches the `${JOB_DIR}` directory for new jobs, writes results to folders created in `${OUT_DIR}` and will run up to `${CONCURRENCY}` experiments in parallel.

You can then schedule jobs with

```
python schedule.py --repo-path=https://github.com/cswinter/DeepCodeCraft.git --queue-dir=${JOB_DIR} --params-file=params.yaml
```

where `params.yaml` is a file that specifies the set of hyperparameters to use, for example:

```
- hpset: standard
  adr_variety: [0.5, 0.3]
  lr: [0.001, 0.0003]
- hpset: standard
  repeat: 4
  steps: 300e6
```

The `repeat` parameter tells the job runner to spawn multiple runs.
When a hyperparameter is set to a list of different values, one experiment is spawned for each combination.
So above `params.yaml` will spawn a total of 8 experiment runs, 4 of which will run for 300 million samples with the default set of hyperparameters, and one additional run for all 4 combinations of the `adr_variety` and `lr` hyperparameters.

The `${JOB_DIR}` may be on a remote machine that you can access via ssh/rsync, e.g. `--queue-dir=192.168.0.101:/home/clemens/xprun/queue`.

### Citation

```
@misc{DeepCodeCraft2020,
  author = {Winter, Clemens},
  title = {Deep CodeCraft},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/cswinter/DeepCodeCraft}}
}
```
