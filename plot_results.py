import wandb
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Union
from functools import lru_cache
from dataclasses import dataclass

EVAL_METRICS = {
    "Replicator": "eval_mean_score_vs_replicator",
    "Destroyer": "eval_mean_score_vs_destroyer",
    "Curious Galaxy 40M": "eval_mean_score_vs_curious-galaxy-40",
    "Graceful Frog 100M": "eval_mean_score_vs_graceful-frog-100",
}

@dataclass
class Experiment:
    descriptor: str
    label: str

@lru_cache(maxsize=None)
def fetch_run_data(descriptor: str, metrics: Union[List[str], str]) -> List[Tuple[np.array, np.array]]:
    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)
    api = wandb.Api()
    runs = api.runs("cswinter/deep-codecraft-ablations", {"config.descriptor": descriptor})
    
    curves = []
    for run in runs:
        step = []
        value = []
        vals = run.history(keys=metrics, samples=100, pandas=False)
        for entry in vals:
            if metrics[0] in entry:
                step.append(entry['_step'] * 1e-6)
                meanvalue = np.array([entry[metric] for metric in metrics]).mean()
                value.append(meanvalue)
        curves.append((np.array(step), np.array(value)))
    return curves

def final_score(descriptor: str) -> Tuple[float, float]:
    runs = fetch_run_data(descriptor, tuple(EVAL_METRICS.values()))
    runs = [run for run in runs if len(run[0]) == 26]
    if len(runs) < 8:
        print(f"Only {len(runs)} for {descriptor}")
    values = np.array([[run[1][i] for run in runs] for i in range(len(runs[0][0]))])
    return values.mean(axis=1)[-1], (values.std(axis=1, ddof=1)/math.sqrt(len(runs)))[-1]

def errplot3(ax, xps: List[Experiment], metrics: Union[List[str], str], title: str):
    colors = ['tab:blue', 'tab:orange']
    markers = ['x', '+']
  
    for i, xp in enumerate(xps):
        curves = fetch_run_data(xp.descriptor, metrics)
        curves = [curve for curve in curves if len(curve[0]) == 26]

        samples = curves[0][0]
        values = np.array([[curve[1][i] for curve in curves] for i in range(len(samples))])
        ax.errorbar(
            samples,
            values.mean(axis=1),
            yerr=values.std(axis=1, ddof=1)/math.sqrt(len(curves)),
            color=colors[i],
            alpha=0.75,
            capsize=3,
            capthick=1,
            linestyle=":",
            label=xp.label,
        )
        ax.fill_between(
            samples,
            values.min(axis=1),
            values.max(axis=1),
            alpha=.25
        )
            

    ax.set(xlabel='million samples', ylabel='eval score', title=title, xlim=(0, 125.35), ylim=(-1, 1))
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1])
    ax.set_xticks([0, 25, 50, 75, 100, 125])
    ax.legend(loc='upper left')
    ax.grid()


def plot_drone_types(ax, runid: str):
    run = wandb.Api().run(f'cswinter/deep-codecraft-ablations/{runid}')
    frac_metrics = sorted([key for key in run.summary.keys() if key.startswith('frac')])
    fracs = [
        {row['_step']: row[fm] for row in run.scan_history(keys=['_step', fm], page_size=int(1e9))}
        for fm in frac_metrics
    ]
    steps = set()
    for frac in fracs:
        steps.update(frac.keys())
    steps = sorted(list(steps))
    fixed_fracs = []
    for frac in fracs:
        fixed_frac = []
        for step in steps:
           fixed_frac.append(frac.get(step, 0.0)) 
        fixed_fracs.append(np.array(fixed_frac))

    binned_fracs = []
    bins = np.linspace(0, 125e6, 250)
    digitized = np.digitize(steps, bins)
    for frac in fixed_fracs:
        bin_means = [frac[digitized == i].mean() for i in range(1, len(bins)+1)]
        binned_fracs.append(bin_means)

    labels = [m[len('frac_'):] for m in frac_metrics]
    ax.stackplot([0.5 * i + 0.25 for i in range(250)], binned_fracs, labels=labels)
    ax.set(xlabel='million samples', ylabel='drone type fraction', xlim=(0, 125), ylim=(0, 1), title=' '.join(run.name.split('-')[:2]))
    ax.set_xticks([0, 25, 50, 75, 100, 125])
    ax.legend(reversed(plt.legend().legendHandles), reversed(labels), loc='lower right')


def plot2dt(runid1: str, runid2: str):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_drone_types(axs[0], runid1)
    plot_drone_types(axs[1], runid2)
    fig.savefig(f"plots/dronetypes.svg")
    fig.savefig(f"plotspng/dronetypes.png")
    plt.show()


def plot(xps: List[Experiment], metrics: List[str], name: str):
    fig, ax = plt.subplots(figsize=(12, 9))
    errplot3(ax, xps, metrics, name)
    fig.savefig(f"plots/{name}.svg")
    fig.savefig(f"plotspng/{name}.png")
    plt.show()


def plot4(descriptors: List[str], metrics: Dict[str, str], name: str):
    assert len(metrics) == 4

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    #fig.suptitle(name)
    for i, (metric_title, metric_name) in enumerate(metrics.items()):
        print(f"{i}/{len(metrics)} {len(axs)}")
        errplot3(axs[i // 2, i % 2], descriptors, metric_name, metric_title)
    fig.savefig(f"plots/{name}.svg")
    fig.savefig(f"plotspng/{name}.png")
    plt.show()


if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plotspng'):
    os.makedirs('plotspng')

plot2dt('i17gv7pw', 'sidk0gu4')

baseline = Experiment(descriptor="f2034f-hpsetstandard", label="baseline")
adr_ablations = [
    Experiment("f2034f-hpsetstandard-mothership_damage_scale0.0-mothership_damage_scale_schedule", "module cost, map curriculum"),
    Experiment("f2034f-adr_variety0.0-adr_variety_schedule-hpsetstandard", "mothership damage, map curriculum"),
    Experiment("f2034f-adr_variety0.0-adr_variety_schedule-hpsetstandard-mothership_damage_scale0.0-mothership_damage_scale_schedule", "map curriculum"),

    Experiment("f2034f-adr_hstepsize0.0-hpsetstandard-linear_hardnessFalse-task_hardness150", "mothership damage, module cost, map randomization"),
    Experiment("f2034f-adr_hstepsize0.0-hpsetstandard-linear_hardnessFalse-mothership_damage_scale0.0-mothership_damage_scale_schedule-task_hardness150", "module cost, map randomization"),
    Experiment("f2034f-adr_hstepsize0.0-adr_variety0.0-adr_variety_schedule-hpsetstandard-linear_hardnessFalse-task_hardness150", "mothership damage, map randomization"),
    Experiment("f2034f-adr_hstepsize0.0-adr_variety0.0-adr_variety_schedule-hpsetstandard-linear_hardnessFalse-mothership_damage_scale0.0-mothership_damage_scale_schedule-task_hardness150", "map randomization"),

    Experiment("049430-batches_per_update64-bs256-hpsetstandard", "mothership damage, module cost, fixed map"),
    Experiment("049430-adr_variety0.0-adr_variety_schedule-batches_per_update64-bs256-hpsetstandard", "mothership damage, fixed map"),
    Experiment("049430-adr_variety0.0-adr_variety_schedule-batches_per_update64-bs256-hpsetstandard-mothership_damage_scale0.0-mothership_damage_scale_schedule", "fixed map"),
]
ablations = [
    Experiment("f2034f-hpsetstandard-partial_score0.0", "sparse reward"),
    Experiment("f2034f-hpsetstandard-use_privilegedFalse",  "vf hidden information"),
    Experiment("f2034f-d_agent128-d_item64-hpsetstandard", "smaller network"),
    Experiment("f2034f-batches_per_update64-bs256-hpsetstandard-rotational_invarianceFalse", "no rotational invariance"),
    Experiment("7a9d92-hpsetstandard", "no shared spatial embeddings"),
    *adr_ablations,
]


for xp in [baseline] + adr_ablations:
    label = xp.label
    score_mean, score_sem = final_score(xp.descriptor)
    print(f"{label} {score_mean} {score_sem}")

plot([baseline], tuple(EVAL_METRICS.values()), "baseline")
plot4([baseline], EVAL_METRICS, "breakdown")
plot4([baseline, ablations[3]], EVAL_METRICS, "breakdown cost adr")


for xp in ablations:
    print(f"plotting {xp.label}")
    plot([baseline, xp], tuple(EVAL_METRICS.values()), xp.label)
    plot4([baseline, xp], EVAL_METRICS, f"breakdown {xp.label}")

