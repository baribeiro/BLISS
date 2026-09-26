"""The search space of each architecture, in one place, so the budget is comparable across them.

The v1 search was three learning rates by two capacities, a six-cell grid per model, chosen because
it was cheap and equal. It is equal in COUNT but not in coverage: a grid of six points explores a
learning rate well and a capacity barely, and it cannot express the knobs that differ between
architectures (the number of slices of a Physics-Attention block, the spectral modes of an SFNO, the
neighbourhood of a graph network). v2 replaces it with a fixed budget of trials per model over a
space of comparable complexity, so that what is held equal is the SEARCH EFFORT and not the shape of
the grid.

What is NOT searched, and is fixed by the protocol for every model: the dataset, the split, the
input definition, the targets, the loss, AdamW, the warm-up and cosine schedule, the weight EMA,
gradient clipping, bf16 autocast, the validation frequency, the epoch budget and the selection
metric. A hyper-parameter search that is allowed to change those is not comparing architectures.

The objective is the VALIDATION buckling_dimple_r2 in the released definition (train --select buckling:
uniform contraction removed, scalar-mean reference, dimple mask on the true deviation), at each run's best
validation epoch; phase 2 keeps the top N_TOP trials and picks the configuration with the highest mean over
N_SEEDS seeds. Never the two-way winner: selecting on the decision metric would answer a different question
from the one the benchmark asks, which is whether a model chosen for being excellent at the continuous
physical response also makes the discrete physical choice. No explicit tie-break is implemented (the code
takes the maximum; exact ties did not occur).

Functions
---------
suggest
    Draw one configuration for `model` from its space.
as_args
    The configuration as train.py command-line arguments.
"""

# a parameter ceiling, so that "more capacity" cannot be bought indefinitely by one family
MAX_PARAMS_M = 60.0

SPACE = {
    "figconv": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        width_mult=("categorical", [0.75, 1.0, 1.25, 1.5]),
        levels=("categorical", [3, 4])),
    "sfno": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        embed=("categorical", [96, 128, 160, 192, 256]),
        layers=("categorical", [4, 6, 8])),
    "mgn": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        hidden=("categorical", [96, 128, 192, 256]),
        blocks=("categorical", [6, 8, 10, 12]), knn=("categorical", [6, 8, 12, 16])),
    "transolver": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        dim=("categorical", [128, 192, 256, 384]), layers=("categorical", [4, 6, 8]),
        slices=("categorical", [32, 64, 128, 256]), dropout=("uniform", 0.0, 0.2)),
    "transformer": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        dim=("categorical", [192, 256, 320, 384]), layers=("categorical", [4, 6, 8]),
        heads=("categorical", [4, 8]), dropout=("uniform", 0.0, 0.2)),
    "deeponet": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        p=("categorical", [64, 128, 256, 512]), width=("categorical", [256, 512, 768]),
        depth=("categorical", [3, 4, 5]), dropout=("uniform", 0.0, 0.2)),
    "mlp": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        hidden=("categorical", [128, 256, 512, 768]),
        blocks=("categorical", [2, 3, 4, 5]), dropout=("uniform", 0.0, 0.3)),
    "mlp80": dict(
        lr=("loguniform", 1e-4, 2e-3), wd=("loguniform", 1e-5, 1e-1),
        hidden=("categorical", [128, 256, 512, 768]),
        blocks=("categorical", [2, 3, 4, 5]), dropout=("uniform", 0.0, 0.3)),
}

N_TRIALS = 25          # budget per model (completed + median-pruned trials count); N_STARTUP random, then TPE
N_STARTUP = 7
EPOCHS_TRIAL = 60
EPOCHS_FINAL = 200     # the confirmation budget
N_TOP = 2
N_SEEDS = 3


def suggest(trial, model):
    """Draw one configuration for `model` from its space.

    Parameters
    ----------
    trial : optuna.Trial
        The trial.
    model : str
        Model name, a key of SPACE.

    Returns
    -------
    cfg : dict
        Hyperparameters of the configuration.
    """
    out = {}
    for name, spec in SPACE[model].items():
        kind = spec[0]
        if kind == "loguniform": out[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "uniform": out[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "categorical": out[name] = trial.suggest_categorical(name, spec[1])
        else: raise ValueError(kind)
    return out


def as_args(model, cfg):
    """The configuration as train.py command-line arguments.

    Parameters
    ----------
    model : str
        Model name.
    cfg : dict
        Hyperparameters.

    Returns
    -------
    args : list of str
        Arguments for train.py.
    """
    a = ["--lr", "%.6g" % cfg["lr"], "--wd", "%.6g" % cfg["wd"]]
    for k, v in cfg.items():
        if k in ("lr", "wd"): continue
        a += ["--%s" % k.replace("_", "-"), "%.6g" % v if isinstance(v, float) else str(v)]
    return a
