"""Set models on the list of defects: flat -> DeepSets -> pairwise, on the parametric master table.

    python benchmark/parametric/set_train.py hpo   --arch pairwise --split B1 --target residual [--extra singles,doubles]
    python benchmark/parametric/set_train.py final --arch pairwise --split B1 --target residual     (opens the test set)

SPLITS. B1/B2/B3 are the released splits. `comp` is the compositional split: train on single- and two-defect
shells, test on the multi-defect core.

`--extra singles,doubles` adds those shells to the TRAINING part of a released split, never to validation or test.

The intervention tests P1-P4 never enter training, validation or test (the loader drops them).

TARGET. `residual` learns kappa - kappa_WL and predicts kappa_WL + f(.); `kappa` learns kappa directly. The
reported error is always on kappa.

Functions
---------
prep
    Standardised tensors of one part of the split.
erro
    Median relative error of kappa, in percent.
r2
    Coefficient of determination of kappa.
treina
    Train one set model with the hyperparameters of a trial.
avalia
    Error and R^2 of kappa on one part of the split.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
import set_models as SM                                                  # noqa: E402
from weakest_link import EloFraco                                        # noqa: E402

B = os.path.dirname(os.path.dirname(ML))
OUT = B + "/outputs"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("modo", choices=["hpo", "final"])
ap.add_argument("--arch", required=True, choices=["flat", "deepsets", "pairwise"])
ap.add_argument("--split", default="B1", choices=["B1", "B2", "B3", "comp"])
ap.add_argument("--target", default="residual", choices=["kappa", "residual"])
ap.add_argument("--extra", default="", help='singles,doubles to add to TRAINING')
ap.add_argument("--n-trials", type=int, default=50)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--device", default="cuda")
ap.add_argument("--tag", default="")
ap.add_argument("--storage", default=OUT + "/optuna")
A = ap.parse_args()
DEV = torch.device(A.device if torch.cuda.is_available() or A.device == "cpu" else "cpu")

# ---------------------------------------------------------------- dados
DF = pd.read_parquet(OUT + ("/parametric_master_kappa_nosso.parquet" if os.environ.get("BLISS_KAPPA_NOSSO")
                             else "/parametric_master.parquet")).set_index("id", drop=False)   # multi kappa ours with the variable
if "held_out" in DF.columns and DF["held_out"].any():
    n = int(DF["held_out"].sum())
    DF = DF[~DF["held_out"]]
    print('removed %d shells marked held_out (intervention tests P1-P4)' % n, flush=True)
ELO = EloFraco()

EXTRA = {"singles": "single", "doubles": "double"}
extra_ids = []
for nome in [x for x in A.extra.split(",") if x]:
    extra_ids += DF[DF.origin == EXTRA[nome]].index.tolist()

if A.split == "comp":
    par = DF[DF.origin.isin(["single", "double"])].index.to_numpy()
    _sp = json.load(open("%s/BLISS-1.0/splits/B1.json" % B))
    _canon = set(_sp["train"]) | set(_sp["val"]) | set(_sp["test"])
    mul = np.array([i for i in DF.index[DF.origin.str.startswith("core")] if i in _canon])
    rng = np.random.default_rng(2027)
    par = rng.permutation(par)
    corte = int(0.88 * len(par))
    TR, VA, TE = par[:corte].tolist(), par[corte:].tolist(), mul.tolist()
    extra_ids = []
else:
    sp = json.load(open("%s/BLISS-1.0/splits/%s.json" % (B, A.split)))
    ok = lambda ids: [i for i in ids if i in DF.index]
    TR, VA, TE = ok(sp["train"]), ok(sp["val"]), ok(sp["test"])
    TR = TR + [i for i in extra_ids if i not in set(VA) | set(TE)]

NMAX = int(DF.loc[TR + VA + TE, "N"].max())
DADOS = {n: SM.tensores(DF, ids, ELO, DEV, Nmax=NMAX)
         for n, ids in (("tr", TR), ("va", VA), ("te", TE))}
MU_X, SD_X, MU_G, SD_G = SM.normaliza(*[DADOS["tr"][i] for i in (0, 1, 2)])


def prep(chave):
    """Standardised tensors of one part of the split.

    Parameters
    ----------
    chave : str, {'tr', 'va', 'te'}
        Part of the split.

    Returns
    -------
    X, G, M, y, kwl : torch.Tensor
        Defect features, shell features, mask, kappa and weakest-link reference.
    """
    X, G, M, y, kwl = DADOS[chave]
    return (X - MU_X) / SD_X * M.unsqueeze(-1), (G - MU_G) / SD_G, M, y, kwl


def erro(p, y):
    """Median relative error of kappa, in percent.

    Parameters
    ----------
    p : numpy.ndarray
        Predicted kappa.
    y : numpy.ndarray
        True kappa.

    Returns
    -------
    err : float
        Median relative error (%).
    """
    return float(np.median(np.abs(p - y) / y) * 100)


def r2(p, y):
    """Coefficient of determination of kappa.

    Parameters
    ----------
    p : numpy.ndarray
        Predicted kappa.
    y : numpy.ndarray
        True kappa.

    Returns
    -------
    r2 : float
        R^2.
    """
    return float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())


CURVA = []                                                                # per-epoch curve of the last treina()
REGISTA_CURVA = bool(os.environ.get("BLISS_KAPPA_NOSSO"))                 # only in the _kn finals


def treina(t, seed, ep=None):
    """Train one set model with the hyperparameters of a trial.

    Parameters
    ----------
    t : optuna.Trial
        The trial.
    seed : int
        Seed.
    ep : int, default=None
        Number of epochs; the default of the protocol if None.

    Returns
    -------
    net : torch.nn.Module
        The trained model.
    """
    torch.manual_seed(seed)
    net = SM.constroi(A.arch, t, NMAX).to(DEV)
    lr = t.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = t.suggest_float("wd", 1e-6, 1e-1, log=True)
    bs = t.suggest_categorical("bs", [32, 64, 128])
    ep = ep or A.epochs
    Xa, Ga, Ma, ya, wa = prep("tr")
    alvo = (ya - wa) if A.target == "residual" else ya
    if A.target == "residual" and not torch.isfinite(alvo).all():
        raise RuntimeError('kappa_WL is NaN in training: the weakest link does not cover these shells')
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep, eta_min=lr * 0.01)
    g = torch.Generator().manual_seed(seed)
    CURVA.clear()
    for e_ in range(ep):
        net.train(); soma, n_ = 0.0, 0
        for j in torch.randperm(len(Xa), generator=g).to(DEV).split(bs):
            opt.zero_grad()
            l_ = nn.functional.mse_loss(net(Xa[j], Ga[j], Ma[j]), alvo[j])
            l_.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            soma += float(l_) * len(j); n_ += len(j)
        sch.step()
        if REGISTA_CURVA:
            ev_, r2_ = avalia(net, "va")
            CURVA.append(dict(epoch=e_ + 1, train_mse=soma / max(n_, 1), val_err_pct=float(ev_), val_r2=float(r2_)))
    return net


@torch.no_grad()
def avalia(net, chave):
    """Error and R^2 of kappa on one part of the split.

    Parameters
    ----------
    net : torch.nn.Module
        The trained model.
    chave : str, {'tr', 'va', 'te'}
        Part of the split.

    Returns
    -------
    err : float
        Median relative error of kappa (%).
    r2 : float
        R^2 of kappa.
    """
    net.eval()
    X, G, M, y, w = prep(chave)
    p = net(X, G, M)
    if A.target == "residual":
        p = p + w
    p, y = p.cpu().numpy(), y.cpu().numpy()
    return erro(p, y), r2(p, y)


import optuna                                                            # noqa: E402
import socket                                                            # noqa: E402
from optuna.storages import JournalStorage                               # noqa: E402
from optuna.storages.journal import JournalFileBackend                   # noqa: E402
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs(A.storage, exist_ok=True)
NOME = "set_%s_%s_%s%s" % (A.arch, A.split, A.target, A.tag)
SAIDA = NOME + ("_kn" if os.environ.get("BLISS_KAPPA_NOSSO") else "")   # outputs only; the study is NOME
STU = optuna.create_study(
    study_name=NOME, storage=JournalStorage(JournalFileBackend("%s/%s.log" % (A.storage, NOME))),
    direction="minimize", load_if_exists=True,
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=12,
        seed=(hash((socket.gethostname(), os.getpid())) & 0x7fffffff)))

if A.modo == "hpo":
    print("%s | %s | alvo %s | Nmax %d | %s" % (A.arch, A.split, A.target, NMAX, DEV), flush=True)
    print('train %d (extra: %s)  val %d  test %d NOT opened'
          % (len(TR), A.extra or "nenhum", len(VA), len(TE)), flush=True)

    def obj(t):
        """Optuna objective: train one configuration and return the validation error.

        Parameters
        ----------
        t : optuna.Trial
            The trial.

        Returns
        -------
        err : float
            Median relative error of kappa on validation (%).
        """
        t0 = time.time()
        try:
            net = treina(t, seed=0)
        except Exception as e:
            print('  trial %3d FAILED %s' % (t.number, str(e)[:70]), flush=True)
            raise optuna.TrialPruned()
        e, q = avalia(net, "va")
        t.set_user_attr("r2", q)
        print("  ensaio %3d  erro %.3f%%  R2 %.4f  (%.0fs)" % (t.number, e, q, time.time() - t0),
              flush=True)
        return e

    import datetime as _dt

    def _validos(s):
        """Trials with a value.

        Parameters
        ----------
        s : optuna.Study
            The study.

        Returns
        -------
        trials : list
            Completed trials.
        """
        return [x for x in s.trials if x.value is not None]

    def _em_voo(s):
        """Trials running with a recent start (dead workers excluded).

        Parameters
        ----------
        s : optuna.Study
            The study.

        Returns
        -------
        trials : list
            Running trials.
        """
        agora = _dt.datetime.now()
        vivos = []
        for x in s.trials:
            if x.state.is_finished() or x.datetime_start is None:
                continue
            if (agora - x.datetime_start).total_seconds() < 3600:
                vivos.append(x)
        return vivos

    def _para_no_alvo(s, _t):
        """Optuna callback: stop when valid plus running trials reach the budget.

        Parameters
        ----------
        s : optuna.Study
            The study.
        _t : optuna.Trial
            The finished trial (unused).
        """
        if len(_validos(s)) + len(_em_voo(s)) >= A.n_trials:
            s.stop()

    feitos = len(_validos(STU))
    print('study %s: %d valid trials, target %d' % (NOME, feitos, A.n_trials), flush=True)
    STU.optimize(obj, n_trials=max(0, 3 * A.n_trials - feitos), callbacks=[_para_no_alvo])
    b = STU.best_trial
    print("\nmelhor: erro %.3f%%  R2 %.4f  %s" % (b.value, b.user_attrs["r2"], json.dumps(b.params)))
    json.dump(dict(study=NOME, arch=A.arch, split=A.split, target=A.target, extra=A.extra,
                   n_train=len(TR), n_trials=len(STU.trials), best_val_err=b.value,
                   best_val_r2=b.user_attrs["r2"], params=b.params),
              open("%s/%s_best.json" % (OUT, NOME), "w"), indent=1)
else:
    b = STU.best_trial
    print('=== confirmation %s | %d seeds | OPENS THE TEST SET ===' % (NOME, A.seeds), flush=True)
    ev, er, por_regime = [], [], {}
    for s in range(A.seeds):
        net = treina(optuna.trial.FixedTrial(b.params), seed=s)
        e, q = avalia(net, "te")
        ev.append(e); er.append(q)
        print("  semente %d  erro %.3f%%  R2 %.4f" % (s, e, q), flush=True)
        with torch.no_grad():
            net.eval(); X, G, M, y, w = prep("te"); p = net(X, G, M)
            p = (p + w) if A.target == "residual" else p
            p, y = p.cpu().numpy(), y.cpu().numpy()
        np.savez_compressed("%s/%s_s%d_test_pred.npz" % (OUT, SAIDA, s), ids=np.array(TE), kappa_true=y, kappa_pred=p)
        if REGISTA_CURVA:
            json.dump(CURVA, open("%s/%s_s%d_curve.json" % (OUT, SAIDA, s), "w"))
        org = DF.loc[TE, "origin"].to_numpy()
        for o in sorted(set(org)):
            m = org == o
            por_regime.setdefault(o, []).append([erro(p[m], y[m]), r2(p[m], y[m]), int(m.sum())])
    ev, er = np.array(ev), np.array(er)
    for o, v in por_regime.items():
        v = np.array(v); print("  %-9s n=%d  erro %.3f%% +- %.3f  R2 %.4f +- %.4f" % (o, v[0, 2], v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 1].std()))
    print("\n  erro %.3f%% +- %.3f   R2 %.4f +- %.4f" % (ev.mean(), ev.std(), er.mean(), er.std()))
    json.dump(dict(study=NOME, arch=A.arch, split=A.split, target=A.target, extra=A.extra,
                   params=b.params, seeds=A.seeds, n_test=len(TE),
                   test_err_mean=float(ev.mean()), test_err_std=float(ev.std()),
                   test_r2_mean=float(er.mean()), test_r2_std=float(er.std()),
                   test_err_per_seed=ev.tolist(), test_r2_per_seed=er.tolist(),
                   por_regime={o: dict(n=int(np.array(v)[0, 2]), err_mean=float(np.array(v)[:, 0].mean()), err_std=float(np.array(v)[:, 0].std()),
                                       r2_mean=float(np.array(v)[:, 1].mean()), r2_std=float(np.array(v)[:, 1].std())) for o, v in por_regime.items()}),
              open("%s/%s_final.json" % (OUT, SAIDA), "w"), indent=1)
