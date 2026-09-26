"""Tabular models on the list of defects: how much of the knockdown factor the descriptors hold, without the mesh.

This is the reference the geometry models must beat to justify reading the mesh, so it is made as strong as
the protocol allows: several model families, two representations, three ways of padding the empty slots, the
three splits, a long TPE search, and the winning configuration confirmed with ten seeds.

    python benchmark/parametric/tabular.py hpo   --split B1 --rep topk [--families trees|mlp|all] [--device cuda]
    python benchmark/parametric/tabular.py final --split B1 --rep topk        (10 seeds, opens the test set)
    python benchmark/parametric/tabular.py curve --split B1 --rep topk        (learning curve)

`hpo` never touches the test set. Only `final` and `curve` open it, once, with the configuration frozen on
validation. The descriptors are in defect_features.py.

Functions
---------
build
    One model family per trial, from the reference set in tabular_models.py.
erro
    The objective: median relative error of kappa, in percent.
r2
    Coefficient of determination of kappa.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import defect_features as F                                                     # noqa: E402
import tabular_models as TM                                                  # noqa: E402

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = B + "/outputs"

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("modo", choices=["hpo", "final", "curve"])
ap.add_argument("--split", default="B1", choices=["B1", "B2", "B3"])
ap.add_argument("--rep", default="topk", choices=["phys", "topk", "interac", "adim"])
ap.add_argument("--pad", default="search", choices=list(F.PAD) + ["search"],
                help="how to fill the unused slots: zero, mask (zero + presence bit) or nan. 'search' lets validation choose.")
ap.add_argument("--families", default="tudo", choices=sorted(TM.GRUPOS),
                help="group of families: 'tudo' (all), 'chao' (floor), 'arvores' (trees), 'redes' (networks), 'gpu_arvores', 'trees' (CPU-friendly) or 'mlp'")
ap.add_argument("--device", default="cpu",
                help='cpu or cuda. Affects the MLP, xgboost and lightgbm; the sklearn families always run on CPU.')
ap.add_argument("--n-trials", type=int, default=200)
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--topk", default="search",
                help="number of defect slots: an integer, or 'search' to let validation choose among %s. Only affects the topk representation." % (F.KS,))
ap.add_argument("--tag", default="", help='study suffix, to separate workers')
ap.add_argument("--storage", default=OUT + "/optuna")
ap.add_argument("--frozen-params", default="", help="final: use the params of this *_final.json (e.g. the B1 choice) "
                "instead of this split's study; refit on this split's train+val, test opened once")
A = ap.parse_args()

PADS = list(F.PAD) if A.pad == "search" else [A.pad]
FAMILIAS = None
KSS = ([1] if A.rep == "phys" else
       (list(F.KS) if A.topk == "search" else [int(A.topk)]))

# ---------------------------------------------------------------- dados
sp = json.load(open("%s/BLISS-1.0/splits/%s.json" % (B, A.split)))
sel = lambda k: [i for i in sp[k] if i in F.DEF and i in F.KD]
TR, VA, TE = sel("train"), sel("val"), sel("test")
ytr, yva, yte = F.alvo(TR), F.alvo(VA), F.alvo(TE)
MAT = {(q, k): dict(tr=F.matriz(TR, A.rep, k, q), va=F.matriz(VA, A.rep, k, q),
                    te=F.matriz(TE, A.rep, k, q)) for q in PADS for k in KSS}
NOMES = {(q, k): F.nomes(A.rep, k, q) for q in PADS for k in KSS}
_DMAX = max(v["tr"].shape[1] for v in MAT.values())
FAMILIAS = TM.familias(A.families, _DMAX, len(TR))
_FORA = sorted(set(TM.GRUPOS[A.families]) - set(FAMILIAS))


# ---------------------------------------------------------------- familias
def build(t, seed=0):
    """One model family per trial, from the reference set in tabular_models.py.

    Parameters
    ----------
    t : optuna.Trial
        The trial.
    seed : int
        Seed.

    Returns
    -------
    model : object
        An object with fit and predict.
    """
    fam = t.suggest_categorical("fam", FAMILIAS)
    return TM.build(fam, t, seed=seed, device=A.device,
                    n_features=_DMAX, n_train=len(TR))


def erro(p, y):
    """The objective: median relative error of kappa, in percent.

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


# ---------------------------------------------------------------- optuna
import optuna                                                            # noqa: E402
from optuna.storages import JournalStorage                               # noqa: E402
from optuna.storages.journal import JournalFileBackend                   # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
os.makedirs(A.storage, exist_ok=True)
NOME = "oracle_%s_%s%s" % (A.split, A.rep, A.tag)
SAIDA = NOME + ("_kn" if os.environ.get("BLISS_KAPPA_NOSSO") else "")   # outputs only; the study (and its hyperparameters) is NOME
STU = optuna.create_study(
    study_name=NOME, storage=JournalStorage(JournalFileBackend("%s/%s.log" % (A.storage, NOME))),
    direction="minimize", load_if_exists=True,
    sampler=optuna.samplers.TPESampler(n_startup_trials=25, seed=0))


def _melhor(chave):
    """Best validation value per value of `chave`, for the ablation tables.

    Parameters
    ----------
    chave : str
        Name of a hyperparameter.

    Returns
    -------
    best : dict
        Value of the hyperparameter -> best validation error.
    """
    d = {}
    for t in STU.trials:
        if t.value is not None and chave in t.params:
            k = t.params[chave]
            d[k] = min(d.get(k, float("inf")), t.value)
    return d


if A.modo == "hpo":
    print('%s | %s | on %s | padding: %s | slots: %s'
          % (A.split, A.rep, A.device, ", ".join(PADS), ", ".join(str(x) for x in KSS)), flush=True)
    print("familias (%d): %s" % (len(FAMILIAS), " ".join(FAMILIAS)), flush=True)
    if _FORA:
        print('left out, too large for this data set: %s' % " ".join(_FORA), flush=True)
    print('train %d  val %d  (test %d NOT opened)  | %d descriptors'
          % (len(TR), len(VA), len(TE), MAT[(PADS[0], KSS[-1])]["tr"].shape[1]), flush=True)
    _sd = np.nanstd(MAT[(PADS[0], KSS[-1])]["tr"], 0)
    _c = [n for n, v in zip(NOMES[(PADS[0], KSS[-1])], _sd) if v < 1e-12]
    if _c:
        print('WARNING: %d descriptors have ZERO variance in training (%s%s). The model cannot learn to depend on them.'
              % (len(_c), ", ".join(_c[:8]), " ..." if len(_c) > 8 else ""), flush=True)

    def objectivo(t):
        """Optuna objective: fit one configuration on the training shells and return the validation error.

        Parameters
        ----------
        t : optuna.Trial
            The trial.

        Returns
        -------
        err : float
            Median relative error of kappa on validation (%).
        """
        q = t.suggest_categorical("pad", PADS)
        k = t.suggest_categorical("k", KSS)
        t0 = time.time()
        try:
            m = build(t).fit(MAT[(q, k)]["tr"], ytr)
        except Exception as e:
            print('  trial %3d  FAILED  %s' % (t.number, str(e)[:70]), flush=True)
            raise optuna.TrialPruned()
        p = m.predict(MAT[(q, k)]["va"])
        e = erro(p, yva)
        t.set_user_attr("r2", r2(p, yva))
        t.set_user_attr("fit_s", round(time.time() - t0, 1))
        melhor = min([x.value for x in t.study.trials if x.value is not None] + [float("inf")])
        if t.number % 10 == 0 or e <= melhor:
            print("  ensaio %3d  %-9s %-5s K=%-2d  erro %.3f%%  R2 %.4f  (%.0fs)"
                  % (t.number, t.params["fam"], q, k, e, t.user_attrs["r2"], time.time() - t0),
                  flush=True)
        return e

    feitos = len([t for t in STU.trials if t.value is not None])
    STU.optimize(objectivo, n_trials=max(0, A.n_trials - feitos))
    b = STU.best_trial
    porfam, porpad, pork = _melhor("fam"), _melhor("pad"), _melhor("k")
    print('\n--- best per family (validation) ---')
    for k, v in sorted(porfam.items(), key=lambda x: x[1]):
        print("  %-9s %.3f%%" % (k, v))
    print('--- best per padding (validation) ---')
    for k, v in sorted(porpad.items(), key=lambda x: x[1]):
        print("  %-9s %.3f%%" % (k, v))
    if len(KSS) > 1:
        print('--- best per number of slots (validation) ---')
        for k, v in sorted(pork.items(), key=lambda x: x[1]):
            print("  K=%-7s %.3f%%" % (k, v))
    print("\nmelhor global: %s / %s / K=%s  erro %.3f%%  R2 %.4f"
          % (b.params["fam"], b.params["pad"], b.params["k"], b.value, b.user_attrs["r2"]))
    json.dump(dict(study=NOME, split=A.split, rep=A.rep, n_trials=len(STU.trials),
                   best_val_err=b.value, best_val_r2=b.user_attrs["r2"], params=b.params,
                   best_per_family=porfam, best_per_pad=porpad, best_per_k=pork,
                   n_features=MAT[(b.params["pad"], b.params["k"])]["tr"].shape[1],
                   features=NOMES[(b.params["pad"], b.params["k"])]),
              open("%s/%s_best.json" % (OUT, NOME), "w"), indent=1)

elif A.modo == "final":
    if A.frozen_params:
        _fz = json.load(open(A.frozen_params))
        b = type("Frozen", (), dict(params=_fz["params"], value=float("nan")))()
        SAIDA = SAIDA + "_frozen_" + os.path.basename(A.frozen_params).split("_topk")[0].replace("oracle_", "")
        print("parametros congelados de %s: %s" % (A.frozen_params, _fz["params"]), flush=True)
    else:
        b = STU.best_trial
    q, kk = b.params["pad"], b.params["k"]
    print('=== confirmation: %s %s | %s / %s | %d seeds | OPENS THE TEST SET ==='
          % (A.split, A.rep, b.params["fam"], "%s/K=%s" % (q, kk), A.seeds), flush=True)
    Xf = np.vstack([MAT[(q, kk)]["tr"], MAT[(q, kk)]["va"]])
    yf = np.r_[ytr, yva]
    ev, er, PREVS = [], [], []
    for s in range(A.seeds):
        m = build(optuna.trial.FixedTrial(b.params), seed=s).fit(Xf, yf)
        p = m.predict(MAT[(q, kk)]["te"])
        PREVS.append(p)
        ev.append(erro(p, yte))
        er.append(r2(p, yte))
        print("  semente %d  erro %.3f%%  R2 %.4f" % (s, ev[-1], er[-1]), flush=True)
    ev, er = np.array(ev), np.array(er)
    print("\n  erro %.3f%% +- %.3f   R2 %.4f +- %.4f" % (ev.mean(), ev.std(), er.mean(), er.std()))
    np.savez("%s/%s_pred.npz" % (OUT, SAIDA), ids=np.array(TE), true_K=yte,
             pred_K=np.array(PREVS).mean(0), pred_K_per_seed=np.array(PREVS))
    json.dump(dict(split=A.split, rep=A.rep, family=b.params["fam"], pad=q, k=kk, seeds=A.seeds,
                   params=b.params, val_err=b.value, n_test=len(TE), frozen_from=A.frozen_params or None,
                   test_err_mean=float(ev.mean()), test_err_std=float(ev.std()),
                   test_r2_mean=float(er.mean()), test_r2_std=float(er.std()),
                   test_err_per_seed=ev.tolist(), test_r2_per_seed=er.tolist()),
              open("%s/%s_final.json" % (OUT, SAIDA), "w"), indent=1)

else:
    b = STU.best_trial
    q, kk = b.params["pad"], b.params["k"]
    print("=== curva de aprendizagem: %s %s | %s / %s ===" % (A.split, A.rep, b.params["fam"], q),
          flush=True)
    linhas = []
    Xtr = MAT[(q, kk)]["tr"]
    for frac in (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        n = max(30, int(frac * len(Xtr)))
        e, r = [], []
        for s in range(5):
            idx = np.random.default_rng(2027 + s).permutation(len(Xtr))[:n]
            m = build(optuna.trial.FixedTrial(b.params), seed=s).fit(Xtr[idx], ytr[idx])
            p = m.predict(MAT[(q, kk)]["te"])
            e.append(erro(p, yte))
            r.append(r2(p, yte))
        linhas.append(dict(frac=frac, n=n, err=float(np.mean(e)), err_sd=float(np.std(e)),
                           r2=float(np.mean(r)), r2_sd=float(np.std(r))))
        print("  n=%5d  erro %.3f%% +- %.3f   R2 %.4f +- %.4f"
              % (n, np.mean(e), np.std(e), np.mean(r), np.std(r)), flush=True)
    json.dump(dict(split=A.split, rep=A.rep, family=b.params["fam"], pad=q, k=kk, curve=linhas),
              open("%s/%s_curve.json" % (OUT, SAIDA), "w"), indent=1)
