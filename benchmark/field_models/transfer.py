"""Phases 4, 5 and 6: the protocols B2 and B3 and the resolution ladder, with FROZEN hyperparameters.

These phases are not a search. The configuration of each model was chosen in phase 2 on B1, and here only the
DATA change: the split (B2 moves to deeper defects, B3 to dense shells) or the input resolution. Re-choosing the
hyperparameters on each split would measure the ability to re-tune rather than transfer.

    python benchmark/field_models/transfer.py --model figconv --phase B2 --only 0        (seed 0, 200 epochs)
    python benchmark/field_models/transfer.py --model figconv --phase scale --nodes 32768 --only 1
    python benchmark/field_models/transfer.py --model figconv --phase B2 --confirm       (opens the test set, 3 seeds)

The configuration is read from the file phase 2 wrote, phase2_<study>.json, field `winner_params`
(the files used in the paper are in configs/). If the file does not exist the script stops.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ML = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ML)
import search_space as HS                                                    # noqa: E402

B = os.path.dirname(os.path.dirname(ML))
OUT = B + "/outputs"
CORTE = {"B1": "v2/random", "B2": "v2/delta", "B3": "v2/regime"}

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", required=True)
ap.add_argument("--phase", required=True, choices=["B2", "B3", "scale"])
ap.add_argument("--nodes", type=int, default=0, help='only for --phase scale')
ap.add_argument("--only", type=int, default=-1, help='one seed (0..n_seeds-1)')
ap.add_argument("--confirm", action="store_true", help='open the test set for the seeds already trained')
ap.add_argument("--n-seeds", type=int, default=HS.N_SEEDS)
ap.add_argument("--epochs", type=int, default=HS.EPOCHS_FINAL)
ap.add_argument("--select", default="buckling", choices=["buckling", "dimple"])
ap.add_argument("--study-suffix", default="",
                help='the SAME suffix phase 2 used. The name of the configuration file is built from it, as from --split-b1.')
ap.add_argument("--split-b1", default="v2/random",
                help='the split on which phase 2 chose. It must be the same as in the phase-2 plan, because the name of the configuration file is built from it.')
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--python", default=os.environ.get("BLISS_PY", sys.executable))
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()

RES_B1 = int(os.environ["NODES"]) if os.environ.get("NODES") else \
         (8192 if A.model in ("transformer", "deeponet") else 76805)
NODES = A.nodes if A.phase == "scale" else RES_B1
SPLIT = A.split_b1 if A.phase == "scale" else CORTE[A.phase]
if A.phase == "scale" and A.nodes <= 0:
    sys.exit('--phase scale needs --nodes')

PLANO = "%s/phase2_%s_%s_n%d_%s%s.json" % (OUT, A.model, A.split_b1.replace("/", ""), RES_B1, A.select, A.study_suffix)
if not os.path.exists(PLANO):
    sys.exit('the phase-2 choice is missing (%s). This phase freezes that configuration and does not choose one itself.' % PLANO)
D = json.load(open(PLANO))
CFG = D["winner_params"]
if "test" not in D:
    print('warning: phase 2 has not confirmed this model yet; using the validation winner',
          flush=True)


def _tag():
    """Tag of the runs of this phase.

    Returns
    -------
    tag : str
        'pb2buck', 'pb3buck' or 'sc<nodes><select>'.
    """
    if A.phase == "scale":
        return "sc%d%s" % (NODES, A.select[:4])
    return "p%s%s" % (A.phase.lower(), A.select[:4])


def _registo(seed):
    """The results.jsonl record of this run, with the tag built as train.py builds it.

    Parameters
    ----------
    seed : int
        Training seed.

    Returns
    -------
    record : dict or None
        The last matching record.
    """
    esperado = ("%s_%s_%s" % (A.model, SPLIT.replace("/", ""), _tag())
                + ("_s%d" % seed if seed else ""))
    achado = None
    for linha in open(OUT + "/results.jsonl"):
        try:
            r = json.loads(linha)
        except Exception:
            continue
        if r.get("tag") == esperado and r.get("seed") == seed:
            achado = r
    return achado


def _corre(seed, abre_teste):
    """Train (or, when the test is opened, evaluate) the frozen configuration with one seed through train.py.

    Parameters
    ----------
    seed : int
        Training seed.
    abre_teste : bool
        If True, score the test set; otherwise pass --no-test.

    Returns
    -------
    record : dict
        The results.jsonl record of the run.
    """
    cmd = [A.python, ML + "/train.py", A.model, SPLIT,
           "--epochs", str(A.epochs), "--bs", str(A.bs), "--select", A.select,
           "--tag", _tag(), "--seed", str(seed)] + HS.as_args(A.model, CFG)
    if NODES != 8192:
        cmd += ["--nodes", str(NODES)]
    # BLISS_TRAIN_EXTRA: extra train flags for a whole campaign (as in search.py / confirm.py); empty = unchanged.
    import shlex; cmd += shlex.split(os.environ.get("BLISS_TRAIN_EXTRA", ""))
    if not abre_teste:
        cmd += ["--no-test"]
    print("  >> %s" % " ".join(cmd[1:]), flush=True)
    if A.dry_run:
        return None
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print('  !! training exited with code %d' % rc, flush=True)
    r = _registo(seed)
    if abre_teste and r is not None:
        v = r.get("test_buckling_dimple_r2")
        if v is None or v != v:
            print('  !! record without test (NaN): the confirmation is NOT done', flush=True)
            return None
    return r


print('=== %s | phase %s | %s | %d nodes | frozen phase-2 config ==='
      % (A.model, A.phase, SPLIT, NODES), flush=True)
print("   %s" % json.dumps(CFG), flush=True)

if not A.confirm:
    if A.only < 0 or A.only >= A.n_seeds:
        sys.exit('--only must be a seed between 0 and %d' % (A.n_seeds - 1))
    r = _corre(A.only, abre_teste=False)
    if not r:
        sys.exit('the run wrote no record: seed %d NOT done' % A.only)
    print("  val_selected %.4f" % r["val_selected"], flush=True)
else:
    linhas = []
    for s in range(A.n_seeds):
        if _registo(s) is None:
            sys.exit('seed %d not trained yet; the confirmation waits for all three' % s)
    print('=== CONFIRMATION: %s phase %s, %d seeds. THE TEST SET OPENS HERE. ==='
          % (A.model, A.phase, A.n_seeds), flush=True)
    for s in range(A.n_seeds):
        r = _corre(s, abre_teste=True)
        if r:
            linhas.append(r)
            print("  semente %d  teste buckling_dimple_r2 %.4f  kappa_r2 %.4f"
                  % (s, r["test_buckling_dimple_r2"], r["test_kappa_r2"]), flush=True)
    if not linhas:
        sys.exit('no confirmation wrote a record')
    chaves = ["test_buckling_dimple_r2", "test_dimple_r2", "test_field_r2", "test_kappa_r2",
              "test_buckling_r2", "test_loc_median", "test_loc10"]
    res = {}
    print("\n%-26s %10s %10s" % ("metrica", "media", "desvio"))
    for k in chaves:
        v = np.array([x.get(k, np.nan) for x in linhas], float)
        res[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)), per_seed=v.tolist())
        print("%-26s %10.4f %10.4f" % (k, res[k]["mean"], res[k]["std"]))
    dest = "%s/phase%s_%s_n%d_%s.json" % (OUT, A.phase, A.model, NODES, A.select)
    tmp = dest + ".tmp%d" % os.getpid()
    json.dump(dict(model=A.model, phase=A.phase, split=SPLIT, nodes=NODES, seeds=A.n_seeds,
                   frozen_params=CFG, frozen_from=os.path.basename(PLANO), test=res),
              open(tmp, "w"), indent=1)
    os.replace(tmp, dest)
    print("\nescrito em %s" % dest)
