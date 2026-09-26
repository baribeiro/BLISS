"""Phase 2: confirm the best configurations of the search with the full budget and several seeds.

The search (search.py) ranks candidates with a short screening horizon (EPOCHS_TRIAL = 60 epochs, one seed).
Its best trial is not the final configuration: 60 epochs rank candidates but do not evaluate them, and a single
seed confounds the configuration with the initialisation. Phase 2 takes the N_TOP best, trains each with
EPOCHS_FINAL = 200 epochs and N_SEEDS seeds, and only then chooses.

    python benchmark/field_models/confirm.py --model figconv --split v2/random --nodes 8192             (trains the 3 x 3 units)
    python benchmark/field_models/confirm.py --model figconv --split v2/random --nodes 8192 --confirm   (opens the test set)

The winner is chosen by the MEAN over seeds, never by the best seed.

The test set is opened only with `--confirm`, and only for the winning configuration. Phase-2 runs leave a
checkpoint, so `--confirm` resumes at epoch 200, trains nothing, loads the selected EMA weights and evaluates.

The margin between the N_TOP-th configuration and the next one is also reported, since it shows what was left
out by keeping three.
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

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--model", required=True)
ap.add_argument("--split", default="v2/random")
ap.add_argument("--nodes", type=int, default=76805)
ap.add_argument("--select", default="buckling", choices=["buckling", "dimple"])
ap.add_argument("--study-suffix", default="")
ap.add_argument("--confirm", action="store_true", help='open the test set for the winning configuration')
ap.add_argument("--freeze", action="store_true",
                help='write the PLAN: the n_top chosen configurations, once. Without it each worker would recompute the top-N when it picked up its unit.')
ap.add_argument("--n-top", type=int, default=HS.N_TOP)
ap.add_argument("--n-seeds", type=int, default=HS.N_SEEDS)
ap.add_argument("--epochs", type=int, default=HS.EPOCHS_FINAL)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--python", default=os.environ.get("BLISS_PY", sys.executable))
ap.add_argument("--storage", default=OUT + "/optuna")
ap.add_argument("--only", type=int, default=-1,
                help='run only the N-th of the n_top x n_seeds combinations (0..8), so that several workers can share the work without overlapping')
ap.add_argument("--wait-complete", type=float, default=0.0,
                help='hours to wait until the search has n_trials trials with a value before choosing the top-N')
ap.add_argument("--n-trials", type=int, default=HS.N_TRIALS,
                help='number of trials the search should have; only used by --wait-complete')
ap.add_argument("--dry-run", action="store_true")
A = ap.parse_args()

ESTUDO = "%s_%s_n%d_%s%s" % (A.model, A.split.replace("/", ""), A.nodes, A.select, A.study_suffix)
DESTINO = "%s/phase2_%s.json" % (OUT, ESTUDO)
PLANO = "%s/phase2_%s_plano.json" % (OUT, ESTUDO)


def _escreve_atomico(caminho, obj):
    """Write a JSON file atomically (write next to it, then rename).

    Parameters
    ----------
    caminho : str
        Destination path.
    obj : dict
        Object to write.
    """
    tmp = "%s.tmp.%d" % (caminho, os.getpid())
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, caminho)


def _le_plano():
    """The frozen plan: the n_top configurations, fixed once. Every worker reads the same file, so
    rank 1 is the same configuration for everybody.

    Returns
    -------
    plan : dict
        The frozen plan.
    """
    if not os.path.exists(PLANO):
        sys.exit('the frozen plan %s is missing; run --freeze first' % PLANO)
    return json.load(open(PLANO))


def _estudo():
    """Open the Optuna study of the search (journal file).

    Returns
    -------
    study : optuna.Study
        The study.
    """
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    caminho = "%s/%s.log" % (A.storage, ESTUDO)
    if not os.path.exists(caminho):
        sys.exit('study not found: %s' % caminho)
    return optuna.load_study(study_name=ESTUDO,
                             storage=JournalStorage(JournalFileBackend(caminho)))


def _ensaios():
    """Trials of the search with a value, best first.

    Returns
    -------
    trials : list
        Completed trials sorted by validation value.
    """
    return sorted([t for t in _estudo().trials if t.value is not None], key=lambda t: -t.value)


def _gastos():
    """Trials that consume budget, with the same definition as run_optuna: trials with a value plus
    the genuinely pruned ones.

    Returns
    -------
    n : int
        Number of trials that consumed budget.
    """
    import optuna
    n = 0
    for t in _estudo().trials:
        if t.value is not None:
            n += 1
        elif t.state == optuna.trial.TrialState.PRUNED and not t.user_attrs.get("invalid_runtime"):
            n += 1
    return n


def _topo():
    """The N_TOP configurations of the search by validation value, plus the margin to the next one.

    With --wait-complete the function waits until the search has finished.

    Returns
    -------
    top : list
        The N_TOP best trials.
    margin : float
        Value gap between the N_TOP-th and the next trial.
    """
    import time as _t
    ok = _ensaios()
    if A.wait_complete > 0:
        limite = _t.time() + A.wait_complete * 3600
        while _gastos() < A.n_trials and _t.time() < limite:
            print('  search at %d/%d trials used, waiting 5 min' % (_gastos(), A.n_trials),
                  flush=True)
            _t.sleep(300)
        ok = _ensaios()
        if _gastos() < A.n_trials:
            sys.exit('the search stopped at %d of %d trials after %.1f h; not choosing the top-%d of an incomplete study' % (_gastos(), A.n_trials, A.wait_complete, A.n_top))
    if len(ok) < A.n_top:
        sys.exit('the search has %d trials with a value, fewer than the %d requested' % (len(ok), A.n_top))
    margem = (ok[A.n_top - 1].value - ok[A.n_top].value) if len(ok) > A.n_top else float("nan")
    return ok[:A.n_top], margem, len(ok)


def _tag(rank):
    """Tag of the phase-2 runs of one configuration rank.

    Parameters
    ----------
    rank : int
        Rank of the configuration in the plan.

    Returns
    -------
    tag : str
        The run tag.
    """
    return "f%s%d_c%d%s" % (A.select[:4], A.nodes, rank, A.study_suffix)


def _corre(cfg, rank, seed, abre_teste):
    """Train (or, when the test is opened, evaluate) one configuration with one seed through train.py.

    Parameters
    ----------
    cfg : dict
        Hyperparameters.
    rank : int
        Rank of the configuration.
    seed : int
        Training seed.
    abre_teste : bool
        If True, score the test set; otherwise pass --no-test.

    Returns
    -------
    record : dict
        The results.jsonl record of the run.
    """
    tag = _tag(rank)
    cmd = [A.python, ML + "/train.py", A.model, A.split,
           "--epochs", str(A.epochs), "--bs", str(A.bs), "--select", A.select,
           "--tag", tag, "--seed", str(seed)] + HS.as_args(A.model, cfg)
    if A.nodes != 8192:
        cmd += ["--nodes", str(A.nodes)]
    # BLISS_TRAIN_EXTRA: extra train flags for a whole campaign (e.g. "--grid 64x128" for the SFNO
    # matched-grid search). Empty by default, so every existing campaign is unchanged.
    import shlex; cmd += shlex.split(os.environ.get("BLISS_TRAIN_EXTRA", ""))
    if not abre_teste:
        cmd += ["--no-test"]
    print("  >> %s" % " ".join(cmd[1:]), flush=True)
    if A.dry_run:
        return None
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print('  !! exited with code %d' % r.returncode, flush=True)
    return _le_registo(tag, seed)


def _le_registo(tag, seed):
    """The last results.jsonl record with this tag and this seed.

    The tag is built exactly as train.py builds it.

    Parameters
    ----------
    tag : str
        Run tag.
    seed : int
        Training seed.

    Returns
    -------
    record : dict or None
        The last matching record.
    """
    esperado = ("%s_%s" % (A.model, A.split.replace("/", "")) + "_" + tag
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


if A.freeze:
    if os.path.exists(PLANO):
        print('plan already frozen in %s; leaving it' % PLANO)
        sys.exit(0)
    topo, margem, n_ok = _topo()
    _escreve_atomico(PLANO, dict(
        study=ESTUDO, model=A.model, split=A.split, nodes=A.nodes, select=A.select,
        n_top=A.n_top, n_seeds=A.n_seeds, epochs=A.epochs, n_search_trials=n_ok,
        search_margin_top_to_next=margem,
        configs=[dict(rank=i, params=t.params, trial_value=t.value) for i, t in enumerate(topo)]))
    print('=== frozen plan: %s ===' % PLANO)
    for i, t in enumerate(topo):
        print("  %d.  val %.4f   %s" % (i, t.value, json.dumps(t.params)))
    print('  margin between rank %d and rank %d: %.4f' % (A.n_top, A.n_top + 1, margem))
    sys.exit(0)

_pl = _le_plano()
topo = [type("C", (), {"params": c["params"], "value": c["trial_value"]})() for c in _pl["configs"]]
margem, n_ok = _pl["search_margin_top_to_next"], _pl["n_search_trials"]
print('=== %s | frozen plan, %d trials in the search, %d configurations ==='
      % (ESTUDO, n_ok, len(topo)), flush=True)

COMBS = [(rank, seed) for rank in range(A.n_top) for seed in range(A.n_seeds)]

if not A.confirm:
    # --------------------------------------------------- treinar as N_TOP com orcamento cheio
    if A.only >= 0:
        if A.only >= len(COMBS):
            print('  nothing to do: --only %d outside the %d combinations' % (A.only, len(COMBS)))
            sys.exit(0)
        rank, seed = COMBS[A.only]
        print('\n--- only combination %d: config %d, seed %d, %d epochs ---'
              % (A.only, rank, seed, A.epochs), flush=True)
        r = _corre(topo[rank].params, rank, seed, abre_teste=False)
        if not r:
            sys.exit('the run wrote no record: unit %d NOT done' % A.only)
        print("  val_selected %.4f" % r["val_selected"], flush=True)
        sys.exit(0)
    resultados = []
    for rank, t in enumerate(topo):
        vals = []
        for seed in range(A.n_seeds):
            print('\n--- config %d, seed %d, %d epochs ---' % (rank, seed, A.epochs), flush=True)
            r = _corre(t.params, rank, seed, abre_teste=False)
            if r:
                vals.append(r["val_selected"])
                print("  val_selected %.4f" % r["val_selected"], flush=True)
        if vals:
            resultados.append(dict(rank=rank, params=t.params, trial_value=t.value,
                                   val_per_seed=vals, val_mean=float(np.mean(vals)),
                                   val_std=float(np.std(vals))))
    if not resultados:
        sys.exit('no run wrote a record')
    resultados.sort(key=lambda x: -x["val_mean"])
    print("\n=== fase 2: %s ===" % ESTUDO)
    print("%5s %10s %10s %10s   %s" % ("rank", "val medio", "desvio", "busca", "config"))
    for r in resultados:
        print("%5d %10.4f %10.4f %10.4f   %s"
              % (r["rank"], r["val_mean"], r["val_std"], r["trial_value"], json.dumps(r["params"])))
    v = resultados[0]
    print('\nwinner: config %d, mean val %.4f +- %.4f over %d seeds'
          % (v["rank"], v["val_mean"], v["val_std"], A.n_seeds))
    if len(resultados) > 1:
        print('margin to the second: %.4f' % (v["val_mean"] - resultados[1]["val_mean"]))
    _escreve_atomico(DESTINO, dict(study=ESTUDO, model=A.model, split=A.split, nodes=A.nodes,
                   select=A.select, epochs=A.epochs, n_top=A.n_top, n_seeds=A.n_seeds,
                   search_margin_top_to_next=margem, n_search_trials=n_ok,
                   results=resultados, winner_rank=v["rank"], winner_params=v["params"],
                   selection_rule="maior media de val_selected sobre as sementes"))
    print("\nescrito em %s" % DESTINO)
    print('to open the test set: confirm.py ... --confirm')
else:
    if not os.path.exists(DESTINO):
        print('  no %s; rebuilding the choice from the run records' % DESTINO,
              flush=True)
        resultados = []
        for rank, t in enumerate(topo):
            vals = [r["val_selected"] for r in
                    (_le_registo(_tag(rank), sd) for sd in range(A.n_seeds)) if r]
            if len(vals) < A.n_seeds:
                sys.exit('config %d has %d of %d seeds done; the confirmation waits for the others' % (rank, len(vals), A.n_seeds))
            resultados.append(dict(rank=rank, params=t.params, trial_value=t.value,
                                   val_per_seed=vals, val_mean=float(np.mean(vals)),
                                   val_std=float(np.std(vals))))
        resultados.sort(key=lambda x: -x["val_mean"])
        v = resultados[0]
        _escreve_atomico(DESTINO, dict(study=ESTUDO, model=A.model, split=A.split, nodes=A.nodes,
                       select=A.select, epochs=A.epochs, n_top=A.n_top, n_seeds=A.n_seeds,
                       search_margin_top_to_next=margem, n_search_trials=n_ok,
                       results=resultados, winner_rank=v["rank"], winner_params=v["params"],
                       selection_rule="maior media de val_selected sobre as sementes"))
        print('  winner: config %d, mean val %.4f +- %.4f'
              % (v["rank"], v["val_mean"], v["val_std"]), flush=True)
    d = json.load(open(DESTINO))
    rank = d["winner_rank"]
    print('\n=== CONFIRMATION: config %d, %d seeds. THE TEST SET OPENS HERE. ==='
          % (rank, A.n_seeds), flush=True)
    print('    (the runs exist; resumes at epoch %d, no training, evaluation only)' % A.epochs,
          flush=True)
    linhas = []
    for seed in range(A.n_seeds):
        r = _corre(d["winner_params"], rank, seed, abre_teste=True)
        if r:
            linhas.append(r)
            print('  seed %d  test buckling_dimple_r2 %.4f  kappa_r2 %.4f'
                  % (seed, r["test_buckling_dimple_r2"], r["test_kappa_r2"]), flush=True)
    if not linhas:
        sys.exit('no confirmation run wrote a record')
    chaves = ["test_buckling_dimple_r2", "test_dimple_r2", "test_field_r2", "test_kappa_r2",
              "test_buckling_r2", "test_loc_median", "test_loc10"]
    resumo = {}
    print("\n%-26s %10s %10s" % ("metrica", "media", "desvio"))
    for k in chaves:
        v = np.array([x.get(k, np.nan) for x in linhas], float)
        resumo[k] = dict(mean=float(np.nanmean(v)), std=float(np.nanstd(v)), per_seed=v.tolist())
        print("%-26s %10.4f %10.4f" % (k, resumo[k]["mean"], resumo[k]["std"]))
    d["test"] = resumo
    d["test_n_seeds"] = len(linhas)
    _escreve_atomico(DESTINO, d)
    print("\nactualizado em %s" % DESTINO)
