"""Hyperparameter search driver (Optuna, TPE) for the field models of BLISS.

One worker per GPU. All workers share one study through a journal file on a shared filesystem, so workers can
join and leave at any time without corrupting it (SQLite is avoided because it corrupts on network filesystems).

    python benchmark/field_models/search.py --model figconv --split v2/random
    python benchmark/field_models/search.py --model figconv --n-trials 40           (several workers in parallel, same command)

The objective is buckling_dimple_r2 on VALIDATION, the published field metric (uniform contraction removed,
scalar reference). A trial that fails (NaN, out of memory, invalid configuration) is marked as failed and does
not affect the study; a trial that converges slowly is not pruned.

Functions
---------
marca_resultados
    The size of results.jsonl BEFORE the training run starts, so that the record written by
last_record
    The record written by THIS run, found by its tag and read from the end of the file.
run_trial
    Run one trial: draw a configuration, train it with train.py and return its validation score.
main
    Parse the options, open the shared study and run trials until the budget is reached.
"""
import argparse, json, os, subprocess, sys, time

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML = B + "/benchmark/field_models"
sys.path.insert(0, ML)
import search_space as HS                                                      # noqa: E402


RESULTADOS = None    # definido em run_trial; caminho do results.jsonl


def marca_resultados():
    """The size of results.jsonl BEFORE the training run starts, so that the record written by
    this run can be told apart from an older record with the same tag.

    Returns
    -------
    size : int
        Size in bytes of outputs/results.jsonl (0 if absent).
    """
    try:
        return os.path.getsize(B + "/outputs/results.jsonl")
    except OSError:
        return 0


def last_record(tag, depois=0):
    """The record written by THIS run, found by its tag and read from the end of the file.

    `depois` is the mark returned by marca_resultados() before training started.

    Parameters
    ----------
    tag : str
        Tag of the run.
    depois : int
        Mark returned by marca_resultados().

    Returns
    -------
    record : dict or None
        The last matching record written after the mark.
    """
    caminho = B + "/outputs/results.jsonl"
    try:
        with open(caminho) as f:
            f.seek(depois)
            novas = f.readlines()
    except Exception:
        return None
    for line in reversed(novas):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("tag") == tag:
            return r
    return None


def run_trial(trial, A):
    """Run one trial: draw a configuration, train it with train.py and return its validation score.

    Parameters
    ----------
    trial : optuna.Trial
        The trial.
    A : argparse.Namespace
        Command-line options.

    Returns
    -------
    value : float
        Validation buckling_dimple_r2 of the selected checkpoint.

    Raises
    ------
    optuna.TrialPruned
        If the run is pruned or fails.
    """
    cfg = HS.suggest(trial, A.model)
    tag = "%s%d_t%d%s" % (A.select[:4], A.nodes, trial.number, A.study_suffix)
    cmd = [A.python, ML + "/train.py", A.model, A.split,
           "--epochs", str(A.epochs), "--bs", "8", "--select", A.select, "--no-test", "--tag", tag] + HS.as_args(A.model, cfg)
    if A.nodes != 8192: cmd += ["--nodes", str(A.nodes)]
    # BLISS_TRAIN_EXTRA: extra train flags for a whole campaign (e.g. "--grid 64x128" for the SFNO
    # matched-grid search). Empty by default, so every existing campaign is unchanged.
    import shlex; cmd += shlex.split(os.environ.get("BLISS_TRAIN_EXTRA", ""))
    t0 = time.time()
    import re
    _marca = marca_resultados()
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    curva, cauda, morto = [], [], False
    for linha in pr.stdout:
        cauda.append(linha)
        if len(cauda) > 40: cauda.pop(0)
        m = re.match(r"^ep\s+(\d+).*?buckling\s+(-?[\d.]+)", linha)
        if not m: continue
        ep, v = int(m.group(1)), float(m.group(2))
        curva.append((ep, v))
        trial.report(v, ep)
        if trial.should_prune():
            pr.kill(); morto = True; trial.set_user_attr("pruned_by", "median")
            print('trial %3d pruned at epoch %d (%.4f)' % (trial.number, ep, v), flush=True)
            break
    pr.wait()
    trial.set_user_attr("curve", curva)
    if morto: raise optuna.TrialPruned()
    p = type("P", (), {"stdout": "".join(cauda), "stderr": ""})()
    r = last_record("%s_%s_%s" % (A.model, A.split.replace("/", ""), tag), depois=_marca)
    if r is None:
        tail = (p.stderr or p.stdout or "")[-400:].replace("\n", " | ")
        print('trial %d FAILED (%.0f s): %s' % (trial.number, time.time() - t0, tail), flush=True)
        trial.set_user_attr("invalid_runtime", True); trial.set_user_attr("invalid_reason", tail[-200:])
        raise optuna.TrialPruned()
    for k in ("val_kappa_r2", "val_dimple_r2", "val_buckling_dimple_r2", "params_M", "train_s", "best_epoch"):
        if k in r: trial.set_user_attr(k, r[k])
    trial.set_user_attr("config", cfg)
    if r.get("params_M", 0) > HS.MAX_PARAMS_M:
        print('trial %d above the size cap: %.1f M > %.1f M' % (trial.number, r["params_M"], HS.MAX_PARAMS_M), flush=True)
        trial.set_user_attr("over_param_cap", True)
        raise optuna.TrialPruned()
    print('trial %3d  %-11s val buckling %.4f  (%.0f min, %.2f M params)'
          % (trial.number, A.model, r["val_selected"], (time.time() - t0) / 60, r.get("params_M", 0)), flush=True)
    return r["val_selected"]


def main():
    """Parse the options, open the shared study and run trials until the budget is reached."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(HS.SPACE))
    ap.add_argument("--split", default="v2/random")
    ap.add_argument("--nodes", type=int, default=8192)
    ap.add_argument("--n-trials", type=int, default=HS.N_TRIALS)
    ap.add_argument("--max-new", type=int, default=0,
                    help='run at most N trials in this call and exit. Useful for queues with a short wall time.')
    ap.add_argument("--epochs", type=int, default=HS.EPOCHS_TRIAL)
    ap.add_argument("--storage", default=B + "/outputs/optuna")
    ap.add_argument("--study-suffix", default="")
    ap.add_argument("--select", default="buckling", choices=["buckling", "dimple"],
                    help='the objective, which also selects the checkpoint within each run')
    ap.add_argument("--python", default=os.environ.get("BLISS_PY", sys.executable))
    A = ap.parse_args()

    global optuna
    import optuna
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    import socket
    _SEED = (hash((socket.gethostname(), os.getpid())) & 0x7fffffff)
    os.makedirs(A.storage, exist_ok=True)
    name = "%s_%s_n%d_%s%s" % (A.model, A.split.replace("/", ""), A.nodes, A.select, A.study_suffix)
    st = JournalStorage(JournalFileBackend("%s/%s.log" % (A.storage, name)))
    study = optuna.create_study(study_name=name, storage=st, direction="maximize",
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(n_startup_trials=HS.N_STARTUP, seed=_SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=HS.N_STARTUP,
                                                                   n_warmup_steps=30, interval_steps=5))
    import datetime as _dt
    MORTO_H = 24

    def em_voo(stu):
        """Trials actually running. A dead worker leaves its trial in RUNNING forever, so only
        recent heartbeats are counted.

        Parameters
        ----------
        stu : optuna.Study
            The study.

        Returns
        -------
        trials : list
            Trials started within the last MORTO_H hours and not finished.
        """
        agora = _dt.datetime.now()
        return [t for t in stu.trials if not t.state.is_finished()
                and t.datetime_start and (agora - t.datetime_start).total_seconds() < MORTO_H * 3600]

    def validos(stu):
        """Trials that count towards the budget. A median pruning counts (the trial ran and was
        legitimately stopped); an execution failure does not.

        Parameters
        ----------
        stu : optuna.Study
            The study.

        Returns
        -------
        trials : list
            Trials that count towards the budget.
        """
        return [t for t in stu.trials
                if t.value is not None or (t.state.name == "PRUNED"
                                           and not t.user_attrs.get("invalid_runtime"))]

    def stop_at_target(stu, _t):
        """Optuna callback: stop the study when the valid and running trials reach the budget.

        Parameters
        ----------
        stu : optuna.Study
            The study.
        _t : optuna.Trial
            The finished trial (unused).
        """
        if len(validos(stu)) + len(em_voo(stu)) >= A.n_trials:
            stu.stop()

    _z = [t for t in study.trials if not t.state.is_finished()]
    if _z:
        print('%d trials RUNNING in the journal; %d are recent, the others belong to dead workers'
              % (len(_z), len(em_voo(study))), flush=True)
    done = len(validos(study))
    maus = len([t for t in study.trials if t.user_attrs.get("invalid_runtime")])
    print('study %s: %d valid trials, target %d%s'
          % (name, done, A.n_trials,
             '  (%d execution failures, not counted)' % maus if maus else ""), flush=True)

    n = max(0, 3 * A.n_trials - done)
    if A.max_new: n = min(n, A.max_new)
    study.optimize(lambda t: run_trial(t, A), n_trials=n, callbacks=[stop_at_target], catch=())
    ok = [t for t in study.trials if t.value is not None]
    if ok:
        b = study.best_trial
        print('\nbest of %d: trial %d, val %.4f' % (len(ok), b.number, b.value))
        print("  %s" % json.dumps(b.params))


if __name__ == "__main__":
    sys.exit(main() or 0)
