"""Semi-analytical shell model for the knockdown factor: its error on the three protocols, and a model compared with it on
the same shells.

    python benchmark/references/shell_model.py [--release BLISS-1.0] [--labels FILE] [--reference FILE]
                                                       [--pred FILE [--benchmark B1|B2|B3]] [--n-boot 2000] [--out FILE]

The reference is a semi-analytical shell model derived from the problem statement without simulation data, with three
coefficients calibrated on the knockdown factors of the training shells of each benchmark; released as per-shell
predictions. This script reads those predictions and scores them. No training and no inference.

INPUTS
  reference    benchmark/references/data/shell_model_kappa.csv, columns benchmark, shell_id, kappa_shell_model:
               the kappa of the reference on the test shells of B1 (660), B2 (1,234) and B3 (2,200).
  labels       the kappa of the solver, per shell. Read from the release (--release, default BLISS_RELEASE or
               <BLISS_ROOT>/BLISS-1.0) opened with bliss.load: the column knock_down of metadata/shells.csv, or the
               kappa of each shell record where the release has no such table. Without a release directory the table
               <BLISS_ROOT>/metadata/shells.csv is read. --labels gives another csv with the columns id and knock_down
               (or kappa). Reference shells without a label (the review sample holds a few dozen shells) are left out
               and counted.
  --pred       optional: test predictions of a model, the file benchmark/field_models/train.py writes and
               benchmark/tables/rescore.py reads (.npz; only ids and pred_K are used here). --benchmark names its
               protocol; by default it is the protocol whose reference shells hold the most ids of the file.

DEFINITIONS
  e = |kappa_pred - kappa_true| / kappa_true * 100 per shell, and its median over the shells: the kappa_rel_err of
  bliss.metrics.score. The interval of a median is the shell bootstrap of bliss.metrics.bootstrap_ci (2,000 resamples,
  seed 0, 95 %).
  With --pred, the model and the reference are scored on the shells both predict. Paired difference = median e of the
  model minus median e of the reference; shell bootstrap as bliss.metrics.compare: each of the 2,000 resamples (one
  generator, seed 0) draws the shells with replacement and applies the same draw to both, 2.5 and 97.5 percentiles.
  A negative difference means the model has the lower error.
  Medians of the reference on the labels of release 1.0: B1 1.44 %, B2 2.74 %, B3 1.41 %.
Output: printed; --out also writes the numbers as json.

Functions
---------
read_reference
    Per-shell kappa of the reference, per benchmark.
read_labels
    Solver kappa per shell, from the release or from a table.
rel_err
    Relative kappa error per shell, in percent.
paired_difference
    Difference of two median errors on the same shells with a paired shell bootstrap.
score_reference
    Median error of the reference on each benchmark.
score_model
    A model against the reference on the shells both predict.
main
    Score the reference and, with --pred, compare a model with it.
"""
import argparse, csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import bliss   # noqa: E402
from bliss.metrics import _stat_median, bootstrap_ci   # noqa: E402

REFERENCE = os.path.join(HERE, "data", "shell_model_kappa.csv")
RELEASE = os.environ.get("BLISS_RELEASE", os.path.join(B, "BLISS-1.0"))
PROT = ("B1", "B2", "B3")


def read_reference(path=REFERENCE):
    """Per-shell kappa of the reference, per benchmark.

    Parameters
    ----------
    path : str
        Csv with the columns benchmark, shell_id, kappa_shell_model.

    Returns
    -------
    ref : dict
        Benchmark -> {shell id: kappa of the reference}.
    """
    ref = {}
    for r in csv.DictReader(open(path)):
        ref.setdefault(r["benchmark"], {})[int(r["shell_id"])] = float(r["kappa_shell_model"])
    return ref


def read_labels(ids, release=RELEASE, labels=None):
    """Solver kappa per shell, from the release or from a table.

    Parameters
    ----------
    ids : iterable of int
        Shells whose label is wanted.
    release : str
        Release directory (opened with bliss.load).
    labels : str, default=None
        Csv with the columns id and knock_down (or kappa); read instead of the release.

    Returns
    -------
    kappa : dict
        Shell id -> kappa of the solver, for the shells that have a label.
    source : str
        Where the labels were read from.
    """
    ids = set(int(i) for i in ids)
    table = lambda rows: {int(r["id"]): float(r["knock_down"] if "knock_down" in r else r["kappa"]) for r in rows if int(r["id"]) in ids}
    if labels: return table(csv.DictReader(open(labels))), os.path.basename(labels)
    if os.path.exists(os.path.join(release, "shared.npz")):
        d = bliss.load(release)
        if os.path.exists(os.path.join(d.root, "metadata", "shells.csv")):
            k = table(d.meta("shells.csv"))
            if k: return k, "metadata/shells.csv of the release"
        return {int(i): d[i].kappa for i in d.ids if int(i) in ids}, "shell records of the release"
    f = os.path.join(B, "metadata", "shells.csv")
    if os.path.exists(f): return table(csv.DictReader(open(f))), "metadata/shells.csv"
    raise SystemExit("no release at %s and no metadata/shells.csv: give --release or --labels" % release)


def rel_err(pred, true):
    """Relative kappa error per shell, in percent (the per-shell values of kappa_rel_err in bliss.metrics.score).

    Parameters
    ----------
    pred, true : array_like, shape (S,)
        Predicted and true knockdown factors.

    Returns
    -------
    e : numpy.ndarray, shape (S,)
        |pred - true| / true * 100.
    """
    pred, true = np.asarray(pred, np.float64), np.asarray(true, np.float64)
    return np.abs(pred - true) / true * 100


def paired_difference(ea, eb, n_boot=2000, seed=0, alpha=0.05):
    """Difference of two median errors on the same shells with a paired shell bootstrap.

    Parameters
    ----------
    ea, eb : numpy.ndarray, shape (S,)
        Per-shell errors of A and of B, in the same order of shells.
    n_boot : int, default=2000
        Resamples.
    seed : int, default=0
        Seed of the generator.
    alpha : float, default=0.05
        Two-sided level.

    Returns
    -------
    diff, lo, hi : float
        median(ea) - median(eb) and its interval; each resample draws the shells once for both.
    """
    va, vb = np.asarray(ea, np.float64)[:, None], np.asarray(eb, np.float64)[:, None]; s = len(va)
    rng = np.random.default_rng(seed); idx = rng.integers(0, s, size=(n_boot, s))
    draws = _stat_median(va[idx]) - _stat_median(vb[idx])
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(_stat_median(va[None])[0] - _stat_median(vb[None])[0]), float(lo), float(hi)


def score_reference(ref, kappa, n_boot=2000):
    """Median error of the reference on each benchmark.

    Parameters
    ----------
    ref : dict
        Output of read_reference.
    kappa : dict
        Shell id -> kappa of the solver.
    n_boot : int, default=2000
        Resamples of the interval.

    Returns
    -------
    rows : dict
        Benchmark -> n_reference, n_scored, kappa_rel_err and ci (None when no shell has a label).
    """
    rows = {}
    for b in [p for p in PROT if p in ref]:
        ids = [i for i in sorted(ref[b]) if i in kappa]
        if not ids: rows[b] = dict(n_reference=len(ref[b]), n_scored=0, kappa_rel_err=None, ci=None); continue
        e = rel_err([ref[b][i] for i in ids], [kappa[i] for i in ids]); pt, lo, hi = bootstrap_ci(e, n_boot=n_boot, seed=0, statistic="median")
        rows[b] = dict(n_reference=len(ref[b]), n_scored=len(ids), kappa_rel_err=pt, ci=[lo, hi])
    return rows


def score_model(pred, ref, kappa, benchmark=None, n_boot=2000):
    """A model against the reference on the shells both predict.

    Parameters
    ----------
    pred : str
        Prediction file of the model (.npz with ids and pred_K).
    ref : dict
        Output of read_reference.
    kappa : dict
        Shell id -> kappa of the solver.
    benchmark : str, default=None
        Protocol of the file; the one whose reference shells hold the most ids of the file if None.
    n_boot : int, default=2000
        Resamples.

    Returns
    -------
    row : dict
        benchmark, n_file, n_scored, the two median errors and the paired difference model - reference with its
        interval.
    """
    z = np.load(pred); pk = {int(i): float(k) for i, k in zip(z["ids"], np.asarray(z["pred_K"]).ravel())}
    if benchmark is None:
        n = {b: len(set(pk) & set(ref[b])) for b in ref}; benchmark = max(n, key=n.get)
        if sorted(n.values())[-2:].count(n[benchmark]) > 1 or not n[benchmark]: raise SystemExit("the protocol of %s is not clear from its ids: give --benchmark" % os.path.basename(pred))
    ids = [i for i in sorted(ref[benchmark]) if i in pk and i in kappa]
    if len(ids) < 2: raise SystemExit("%s shares %d labelled shells with the reference on %s" % (os.path.basename(pred), len(ids), benchmark))
    kt = [kappa[i] for i in ids]; em = rel_err([pk[i] for i in ids], kt); er = rel_err([ref[benchmark][i] for i in ids], kt)
    d, lo, hi = paired_difference(em, er, n_boot)
    return dict(file=os.path.basename(pred), benchmark=benchmark, n_file=len(pk), n_scored=len(ids), model_kappa_rel_err=float(np.median(em)),
                reference_kappa_rel_err=float(np.median(er)), difference=d, ci=[lo, hi], n_boot=n_boot)


def main():
    """Score the reference and, with --pred, compare a model with it."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default=RELEASE, help="release directory with the solver labels")
    ap.add_argument("--labels", default=None, help="csv with the columns id and knock_down (or kappa), read instead of the release")
    ap.add_argument("--reference", default=REFERENCE, help="per-shell kappa of the reference (benchmark, shell_id, kappa_shell_model)")
    ap.add_argument("--pred", default=None, help="test predictions of a model (.npz with ids and pred_K)")
    ap.add_argument("--benchmark", default=None, choices=PROT, help="protocol of --pred (default: read from its ids)")
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--out", default=None, help="also write the numbers to this json file")
    a = ap.parse_args()
    ref = read_reference(a.reference)
    kappa, src = read_labels({i for v in ref.values() for i in v}, a.release, a.labels)
    print("labels: %s (%d of the %d reference shells)" % (src, len(kappa), len({i for v in ref.values() for i in v})))
    res = dict(labels=src, reference=score_reference(ref, kappa, a.n_boot))
    for b, r in res["reference"].items():
        if r["n_scored"]: print("%s  semi-analytical shell model  median kappa error %.2f %%  [%.2f, %.2f]  (%d of %d shells)" % (b, r["kappa_rel_err"], r["ci"][0], r["ci"][1], r["n_scored"], r["n_reference"]))
        else: print("%s  semi-analytical shell model  no labelled shell (0 of %d)" % (b, r["n_reference"]))
    if a.pred:
        res["model"] = m = score_model(a.pred, ref, kappa, a.benchmark, a.n_boot)
        print("%s  %s on %d shells: model %.2f %%  semi-analytical shell model %.2f %%  model - reference %+.2f [%+.2f, %+.2f] (paired shell bootstrap, %d resamples, seed 0)"
              % (m["benchmark"], m["file"], m["n_scored"], m["model_kappa_rel_err"], m["reference_kappa_rel_err"], m["difference"], m["ci"][0], m["ci"][1], a.n_boot))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
        print("wrote " + a.out)


if __name__ == "__main__":
    main()
