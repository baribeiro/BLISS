"""Label-range control on B3: the kappa error on all dense test shells and on those whose true kappa lies inside the
range of the training labels.

    python benchmark/analyses/label_range.py [--families deeponet,mgn,transformer,figconv,transolver,sfno]
                                  [--runs runs] [--split splits/B3.json] [--release BLISS-1.0] [--labels FILE]
                                  [--reference benchmark/references/data/shell_model_kappa.csv]
                                  [--truth file|labels] [--out-dir outputs]

B3 trains on sparse shells and tests on dense shells, and part of the dense shells have a knockdown factor below the
smallest one seen in training. If the B3 error only came from those shells, it would fall once the test set is cut to
the shells whose true kappa lies inside the training range. This script compares the two errors for every model family,
for the training-mean reference and for the semi-analytical shell model. No training and no inference: stored
predictions only.

INPUTS
  <runs>/final_runs/<fam>_B3_n8192/*_test_pred.npz   test predictions of the B3 finals (transfer.py), one file per
                                                     seed; only ids, pred_K and true_K are read. A file without
                                                     the _s<seed> suffix is seed 0.
  --split                                            the B3 protocol, a json with the lists train and test. Default
                                                     splits/B3.json, read relative to BLISS_ROOT.
  labels                                             the kappa of the solver, per shell, read as
                                                     benchmark/references/shell_model.py reads it: from the release
                                                     (--release, default BLISS_RELEASE or <BLISS_ROOT>/BLISS-1.0), or
                                                     from --labels, a csv with the columns id and knock_down (or
                                                     kappa). Every training shell needs a label; test shells without
                                                     one are left out and counted.
  --reference                                        per-shell kappa of the semi-analytical shell model (columns
                                                     benchmark, shell_id, kappa_shell_model); the rows of B3 are read.

DEFINITIONS
  Range: lo, hi = the smallest and the largest kappa of the B3 training shells.
  Label: the released kappa of a shell, taken by shell id. The range and the mask are computed from the labels for
  every predictor, so that all predictors are cut to the same shells.
  all = the B3 test shells that have a label (all of them dense). in range = those with lo <= label <= hi.
  below, inside, above = the number of these test shells with label < lo, in [lo, hi] and > hi.
  e = |kappa_pred / kappa_true - 1| * 100 per shell. For a family, kappa_true is the true_K stored with the prediction
  (--truth file, the default), the truth of bliss.metrics.score, so that the error on all shells is the kappa_rel_err
  of the tables; --truth labels divides by the label instead. The two truths agree to about 0.02 % per shell. The
  training mean and the shell model are always scored against the labels, as in shell_model.py.
  Family: the median of e over the shells, for each seed, on all and on the shells in range; the mean over the seeds is
  reported, with the per-seed values. A file that does not hold every labelled test shell is skipped.
  Training mean: the constant prediction kappa = mean kappa of the training shells, scored the same way.
  Shell model: the released predictions of the semi-analytical shell model on the same shells, scored the same way;
  left out if they do not cover every labelled test shell.
Output: <out-dir>/label_range.json.

Functions
---------
kappa_range
    Smallest and largest kappa of the training shells.
range_counts
    Number of shells below, inside and above a range of kappa.
range_errors
    Median kappa error on all shells and on the shells inside a range of true kappa.
test_predictions
    Test prediction files of the B3 finals of one family, per seed.
family_errors
    The two median errors of one family, per seed and as the mean over seeds.
main
    Score the label-range control and write label_range.json.
"""
import argparse, glob, json, os, re, sys
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs"); RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references"))
import shell_model as SM   # noqa: E402

FAMILIES = ["deeponet", "mgn", "transformer", "figconv", "transolver", "sfno"]
SPLIT = B + "/splits/B3.json"


def kappa_range(train_kappa):
    """Smallest and largest kappa of the training shells.

    Parameters
    ----------
    train_kappa : array_like, shape (S,)
        Knockdown factors of the training shells.

    Returns
    -------
    lo, hi : float
        The two ends of the range.
    """
    k = np.asarray(train_kappa, np.float64)
    return float(k.min()), float(k.max())


def range_counts(kappa, lo, hi):
    """Number of shells below, inside and above a range of kappa.

    Parameters
    ----------
    kappa : array_like, shape (S,)
        True knockdown factors.
    lo, hi : float
        Ends of the range; both belong to it.

    Returns
    -------
    n : dict
        below, inside and above.
    """
    k = np.asarray(kappa, np.float64)
    return dict(below=int((k < lo).sum()), inside=int(((k >= lo) & (k <= hi)).sum()), above=int((k > hi).sum()))


def range_errors(pred, true, lo, hi, label=None):
    """Median kappa error on all shells and on the shells whose label lies inside a range of kappa.

    Parameters
    ----------
    pred : array_like, shape (S,) or float
        Predicted knockdown factors; a number is a constant prediction.
    true : array_like, shape (S,)
        True knockdown factors the error is taken against.
    lo, hi : float
        Ends of the range; both belong to it.
    label : array_like, shape (S,), default=None
        Knockdown factors the range is applied to; `true` if None.

    Returns
    -------
    r : dict
        all and in_range, the median of |pred / true - 1| * 100 over each set (None for an empty set), and n_all and
        n_in_range, the number of shells of each.
    """
    t = np.asarray(true, np.float64); e = SM.rel_err(np.broadcast_to(np.asarray(pred, np.float64), t.shape), t)
    m = t if label is None else np.asarray(label, np.float64); k = (m >= lo) & (m <= hi)
    return dict(n_all=int(len(t)), n_in_range=int(k.sum()), all=float(np.median(e)) if len(t) else None,
                in_range=float(np.median(e[k])) if k.any() else None)


def test_predictions(fam, runs=RUNS):
    """Test prediction files of the B3 finals of one family, per seed.

    Parameters
    ----------
    fam : str
        Model family.
    runs : str
        Directory of the trained models (holds final_runs/<fam>_B3_n8192).

    Returns
    -------
    files : dict
        Seed -> test prediction file, in increasing seed.
    """
    out = {}
    for f in sorted(glob.glob("%s/final_runs/%s_B3_n8192/*_test_pred.npz" % (runs, fam))):
        m = re.search(r"_s(\d+)_test_pred\.npz$", f); s = int(m.group(1)) if m else 0
        assert s not in out, "more than one B3 final of %s for seed %d under %s" % (fam, s, runs)
        out[s] = f
    return dict(sorted(out.items()))


def family_errors(preds, ids, kappa, lo, hi, truth=None):
    """The two median errors of one family, per seed and as the mean over seeds.

    Parameters
    ----------
    preds : dict
        Seed -> {shell id: predicted kappa}.
    ids : list of int
        Shells to score, the same for every predictor.
    kappa : dict
        Shell id -> label, the kappa the range is applied to.
    lo, hi : float
        Ends of the range of the training labels.
    truth : dict, default=None
        Seed -> {shell id: true kappa} the error of that seed is taken against; the labels if None.

    Returns
    -------
    r : dict or None
        n_seeds, seeds, n_all, n_in_range, all and in_range (means over the seeds of the per-seed medians) and
        per_seed {seed: row of range_errors}; None when no seed holds every shell of `ids`.
    """
    lab = [kappa[i] for i in ids]; rows = {}
    for s, pk in preds.items():
        tk = kappa if truth is None else truth[s]
        if all(i in pk and i in tk for i in ids): rows[s] = range_errors([pk[i] for i in ids], [tk[i] for i in ids], lo, hi, lab)
    if not rows: return None
    first = next(iter(rows.values())); mean = lambda k: float(np.mean([r[k] for r in rows.values()])) if first[k] is not None else None
    return dict(n_seeds=len(rows), seeds=sorted(rows), n_all=first["n_all"], n_in_range=first["n_in_range"], all=mean("all"),
                in_range=mean("in_range"), per_seed={str(s): rows[s] for s in sorted(rows)})


def main():
    """Score the label-range control and write label_range.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--runs", default=RUNS, help="directory of the trained models (holds final_runs/)")
    ap.add_argument("--split", default=SPLIT, help="json of the B3 protocol, with the lists train and test")
    ap.add_argument("--release", default=SM.RELEASE, help="release directory with the solver labels")
    ap.add_argument("--labels", default=None, help="csv with the columns id and knock_down (or kappa), read instead of the release")
    ap.add_argument("--reference", default=SM.REFERENCE, help="per-shell kappa of the semi-analytical shell model (benchmark, shell_id, kappa_shell_model)")
    ap.add_argument("--truth", default="file", choices=("file", "labels"), help="true kappa of the error of a family: the true_K of its prediction file, or the labels")
    ap.add_argument("--out-dir", default=H)
    a = ap.parse_args()
    f = a.split if os.path.isabs(a.split) or os.path.exists(a.split) else os.path.join(B, a.split)
    S = json.load(open(f)); train, test = [int(i) for i in S["train"]], [int(i) for i in S["test"]]
    kappa, src = SM.read_labels(train + test, a.release, a.labels)
    missing = [i for i in train if i not in kappa]
    if missing: raise SystemExit("%d of the %d training shells have no label in %s: the range needs the full release (--release or --labels)" % (len(missing), len(train), src))
    ids = [i for i in test if i in kappa]
    if not ids: raise SystemExit("no test shell of %s has a label in %s" % (os.path.basename(f), src))
    ktr = [kappa[i] for i in train]; kte = [kappa[i] for i in ids]; lo, hi = kappa_range(ktr); cnt = range_counts(kte, lo, hi)
    print("labels: %s; training range of kappa [%.4f, %.4f] from %d shells" % (src, lo, hi, len(train)))
    print("test shells: %d of %d with a label; %d below the range, %d inside, %d above" % (len(ids), len(test), cnt["below"], cnt["inside"], cnt["above"]))
    out = dict(labels=src, truth=a.truth, train=dict(n=len(train), min=lo, max=hi, mean=float(np.mean(ktr))), n_test=len(test), n_scored=len(ids), counts=cnt, families={})
    for fam in a.families.split(","):
        files = test_predictions(fam, a.runs); preds, truth = {}, {}
        for s, x in files.items():
            z = np.load(x); col = lambda k: {int(i): float(v) for i, v in zip(z["ids"], np.asarray(z[k]).ravel())}
            preds[s] = col("pred_K"); truth[s] = col("true_K") if a.truth == "file" else kappa
        r = family_errors(preds, ids, kappa, lo, hi, truth)
        if r is None: print("skip %s (no B3 test predictions that hold every scored shell)" % fam); continue
        if r["n_seeds"] < len(files): print("%s: %d of %d files do not hold every scored shell and are left out" % (fam, len(files) - r["n_seeds"], len(files)))
        out["families"][fam] = r
        print("%-12s all %.2f %%  in range %s  (%d seeds)" % (fam, r["all"], "nan" if r["in_range"] is None else "%.2f %%" % r["in_range"], r["n_seeds"]))
    if not out["families"]: raise SystemExit("no B3 test predictions under %s/final_runs" % a.runs)
    out["training_mean"] = tm = range_errors(float(np.mean(ktr)), kte, lo, hi)
    rows = [("training mean", tm)]
    ref = SM.read_reference(a.reference).get("B3", {})
    if all(i in ref for i in ids): out["shell_model"] = range_errors([ref[i] for i in ids], kte, lo, hi); rows.append(("shell model", out["shell_model"]))
    else: print("skip the shell model (%s does not hold every scored shell of B3)" % os.path.basename(a.reference))
    for name, r in rows: print("%-12s all %.2f %%  in range %s" % (name, r["all"], "nan" if r["in_range"] is None else "%.2f %%" % r["in_range"]))
    os.makedirs(a.out_dir, exist_ok=True); g = a.out_dir + "/label_range.json"
    json.dump(out, open(g, "w"), indent=1); print("wrote " + g)


if __name__ == "__main__":
    main()
