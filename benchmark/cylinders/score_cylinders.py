"""Scores of the second structural system, the axially compressed cylinder: the three cylinder benchmarks, the
training-mean reference, the range check on the dense shells and the rank correlations between the benchmarks.

    python benchmark/cylinders/score_cylinders.py [--pred-dir outputs] [--splits-dir splits/cylinders]
                                                  [--data outputs/cyl_nodes_8192.npz] [--tags B1=cyl_B1cyl B2=cyl_B2cyl B3=cyl_cylB3]
                                                  [--b3-variants TAG [TAG ...]] [--corr-seeds B1=all B2=all B3=all] [--no-fields]
                                                  [--out outputs/score_cylinders.json]

No training and no inference: everything is computed from the stored test predictions of the field models.

INPUTS
  predictions  <pred-dir>/<family>_<tag>[_s<seed>]_test_pred.npz, the file benchmark/field_models/train.py writes at the
               end of a run (ids, pred_U, pred_K, true_U, true_K, P). <tag> names the benchmark (--tags); a file
               without the _s suffix is seed 0. --b3-variants: optional tags of additional B3 runs, each appended
               to the B3 tag of --tags (the same test shells); they are reported as <family>_<variant>.
               --pred-dir defaults to BLISS_HEAVY (or <BLISS_ROOT>/outputs).
  splits       <splits-dir>/B1.json, B2.json, B3.json (benchmark/cylinders/prepare_cylinders.py): train, val and test
               ids, and the regime of every test shell (test_regime {id: sparse|dense}, or the lists test_sparse and
               test_dense). Defaults to <BLISS_ROOT>/splits/cylinders.
  data         the training arrays of the cylinder shells (ids, K), needed only for the training-mean reference and
               the range check, which use the kappa of the training shells. One file that covers the shells of the
               three benchmarks; a benchmark whose training shells are not all in it is skipped for those two parts.

DEFINITIONS
  benchmarks   B1 within distribution (random split), B2 deeper dimples, B3 sparse to dense: trained on sparse
               layouts and tested on sparse and on dense layouts, where the dimples interact.
  kappa error  100 |kappa_pred / kappa_true - 1| per shell, the kappa_rel_err of bliss.metrics.score; reported as
               the median over the shells of a part. bias = the median of the signed error.
  field R2 (w) the cylinder analogue of the buckling R2 of the hemisphere. w = U . n is the radial deflection, with
               n = (x, y, 0) / |(x, y)| the outward normal of a cylinder of axis z. The mean of w over the nodes of
               the shell is removed from the truth and from the prediction (each its own mean), R2 is computed per
               shell (bliss.metrics._r2) and the median over shells is reported. A shell with a constant true w has
               no R2 and is left out.
  field R2 (U) R2 of the three displacement components, per shell, median over shells.
  parts        all = every test shell of the file, sparse and dense = the test shells of each regime.
  per family   the mean over the seeds of each per-run number.
  ratio rule   on B3, ratio = dense kappa error / sparse kappa error of a run. The rule holds when the ratio exceeds
               --rule-ratio (2) in every seed, with at least --rule-min-seeds (3) seeds, for at least
               --rule-min-families (2) of the families of --rule-families (transformer, figconv, mgn).
  training     the constant prediction kappa = mean kappa of the training shells of the benchmark, scored with the
  mean         same kappa error on the same parts.
  range check  B3 only. lo, hi = the smallest and largest kappa of the training shells, p5, p95 = their 5th and 95th
               percentiles. The kappa error of each run on the dense test shells is recomputed on the shells whose
               TRUE kappa lies in [lo, hi] and in [p5, p95]: an error that stays large there is not explained by
               the dense shells leaving the range of kappa seen in training.
  rank         Spearman correlation, over the model families scored on both, between the per-family kappa errors of
  correlations two benchmark columns. Columns: B1 (all test shells), B2 (all test shells), B3 sparse, B3 dense.
               The seeds that enter each protocol are set by --corr-seeds, one entry <benchmark>=<seed> or
               <benchmark>=all per benchmark. Default B1=all B2=all B3=all: every column takes the mean over the
               stored seeds of each family. <benchmark>=<seed> restricts a benchmark to the run of one seed. The
               convention is printed with the correlations and saved with them, with the number of runs behind
               every family and column. The runs of --b3-variants are left out.

OUTPUT
  <out> (default <BLISS_ROOT>/outputs/score_cylinders.json): benchmarks {B1, B2, B3} each with per_run, per_family and
  training_mean (B3 also ratio_rule), in_range, rank_correlations (seeds, convention, n_runs, columns, pairs). The
  same numbers are printed.

Functions
---------
log
    Print a message with the time.
kappa_error
    Relative error of the knockdown factor per shell, in percent.
load_split
    Read a cylinder split: ids of each set and the regime of every test shell.
radial_normals
    Unit outward normals of a cylinder of axis z at the given node positions.
radial_r2
    R2 of the radial deflection per shell, the node mean of each shell removed.
component_r2
    R2 of the three displacement components per shell.
find_runs
    Stored test predictions of one benchmark tag, as (family, seed, path).
score_run
    Kappa error and field R2 of one predictions file on its parts.
family_means
    Mean over the seeds of every per-run number, per family.
ratio_rule
    The dense-over-sparse rule on the sparse-to-dense benchmark.
training_mean
    The training-mean reference of one benchmark.
in_range
    Kappa error on the dense test shells whose true kappa lies inside the training range.
rank_correlations
    Spearman correlations between the per-family kappa errors of the benchmark columns.
main
    Score the cylinder benchmarks from the stored predictions and write outputs/score_cylinders.json.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs")
sys.path.insert(0, B)
from bliss.metrics import _r2   # noqa: E402

PARTS = ("all", "sparse", "dense")
QUANTITIES = ("kappa_med", "fieldR2_w_med", "fieldR2_U3_med")
COLUMNS = (("B1", "all"), ("B2", "all"), ("B3", "sparse"), ("B3", "dense"))   # columns of the rank correlations


def log(*a):
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kappa_error(pred, true):
    """Relative error of the knockdown factor per shell, in percent.

    Parameters
    ----------
    pred, true : array_like, shape (S,)
        Predicted and true knockdown factors.

    Returns
    -------
    err : numpy.ndarray, shape (S,)
        100 |pred / true - 1|.
    """
    return np.abs(np.asarray(pred, np.float64) / np.asarray(true, np.float64) - 1) * 100


def load_split(path):
    """Read a cylinder split: ids of each set and the regime of every test shell.

    Parameters
    ----------
    path : str
        Split file (train, val, test and test_regime, or test_sparse and test_dense).

    Returns
    -------
    split : dict
        train, val, test (lists of int) and regime {id: 'sparse' | 'dense'} of the test shells.
    """
    S = json.load(open(path))
    if "test_regime" in S: regime = {int(k): str(v) for k, v in S["test_regime"].items()}
    else: regime = {**{int(i): "sparse" for i in S.get("test_sparse", [])}, **{int(i): "dense" for i in S.get("test_dense", [])}}
    return dict(train=[int(i) for i in S["train"]], val=[int(i) for i in S.get("val", [])],
                test=[int(i) for i in S.get("test", sorted(regime))], regime=regime)


def radial_normals(P):
    """Unit outward normals of a cylinder of axis z at the given node positions.

    Parameters
    ----------
    P : numpy.ndarray, shape (N, 3)
        Node positions.

    Returns
    -------
    n : numpy.ndarray, shape (N, 3)
        (x, y, 0) / |(x, y)|.
    """
    n = np.c_[P[:, 0], P[:, 1], 0 * P[:, 0]]
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def radial_r2(U_true, U_pred, n):
    """R2 of the radial deflection per shell, the node mean of each shell removed.

    Parameters
    ----------
    U_true, U_pred : numpy.ndarray, shape (S, N, 3)
        True and predicted displacement fields.
    n : numpy.ndarray, shape (N, 3)
        Unit outward normals.

    Returns
    -------
    r2 : numpy.ndarray, shape (S,)
        R2 per shell; NaN where the true radial deflection is constant.
    """
    wt = (U_true * n).sum(-1); wp = (U_pred * n).sum(-1); out = np.full(len(wt), np.nan)
    for i in range(len(wt)):
        t = wt[i] - wt[i].mean()
        if np.ptp(t) > 0: out[i] = _r2(t, wp[i] - wp[i].mean())
    return out


def component_r2(U_true, U_pred):
    """R2 of the three displacement components per shell.

    Parameters
    ----------
    U_true, U_pred : numpy.ndarray, shape (S, N, 3)
        True and predicted displacement fields.

    Returns
    -------
    r2 : numpy.ndarray, shape (S,)
        R2 per shell; NaN where the true field is constant.
    """
    out = np.full(len(U_true), np.nan)
    for i in range(len(U_true)):
        if np.ptp(U_true[i]) > 0: out[i] = _r2(U_true[i].ravel(), U_pred[i].ravel())
    return out


def find_runs(pred_dir, tag):
    """Stored test predictions of one benchmark tag, as (family, seed, path).

    Parameters
    ----------
    pred_dir : str
        Folder of the prediction files.
    tag : str
        Tag of the benchmark in the file names: <family>_<tag>[_s<seed>]_test_pred.npz.

    Returns
    -------
    runs : list of tuple
        (family, seed, path), sorted by family and seed.
    """
    rx = re.compile(r"^([A-Za-z0-9]+)_%s(?:_s(\d+))?_test_pred\.npz$" % re.escape(tag)); runs = []
    for f in sorted(glob.glob("%s/*_%s*_test_pred.npz" % (pred_dir, tag))):
        m = rx.match(os.path.basename(f))
        if m: runs.append((m.group(1), int(m.group(2) or 0), f))
    return sorted(runs)


def score_run(path, regime, fields=True):
    """Kappa error and field R2 of one predictions file on its parts.

    Parameters
    ----------
    path : str
        Predictions file (ids, pred_K, true_K, pred_U, true_U, P).
    regime : dict
        {id: 'sparse' | 'dense'} of the test shells.
    fields : bool, default=True
        Also score the fields (reads pred_U and true_U).

    Returns
    -------
    row : dict
        epoch, one dict per part (n, kappa_med, bias_med, fieldR2_w_med, fieldR2_U3_med) and, when both regimes are
        present, ratio = dense kappa_med / sparse kappa_med.
    """
    z = np.load(path); ids = z["ids"].astype(int)
    b = (z["pred_K"] / z["true_K"] - 1) * 100; e = kappa_error(z["pred_K"], z["true_K"])
    if fields:
        fr = radial_r2(z["true_U"], z["pred_U"], radial_normals(z["P"])); fu = component_r2(z["true_U"], z["pred_U"])
    reg = np.array([regime.get(int(i), "") for i in ids])
    row = {"epoch": int(z["epoch"]) if "epoch" in z.files else -1}
    for part in PARTS:
        k = np.ones(len(ids), bool) if part == "all" else reg == part
        if not k.any(): continue
        row[part] = dict(n=int(k.sum()), kappa_med=float(np.median(e[k])), bias_med=float(np.median(b[k])))
        if fields: row[part].update(fieldR2_w_med=float(np.nanmedian(fr[k])), fieldR2_U3_med=float(np.nanmedian(fu[k])))
    if "sparse" in row and "dense" in row: row["ratio"] = row["dense"]["kappa_med"] / row["sparse"]["kappa_med"]
    return row


def family_means(per_run):
    """Mean over the seeds of every per-run number, per family.

    Parameters
    ----------
    per_run : dict
        {'<family>_s<seed>': row of score_run}.

    Returns
    -------
    per_family : dict
        {family: n_seeds and <part>_<quantity> means}.
    """
    fams = {}
    for k, v in per_run.items(): fams.setdefault(k.rsplit("_", 1)[0], []).append(v)
    return {f: dict(n_seeds=len(v), **{"%s_%s" % (p, q): float(np.mean([x[p][q] for x in v])) for p in PARTS for q in QUANTITIES
                                       if all(p in x and q in x[p] for x in v)}) for f, v in fams.items()}


def ratio_rule(per_run, families, ratio=2.0, min_seeds=3, min_families=2):
    """The dense-over-sparse rule on the sparse-to-dense benchmark.

    Parameters
    ----------
    per_run : dict
        {'<family>_s<seed>': row of score_run}, with a ratio in every row.
    families : sequence of str
        Families the rule is read on.
    ratio : float, default=2.0
        Threshold on dense kappa error / sparse kappa error.
    min_seeds : int, default=3
        Seeds a family needs to count.
    min_families : int, default=2
        Families that must pass.

    Returns
    -------
    rule : dict
        per_family {family: n_seeds, all_seeds_above}, families_passing, holds and the settings.
    """
    fams = {}
    for k, v in per_run.items(): fams.setdefault(k.rsplit("_", 1)[0], []).append(v["ratio"])
    per = {f: dict(n_seeds=len(v), all_seeds_above=bool(all(x > ratio for x in v))) for f, v in fams.items()}
    n = sum(per[f]["all_seeds_above"] and per[f]["n_seeds"] >= min_seeds for f in families if f in per)
    return dict(per_family=per, families_passing=int(n), holds=bool(n >= min_families), families=list(families), ratio=ratio,
                min_seeds=min_seeds, min_families=min_families)


def training_mean(split, kappa):
    """The training-mean reference of one benchmark.

    Parameters
    ----------
    split : dict
        Output of load_split.
    kappa : dict
        {id: kappa} of the shells.

    Returns
    -------
    ref : dict or None
        kappa_train_mean and, per part, n, kappa_med and bias_med of the constant prediction; None when a shell of
        the split is missing from `kappa`.
    """
    if not all(i in kappa for i in split["train"] + split["test"]): return None
    km = float(np.mean([kappa[i] for i in split["train"]])); out = dict(kappa_train_mean=km, n_train=len(split["train"]))
    t = np.array([kappa[i] for i in split["test"]]); reg = np.array([split["regime"].get(i, "") for i in split["test"]])
    for part in PARTS:
        k = np.ones(len(t), bool) if part == "all" else reg == part
        if k.any(): out[part] = dict(n=int(k.sum()), kappa_med=float(np.median(kappa_error(km, t[k]))),
                                     bias_med=float(np.median((km / t[k] - 1) * 100)))
    return out


def in_range(split, kappa, runs, part="dense"):
    """Kappa error on the dense test shells whose true kappa lies inside the training range.

    Parameters
    ----------
    split : dict
        Output of load_split.
    kappa : dict
        {id: kappa} of the shells.
    runs : list of tuple
        (name, path) of the predictions files to restrict.
    part : str, default='dense'
        Regime of the test shells the check is made on.

    Returns
    -------
    res : dict
        train (min, p5, p95, max), the share of the test shells of `part` inside each band, per run and for the
        training-mean reference [n, median kappa error] on all shells of `part`, inside [min, max] and inside
        [p5, p95].
    """
    ktr = np.array([kappa[i] for i in split["train"]]); sel = {i for i, v in split["regime"].items() if v == part}
    lo, hi = ktr.min(), ktr.max(); p5, p95 = np.percentile(ktr, [5, 95])
    bands = (("all", (-1, 9)), ("in_minmax", (lo, hi)), ("in_p5p95", (p5, p95)))
    kd = np.array([kappa[i] for i in sorted(sel)])
    def restrict(ids, t, e):
        """Median error of the shells of `part` inside each band.

        Parameters
        ----------
        ids : numpy.ndarray
            Shell ids.
        t : numpy.ndarray
            True kappa.
        e : numpy.ndarray
            Kappa error per shell.

        Returns
        -------
        r : dict
            {band: [n, median error or None]}.
        """
        isd = np.array([int(i) in sel for i in ids]); r = {}
        for nm, band in bands:
            k = isd & (t >= band[0]) & (t <= band[1]); r[nm] = [int(k.sum()), float(np.median(e[k])) if k.any() else None]
        return r
    out = {}
    for name, f in runs:
        z = np.load(f); out[name] = restrict(z["ids"].astype(int), z["true_K"], kappa_error(z["pred_K"], z["true_K"]))
    return dict(part=part, train=dict(min=float(lo), p5=float(p5), p95=float(p95), max=float(hi)),
                true=dict(n=int(len(kd)), median=float(np.median(kd)), share_in_minmax=float(np.mean((kd >= lo) & (kd <= hi))),
                          share_in_p5p95=float(np.mean((kd >= p5) & (kd <= p95)))),
                runs=out, training_mean=restrict(np.array(sorted(sel)), kd, kappa_error(ktr.mean(), kd)))


def rank_correlations(benchmarks, seeds, columns=COLUMNS, skip=()):
    """Spearman correlations between the per-family kappa errors of the benchmark columns.

    Parameters
    ----------
    benchmarks : dict
        {benchmark: {'per_run': {'<family>_s<seed>': row of score_run}}}.
    seeds : dict
        {benchmark: 'all' or a seed}: the runs each family's error is averaged over.
    columns : sequence of tuple, default=COLUMNS
        (benchmark, part) of each column.
    skip : collection of str, default=()
        Families left out (the variants of a benchmark, which are not separate families).

    Returns
    -------
    res : dict
        seeds {benchmark: 'all' or a seed}, convention (the same in one sentence), n_runs {name: {family: runs
        averaged}}, columns {name: {family: error}} and pairs {'<a> vs <b>': families, n, spearman}.
    """
    from scipy.stats import spearmanr
    col = {}; nrun = {}
    for bench, part in columns:
        fams = {}
        for k, v in benchmarks.get(bench, {}).get("per_run", {}).items():
            fam, s = k.rsplit("_s", 1)
            if fam not in skip and part in v and (seeds.get(bench, "all") == "all" or int(s) == int(seeds[bench])): fams.setdefault(fam, []).append(v[part]["kappa_med"])
        if fams:
            col["%s %s" % (bench, part)] = {f: float(np.mean(v)) for f, v in fams.items()}
            nrun["%s %s" % (bench, part)] = {f: len(v) for f, v in fams.items()}
    names = list(col); pairs = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fam = sorted(set(col[a]) & set(col[b]))
            if len(fam) < 3: continue
            r = spearmanr([col[a][f] for f in fam], [col[b][f] for f in fam]).correlation
            pairs["%s vs %s" % (a, b)] = dict(families=fam, n=len(fam), spearman=float(r) if np.isfinite(r) else None)
    used = {k: seeds.get(k, "all") for k in benchmarks}
    conv = "; ".join("%s: %s" % (k, "mean over the stored seeds of each family" if v == "all" else "seed %d" % v) for k, v in sorted(used.items()))
    return dict(seeds=used, convention=conv, n_runs=nrun, columns=col, pairs=pairs)


def main():
    """Score the cylinder benchmarks from the stored predictions and write outputs/score_cylinders.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default=H, help="folder of the <family>_<tag>[_s<seed>]_test_pred.npz files")
    ap.add_argument("--splits-dir", default=B + "/splits/cylinders", help="folder of B1.json, B2.json, B3.json")
    ap.add_argument("--data", default=H + "/cyl_nodes_8192.npz", help="training arrays of the cylinder shells (ids, K)")
    ap.add_argument("--tags", nargs="+", default=["B1=cyl_B1cyl", "B2=cyl_B2cyl", "B3=cyl_cylB3"], help="tag of each benchmark in the file names")
    ap.add_argument("--b3-variants", nargs="*", default=[], help="optional tags of additional B3 runs, appended to the B3 tag (same test shells)")
    ap.add_argument("--no-fields", action="store_true", help="kappa only, without reading the fields")
    ap.add_argument("--rule-families", nargs="+", default=["transformer", "figconv", "mgn"])
    ap.add_argument("--rule-ratio", type=float, default=2.0); ap.add_argument("--rule-min-seeds", type=int, default=3)
    ap.add_argument("--rule-min-families", type=int, default=2)
    ap.add_argument("--corr-seeds", nargs="+", default=["B1=all", "B2=all", "B3=all"],
                    help="seeds that enter each protocol in the rank correlations, <benchmark>=<seed> or <benchmark>=all (mean over the stored seeds); "
                         "default: the mean over the stored seeds on the three benchmarks")
    ap.add_argument("--out", default=B + "/outputs/score_cylinders.json"); a = ap.parse_args()
    tags = dict(t.split("=", 1) for t in a.tags); kappa = None
    if os.path.exists(a.data):
        d = np.load(a.data); kappa = {int(i): float(k) for i, k in zip(d["ids"], d["K"])}
    else: log("no training arrays at %s: the training-mean reference and the range check are skipped" % a.data)
    res = {"benchmarks": {}}; splits = {}
    for bench, tag in tags.items():
        sf = "%s/%s.json" % (a.splits_dir, bench)
        if not os.path.exists(sf): log("skip %s: no split file %s" % (bench, sf)); continue
        sp = splits[bench] = load_split(sf); per_run = {}
        for var in [""] + (list(a.b3_variants) if bench == "B3" else []):
            for fam, seed, f in find_runs(a.pred_dir, tag + var):
                name = "%s%s_s%d" % (fam, "_" + var if var else "", seed); per_run[name] = row = score_run(f, sp["regime"], not a.no_fields)
                log("%s %-20s " % (bench, name) + " | ".join("%s n %d kappa %5.2f%s" % (p, row[p]["n"], row[p]["kappa_med"],
                    "" if a.no_fields else " R2(w) %.3f R2(U) %.3f" % (row[p]["fieldR2_w_med"], row[p]["fieldR2_U3_med"])) for p in PARTS if p in row)
                    + (" | ratio %.1f" % row["ratio"] if "ratio" in row else ""))
        R = res["benchmarks"][bench] = dict(tag=tag, per_run=per_run, per_family=family_means(per_run))
        if bench == "B3" and per_run:
            R["ratio_rule"] = ratio_rule(per_run, a.rule_families, a.rule_ratio, a.rule_min_seeds, a.rule_min_families)
            for f, v in R["ratio_rule"]["per_family"].items(): R["per_family"][f]["all_seeds_ratio_above"] = v["all_seeds_above"]
            log("B3 ratio rule (dense > %g x sparse in every seed, >= %d seeds, >= %d of %s): %s, %d families pass"
                % (a.rule_ratio, a.rule_min_seeds, a.rule_min_families, "/".join(a.rule_families),
                   "holds" if R["ratio_rule"]["holds"] else "does not hold", R["ratio_rule"]["families_passing"]))
        if kappa is not None:
            R["training_mean"] = tm = training_mean(sp, kappa)
            if tm is None: log("%s training mean: skipped, the arrays do not hold every shell of the split" % bench)
            else: log("%s training mean (kappa %.4f): " % (bench, tm["kappa_train_mean"]) + " | ".join("%s n %d kappa %5.2f" % (p, tm[p]["n"], tm[p]["kappa_med"]) for p in PARTS if p in tm))
    if kappa is not None and "B3" in splits and all(i in kappa for i in splits["B3"]["train"] + list(splits["B3"]["regime"])):
        runs = [(k, f) for var in [""] + list(a.b3_variants) for fam, seed, f in find_runs(a.pred_dir, tags["B3"] + var)
                for k in ["%s%s_s%d" % (fam, "_" + var if var else "", seed)]]
        res["in_range"] = ir = in_range(splits["B3"], kappa, runs); t = ir["train"]
        log("range check: train kappa min %.4f p5 %.4f p95 %.4f max %.4f; dense true kappa median %.4f, %.0f %% inside [min, max], %.0f %% inside [p5, p95]"
            % (t["min"], t["p5"], t["p95"], t["max"], ir["true"]["median"], 100 * ir["true"]["share_in_minmax"], 100 * ir["true"]["share_in_p5p95"]))
        for k, v in list(ir["runs"].items()) + [("training_mean", ir["training_mean"])]:
            log("  %-20s " % k + " | ".join("%s n %d kappa %s" % (nm, v[nm][0], "nan" if v[nm][1] is None else "%5.2f" % v[nm][1]) for nm in v))
    res["rank_correlations"] = rc = rank_correlations(res["benchmarks"], {k: (v if v == "all" else int(v)) for k, v in (c.split("=", 1) for c in a.corr_seeds)},
                                                 skip={k.rsplit("_s", 1)[0] for k in res["benchmarks"].get("B3", {}).get("per_run", {}) if any(k.rsplit("_s", 1)[0].endswith("_" + v) for v in a.b3_variants)})
    log("rank correlations, seeds per protocol: %s" % rc["convention"])
    for k, v in rc["n_runs"].items(): log("  column %-10s runs per family: %s" % (k, ", ".join("%s %d" % (f, n) for f, n in sorted(v.items()))))
    for k, v in rc["pairs"].items(): log("Spearman %-24s %s over %d families" % (k, "nan" if v["spearman"] is None else "%+.2f" % v["spearman"], v["n"]))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1); log("wrote", a.out)


if __name__ == "__main__":
    main()
