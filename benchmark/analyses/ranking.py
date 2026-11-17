"""Ranking of the model families across protocols: does the order of the families on one protocol carry over to another?

    python benchmark/analyses/ranking.py [--runs-dir runs/final_runs] [--protocols B1,B2,B3] [--reference B1]
                                         [--families ...] [--exclude mlp,mlp80] [--all-pairs] [--n-boot 2000]
    python benchmark/analyses/ranking.py --table errors.csv [--reference NAME]     # any table of errors per family

INPUTS (read only), one of
  prediction files   <runs-dir>/<family>_<protocol>_n8192/*_test_pred.npz (keys pred_K, true_K), one file per seed, as
                     written by benchmark/field_models/train.py. The folder name is set by --pattern. Families are
                     found from the folders of the reference protocol; a family without predictions on every
                     protocol is dropped.
  --table            a csv with the columns family, protocol, error (one row per family and protocol; several rows
                     of one family and protocol, e.g. one per seed, are averaged). Any error where lower is better
                     and any set of protocols can be passed, so the table of another dataset is scored the same
                     way. Without per-shell errors the two bootstraps are skipped.

DEFINITIONS
  error of a family on a protocol = mean over its seeds of the median over the test shells of |kappa_pred /
       kappa_true - 1|, in percent (the kappa_rel_err of bliss.metrics, averaged over seeds).
  rho, tau = Spearman and Kendall rank correlation, over the families, of the errors on two protocols.
  exact permutation p-value = share of the n! orderings of the families on the second protocol whose |rho| (|tau|)
       is at least the observed one (two-sided), and the share with rho (tau) at most the observed one (one-sided,
       negative). Computed for up to --max-exact families (9).
  leave one family out = rho and tau with each family removed in turn.
  shell bootstrap (families fixed) = --n-boot resamples (2,000), seed 0; in each one, for every family and protocol,
       one seed is drawn and its test shells are resampled with replacement; 95 % percentile interval of rho. The
       pairs of the reference protocol with every other protocol share one random stream, in the order of
       --protocols; with --all-pairs each remaining pair has its own stream.
  family bootstrap = the same, with the families also resampled with replacement (one stream per pair, seed 0);
       resamples with fewer than three distinct families are dropped. 95 % percentile interval of rho and tau.
Output: outputs/ranking.json (--out) and a printed table.

Functions
---------
discover_families
    Families that have a folder of predictions on one protocol.
load_errors
    Per-shell relative errors of kappa of every family, protocol and seed, from the prediction files.
read_table
    Errors per family and protocol from a csv table.
point_errors
    Error of every family on every protocol: mean over seeds of the median over shells, in percent.
spearman
    Spearman rank correlation.
kendall
    Kendall rank correlation.
exact_permutation
    Exact permutation p-values of the Spearman and Kendall correlations.
leave_one_out
    Rank correlations with each family removed in turn.
draw_error
    One bootstrap draw of the error of a family: one seed, shells resampled.
shell_bootstrap
    Interval of rho with the families fixed, seeds and shells resampled.
family_bootstrap
    Interval of rho and tau with families, seeds and shells resampled.
rank_table
    Rank of every family on every protocol (1 = lowest error).
compare_protocols
    All the statistics of one pair of protocols from the table of errors.
main
    Compare the ranking of the families across protocols and write outputs/ranking.json.
"""
import argparse, csv, glob, itertools, json, os, re, sys
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"
RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, B)
PATTERN = "{family}_{protocol}_n8192"


def discover_families(runs_dir, protocol, pattern=PATTERN):
    """Families that have a folder of predictions on one protocol.

    Parameters
    ----------
    runs_dir : str
        Directory with one folder per family and protocol.
    protocol : str
        Protocol.
    pattern : str, default='{family}_{protocol}_n8192'
        Name of the folder of a family and protocol.

    Returns
    -------
    families : list of str
        Sorted names.
    """
    rx = re.compile("^" + re.escape(pattern).replace(re.escape("{family}"), "(.+)").replace(re.escape("{protocol}"), re.escape(protocol)) + "$")
    found = [rx.match(os.path.basename(d)) for d in glob.glob(os.path.join(runs_dir, pattern.format(family="*", protocol=protocol)))]
    return sorted({m.group(1) for m in found if m})


def load_errors(runs_dir, families, protocols, pattern=PATTERN):
    """Per-shell relative errors of kappa of every family, protocol and seed, from the prediction files.

    Parameters
    ----------
    runs_dir : str
        Directory with one folder per family and protocol.
    families : list of str
        Families.
    protocols : list of str
        Protocols.
    pattern : str, default='{family}_{protocol}_n8192'
        Name of the folder of a family and protocol.

    Returns
    -------
    errors : dict
        (family, protocol) -> list (one entry per seed, files in sorted order) of arrays |kappa_pred / kappa_true - 1|.
    """
    E = {}
    for f in families:
        for b in protocols:
            E[(f, b)] = []
            for x in sorted(glob.glob(os.path.join(runs_dir, pattern.format(family=f, protocol=b), "*_test_pred.npz"))):
                z = np.load(x); E[(f, b)].append(np.abs(z["pred_K"] / z["true_K"] - 1))
    return E


def read_table(path):
    """Errors per family and protocol from a csv table.

    Parameters
    ----------
    path : str
        csv with the columns family, protocol, error; repeated rows of one family and protocol are averaged.

    Returns
    -------
    point : dict
        protocol -> {family: error}, protocols in the order of the file.
    """
    acc = {}
    for r in csv.DictReader(open(path)):
        acc.setdefault(r["protocol"].strip(), {}).setdefault(r["family"].strip(), []).append(float(r["error"]))
    return {b: {f: float(np.mean(v)) for f, v in fam.items()} for b, fam in acc.items()}


def point_errors(E, families, protocols):
    """Error of every family on every protocol: mean over seeds of the median over shells, in percent.

    Parameters
    ----------
    E : dict
        Output of load_errors.
    families : list of str
        Families.
    protocols : list of str
        Protocols.

    Returns
    -------
    point : dict
        protocol -> {family: error in percent}.
    """
    return {b: {f: float(np.mean([100 * np.median(x) for x in E[(f, b)]])) for f in families} for b in protocols}


def spearman(x, y):
    """Spearman rank correlation.

    Parameters
    ----------
    x, y : array_like
        Data.

    Returns
    -------
    rho : float
        Correlation.
    """
    from scipy.stats import spearmanr
    return float(spearmanr(x, y)[0])


def kendall(x, y):
    """Kendall rank correlation.

    Parameters
    ----------
    x, y : array_like
        Data.

    Returns
    -------
    tau : float
        Correlation.
    """
    from scipy.stats import kendalltau
    return float(kendalltau(x, y)[0])


def exact_permutation(x, y):
    """Exact permutation p-values of the Spearman and Kendall correlations: all n! orderings of y against x.

    The closed forms of rho and tau used here assume no ties.

    Parameters
    ----------
    x, y : array_like
        Data, one value per family.

    Returns
    -------
    test : dict
        n_perm, rho, tau, the two-sided p-values and the one-sided (negative) p-values.
    """
    from scipy.stats import rankdata
    n = len(x); rx = rankdata(x); ry = rankdata(y)
    perms = np.array(list(itertools.permutations(range(n))))
    RY = ry[perms]                                                     # (n!, n)
    d2 = ((RY - rx[None]) ** 2).sum(1); rhos = 1 - 6 * d2 / (n * (n * n - 1))
    iu, ju = np.triu_indices(n, 1)
    sx = np.sign(rx[ju] - rx[iu]); taus = (np.sign(RY[:, ju] - RY[:, iu]) * sx[None]).sum(1) / len(iu)
    r0 = 1 - 6 * ((ry - rx) ** 2).sum() / (n * (n * n - 1)); t0 = float((np.sign(ry[ju] - ry[iu]) * sx).sum() / len(iu))
    eps = 1e-12
    return dict(n_perm=int(len(perms)), rho=float(r0), p_rho_two_sided=float(np.mean(np.abs(rhos) >= abs(r0) - eps)),
                p_rho_one_sided_negative=float(np.mean(rhos <= r0 + eps)),
                tau=t0, p_tau_two_sided=float(np.mean(np.abs(taus) >= abs(t0) - eps)),
                p_tau_one_sided_negative=float(np.mean(taus <= t0 + eps)))


def leave_one_out(point, families, a, b):
    """Rank correlations with each family removed in turn.

    Parameters
    ----------
    point : dict
        protocol -> {family: error}.
    families : list of str
        Families.
    a, b : str
        The two protocols.

    Returns
    -------
    out : dict
        family removed -> dict(rho, tau).
    """
    out = {}
    for f in families:
        x = [point[a][g] for g in families if g != f]; y = [point[b][g] for g in families if g != f]
        out[f] = dict(rho=spearman(x, y), tau=kendall(x, y))
    return out


def draw_error(runs, rng):
    """One bootstrap draw of the error of a family: one seed, shells resampled.

    Parameters
    ----------
    runs : list of numpy.ndarray
        Per-shell relative errors, one array per seed.
    rng : numpy.random.Generator
        Random stream.

    Returns
    -------
    error : float
        Median relative error of the resampled shells, in percent.
    """
    r = runs[rng.integers(len(runs))]
    return 100 * np.median(r[rng.integers(0, len(r), len(r))])


def shell_bootstrap(E, families, pairs, n_boot=2000, seed=0):
    """Interval of rho with the families fixed, seeds and shells resampled. The pairs share one random stream.

    Parameters
    ----------
    E : dict
        Output of load_errors.
    families : list of str
        Families, in the order they are drawn.
    pairs : list of (str, str)
        Pairs of protocols, in the order they are drawn.
    n_boot : int, default=2000
        Resamples per pair.
    seed : int, default=0
        Seed of the stream.

    Returns
    -------
    ci : dict
        (a, b) -> [lo, hi], the 95 % percentile interval of rho.
    """
    rng = np.random.default_rng(seed); out = {}
    for a, b in pairs:
        bs = []
        for _ in range(n_boot):
            e1 = [draw_error(E[(f, a)], rng) for f in families]; e2 = [draw_error(E[(f, b)], rng) for f in families]
            bs.append(spearman(e1, e2))
        lo, hi = np.percentile(bs, [2.5, 97.5]); out[(a, b)] = [float(lo), float(hi)]
    return out


def family_bootstrap(E, families, a, b, n_boot=2000, seed=0):
    """Interval of rho and tau with families, seeds and shells resampled.

    Parameters
    ----------
    E : dict
        Output of load_errors.
    families : list of str
        Families.
    a, b : str
        The two protocols.
    n_boot : int, default=2000
        Resamples.
    seed : int, default=0
        Seed.

    Returns
    -------
    out : dict
        n_boot, n_valid (resamples with at least three distinct families), rho_ci, tau_ci and the shares of
        negative rho and tau.
    """
    rng = np.random.default_rng(seed); R, T = [], []
    for _ in range(n_boot):
        fs = [families[i] for i in rng.integers(0, len(families), len(families))]
        x = [draw_error(E[(f, a)], rng) for f in fs]; y = [draw_error(E[(f, b)], rng) for f in fs]
        if len(set(fs)) < 3: R.append(np.nan); T.append(np.nan); continue
        R.append(spearman(x, y)); T.append(kendall(x, y))
    R, T = np.array(R), np.array(T); ok = np.isfinite(R)
    return dict(n_boot=n_boot, n_valid=int(ok.sum()),
                rho_ci=[float(np.percentile(R[ok], 2.5)), float(np.percentile(R[ok], 97.5))],
                tau_ci=[float(np.percentile(T[ok], 2.5)), float(np.percentile(T[ok], 97.5))],
                frac_rho_negative=float(np.mean(R[ok] < 0)), frac_tau_negative=float(np.mean(T[ok] < 0)))


def rank_table(point, families, protocols):
    """Rank of every family on every protocol (1 = lowest error).

    Parameters
    ----------
    point : dict
        protocol -> {family: error}.
    families : list of str
        Families.
    protocols : list of str
        Protocols.

    Returns
    -------
    ranks : dict
        family -> {protocol: rank}.
    """
    from scipy.stats import rankdata
    R = {b: rankdata([point[b][f] for f in families]) for b in protocols}
    return {f: {b: float(R[b][i]) for b in protocols} for i, f in enumerate(families)}


def compare_protocols(point, families, a, b, max_exact=9):
    """All the statistics of one pair of protocols from the table of errors.

    Parameters
    ----------
    point : dict
        protocol -> {family: error}.
    families : list of str
        Families.
    a, b : str
        The two protocols.
    max_exact : int, default=9
        Largest number of families for which the n! orderings are enumerated.

    Returns
    -------
    out : dict
        rho, tau, n_families, exact (None above max_exact families) and leave_one_out.
    """
    x = [point[a][f] for f in families]; y = [point[b][f] for f in families]
    out = dict(rho=spearman(x, y), tau=kendall(x, y), n_families=len(families),
               exact=exact_permutation(np.array(x), np.array(y)) if len(families) <= max_exact else None)
    if len(families) > 3:
        out["leave_one_out"] = leave_one_out(point, families, a, b)
        out["leave_one_out_rho_range"] = [min(v["rho"] for v in out["leave_one_out"].values()), max(v["rho"] for v in out["leave_one_out"].values())]
    return out


def main():
    """Compare the ranking of the families across protocols and write outputs/ranking.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=RUNS + "/final_runs", help="folders <family>_<protocol>_n8192 with the test predictions")
    ap.add_argument("--pattern", default=PATTERN, help="name of the folder of a family and protocol")
    ap.add_argument("--table", default="", help="csv with the columns family, protocol, error, instead of prediction files")
    ap.add_argument("--protocols", default="", help="comma-separated; default B1,B2,B3, or every protocol of --table")
    ap.add_argument("--reference", default="", help="protocol every other one is compared with; default the first")
    ap.add_argument("--families", default="", help="comma-separated; default every family found")
    ap.add_argument("--exclude", default="", help="comma-separated families to leave out, e.g. mlp,mlp80 for the models that read w")
    ap.add_argument("--all-pairs", action="store_true", help="also compare the protocols other than the reference with each other")
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-exact", type=int, default=9, help="largest number of families for the exact permutation test")
    ap.add_argument("--no-family-bootstrap", action="store_true")
    ap.add_argument("--out", default=H + "/ranking.json")
    a = ap.parse_args()
    E = None
    if a.table:
        point = read_table(a.table); prot = a.protocols.split(",") if a.protocols else list(point)
        fams = sorted(set.intersection(*[set(point[b]) for b in prot]))
    else:
        prot = (a.protocols or "B1,B2,B3").split(",")
        fams = discover_families(a.runs_dir, a.reference or prot[0], a.pattern)
    if a.families: fams = [f for f in fams if f in a.families.split(",")]
    fams = [f for f in fams if f not in a.exclude.split(",")]
    if not a.table:
        E = load_errors(a.runs_dir, fams, prot, a.pattern)
        fams = [f for f in fams if all(E[(f, b)] for b in prot)]
        point = point_errors(E, fams, prot)
    ref = a.reference or prot[0]; others = [b for b in prot if b != ref]
    if len(fams) < 3: sys.exit("fewer than three families with an error on every protocol (%s)" % ", ".join(prot))
    pairs = [(ref, b) for b in others] + (list(itertools.combinations(others, 2)) if a.all_pairs else [])
    out = dict(families=fams, point={b: {f: point[b][f] for f in fams} for b in prot}, ranks=rank_table(point, fams, prot))
    ci = {}
    if E is not None and a.n_boot:
        ci = shell_bootstrap(E, fams, [(ref, b) for b in others], a.n_boot, a.seed)       # one stream for the reference pairs
        for p in pairs[len(others):]: ci.update(shell_bootstrap(E, fams, [p], a.n_boot, a.seed))
    print("error per family (lower is better)\n%-14s %s" % ("family", " ".join("%12s" % b for b in prot)))
    for f in fams: print("%-14s %s" % (f, " ".join("%12.4g" % point[b][f] for b in prot)))
    for p in pairs:
        r = compare_protocols(point, fams, p[0], p[1], a.max_exact)
        s = "%s vs %s: Spearman %.2f" % (p[0], p[1], r["rho"])
        if p in ci: r["ci"] = ci[p]; s += " [%.2f, %.2f]" % tuple(ci[p])
        s += " over %d families" % len(fams)
        if r["exact"]: s += " | Kendall %.2f | exact permutation p %.3f (rho), %.3f (tau)" % (r["tau"], r["exact"]["p_rho_two_sided"], r["exact"]["p_tau_two_sided"])
        if "leave_one_out" in r: s += " | leave one out rho %.2f to %.2f" % tuple(r["leave_one_out_rho_range"])
        if E is not None and a.n_boot and not a.no_family_bootstrap:
            r["family_bootstrap"] = family_bootstrap(E, fams, p[0], p[1], a.n_boot, a.seed)
            s += " | families resampled [%.2f, %.2f]" % tuple(r["family_bootstrap"]["rho_ci"])
        out["%s_vs_%s" % p] = r; print(s, flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
