"""Failure-site distribution: B3 kappa error by where the shell collapses, and the dense error re-weighted to the
sparse distribution of sites.

    python benchmark/analyses/site_reweighting.py [--families deeponet,mgn,transformer,figconv,transolver,sfno]
                                  [--dense-ids splits/analysis/controlled_tests_dense.json | all] [--exclude-sets SETS]
                                  [--bins 0,30,50,70,90] [--min-shells 30] [--min-families 4]

Dense shells collapse closer to the rim than sparse shells do. If the B3 error only followed the site, the dense error
would fall to the sparse one once the two sets are compared at the same site. This script compares them bin by bin and
re-weights the dense reference set to the sparse histogram of sites. No inference and no edited shell: published predictions
only.

INPUTS
  <runs>/final_runs/<fam>_B3_n8192/*_{val,test}_pred.npz   published predictions of the B3 finals (transfer.py): ids,
                                                           pred_K, true_K, true_U and the unit node directions P.
                                                           Files that are not on 8,192 nodes are skipped.
  --dense-ids                                              json list of the ids of the dense reference set, as
                                                           count_matched.py. Default
                                                           splits/analysis/controlled_tests_dense.json, read relative
                                                           to BLISS_ROOT: the B3 test shells outside the 1,000-shell
                                                           few-shot adaptation pool, without the 31 shells that are
                                                           also sources of the neighbour probe (1,169 shells). "all" =
                                                           the full B3 test set of splits/B3.json.
  --exclude-sets                                           optional: sets of edited shells whose sources are also left
                                                           out (<edited-dir>/<set>.npz). Empty by default, no file is
                                                           read.

DEFINITIONS
  Site of a shell (solver): the node of largest |u_n - mean(u_n)|, u_n = U . P the normal displacement at the limit
  point, over the nodes with polar angle below 88 deg; its polar angle from the pole (deg) is the site angle.
  e = |kappa_pred / kappa_true - 1| * 100 per shell. Dense = the dense reference set, sparse = the B3 validation shells.
  Bins of the site angle: [0, 30), [30, 50), [50, 70), [70, 90). Per bin and family: the mean over seeds of the median e
  of the dense and of the sparse shells in the bin, and their ratio. A bin is eligible if it holds at least 30 shells
  of both sets. Criterion reported: the error is not explained by the site iff in every eligible bin the dense error
  is more than twice the sparse one for at least 4 families.
  Re-weighting: every dense shell of a bin gets the weight (share of sparse shells in the bin) / (share of dense shells
  in the bin); the re-weighted dense error is the weighted median of e (the first shell, in increasing e, at which the
  cumulative weight reaches half), mean over seeds. Printed next to the plain dense and sparse medians.
Output: <out-dir>/site_reweighting.json.

Functions
---------
site_angles
    Polar angle of the solver's failure site of every shell of a prediction file.
load_family
    Site angles and kappa errors of the dense test and sparse validation shells, per seed.
by_bin
    Dense against sparse error in every bin of the site angle.
reweighted
    Dense error re-weighted to the sparse histogram of sites.
main
    Score the site bins and the re-weighting and write site_reweighting.json.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edited_shells as ES   # noqa: E402
from bliss.geometry import frame, pole_axis, R   # noqa: E402

FAMILIES = ["deeponet", "mgn", "transformer", "figconv", "transolver", "sfno"]
CAP_DEG = 88.0


def site_angles(z, idx, cache):
    """Polar angle of the solver's failure site of every shell of a prediction file.

    Parameters
    ----------
    z : mapping
        Prediction file (ids, true_U, P).
    idx : list of int
        Rows to read.
    cache : dict
        Shell id -> site angle, filled as shells are read (the truth does not depend on the model).

    Returns
    -------
    pol : numpy.ndarray
        Site angle per row (deg).
    """
    P = z["P"].astype(float); n = P / np.linalg.norm(P, axis=1, keepdims=True)
    pole = frame(pole_axis(P * R)[0])[2]; pol = np.degrees(np.arccos(np.clip(n @ pole, -1, 1))); cap = pol < CAP_DEG
    ids = z["ids"]; U = z["true_U"] if any(int(ids[j]) not in cache for j in idx) else None
    for j in idx:
        i = int(ids[j])
        if i not in cache:
            un = (U[j] * n).sum(1)[cap]; cache[i] = float(pol[cap][int(np.argmax(np.abs(un - un.mean())))])
    return np.array([cache[int(ids[j])] for j in idx])


def load_family(fam, te, runs, cache):
    """Site angles and kappa errors of the dense test and sparse validation shells, per seed.

    Parameters
    ----------
    fam : str
        Model family.
    te : set of int
        Ids of the dense test set.
    runs : str
        Directory of the trained models.
    cache : dict
        Shell id -> site angle.

    Returns
    -------
    rows : list of tuple
        (dense site angles, dense errors, sparse site angles, sparse errors), one per seed.
    """
    rows = []
    for s, (fv, ft) in ES.published_predictions(fam, runs).items():
        zt, zv = np.load(ft), np.load(fv)
        def part(z, keep):
            """Site angles and errors of the rows of a file that pass a filter.

            Parameters
            ----------
            z : mapping
                Prediction file.
            keep : callable
                Filter on the shell id.

            Returns
            -------
            pol, e : numpy.ndarray
                Site angle and kappa error per shell.
            """
            idx = [j for j, i in enumerate(z["ids"]) if keep(int(i))]
            return site_angles(z, idx, cache), ES.relerr(z["pred_K"][idx], z["true_K"][idx])
        if zt["P"].shape[0] != 8192: continue
        rows.append(part(zt, lambda i: i in te) + part(zv, lambda i: True))
    return rows


def by_bin(fam_rows, bins, min_shells):
    """Dense against sparse error in every bin of the site angle.

    Parameters
    ----------
    fam_rows : dict
        Family -> rows of load_family.
    bins : list of tuple
        (lo, hi) of every bin, degrees.
    min_shells : int
        Shells of both sets a bin needs to be eligible.

    Returns
    -------
    res : list of dict
        One record per bin.
    """
    res = []
    for lo, hi in bins:
        pd_, _, ps_, _ = next(iter(fam_rows.values()))[0]
        nd, ns = int(((pd_ >= lo) & (pd_ < hi)).sum()), int(((ps_ >= lo) & (ps_ < hi)).sum())
        ratios = {}
        for fam, rows in fam_rows.items():
            md = np.mean([np.median(ed[(pd_ >= lo) & (pd_ < hi)]) if ((pd_ >= lo) & (pd_ < hi)).any() else np.nan for pd_, ed, _, _ in rows])
            ms = np.mean([np.median(es[(ps_ >= lo) & (ps_ < hi)]) if ((ps_ >= lo) & (ps_ < hi)).any() else np.nan for _, _, ps_, es in rows])
            ratios[fam] = dict(dense=float(md), sparse=float(ms), ratio=float(md / ms) if ms else None)
        res.append(dict(bin=[lo, hi], n_dense=nd, n_sparse=ns, eligible=nd >= min_shells and ns >= min_shells, families=ratios,
                        n_fam_ratio_gt2=int(sum(1 for r in ratios.values() if r["ratio"] and r["ratio"] > 2))))
    return res


def reweighted(rows, bins):
    """Dense error re-weighted to the sparse histogram of sites.

    Parameters
    ----------
    rows : list of tuple
        Rows of load_family of one family.
    bins : list of tuple
        (lo, hi) of every bin, degrees.

    Returns
    -------
    r : dict
        dense_reweighted_median, dense_median and sparse_median (means over seeds).
    """
    vals = []
    for pd_, ed, ps_, es in rows:
        w = np.zeros(len(pd_))
        for lo, hi in bins:
            md, ms = (pd_ >= lo) & (pd_ < hi), (ps_ >= lo) & (ps_ < hi)
            if md.any(): w[md] = ms.mean() / md.mean()
        o = np.argsort(ed); cw = np.cumsum(w[o]) / w.sum(); vals.append(float(ed[o][np.searchsorted(cw, 0.5)]))
    return dict(dense_reweighted_median=float(np.mean(vals)), dense_median=float(np.mean([np.median(r[1]) for r in rows])),
                sparse_median=float(np.mean([np.median(r[3]) for r in rows])))


def main():
    """Score the site bins and the re-weighting and write site_reweighting.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--runs", default=ES.RUNS, help="directory of the trained models (holds final_runs/)")
    ap.add_argument("--dense-ids", default=ES.DENSE_IDS, help='json list of the ids of the dense reference set, or "all" for the full B3 test set of splits/B3.json')
    ap.add_argument("--exclude-sets", default="", help="optional: sets of edited shells whose sources are also left out of the dense reference set")
    ap.add_argument("--edited-dir", default=ES.EDITED)
    ap.add_argument("--bins", default="0,30,50,70,90", help="edges of the bins of the site angle (deg)")
    ap.add_argument("--min-shells", type=int, default=30, help="shells of both sets an eligible bin holds")
    ap.add_argument("--min-families", type=int, default=4, help="families with dense error > 2x sparse in every eligible bin")
    ap.add_argument("--out-dir", default=ES.H)
    a = ap.parse_args()
    edges = [float(x) if "." in x else int(x) for x in a.bins.split(",")]; bins = list(zip(edges[:-1], edges[1:]))
    te = ES.dense_test_ids(a.dense_ids, a.exclude_sets, a.edited_dir); cache = {}; fam_rows = {}
    for fam in a.families.split(","):
        rows = load_family(fam, te, a.runs, cache)
        if rows: fam_rows[fam] = rows
        else: ES.log("skip %s (no published B3 predictions on 8,192 nodes)" % fam)
    if not fam_rows: raise SystemExit("no published B3 predictions under %s/final_runs" % a.runs)
    binres = by_bin(fam_rows, bins, a.min_shells); elig = [b for b in binres if b["eligible"]]
    rew = {fam: reweighted(rows, bins) for fam, rows in fam_rows.items()}
    out = dict(bins=binres, reweighted=rew, n_eligible_bins=len(elig),
               not_explained_by_site=bool(elig) and all(b["n_fam_ratio_gt2"] >= a.min_families for b in elig),
               n_dense_test=len(te), n_seeds={fam: len(rows) for fam, rows in fam_rows.items()})
    for b in binres:
        ES.log("bin %s  dense %d  sparse %d  eligible %s  families with dense > 2x sparse: %d of %d"
               % (b["bin"], b["n_dense"], b["n_sparse"], b["eligible"], b["n_fam_ratio_gt2"], len(fam_rows)))
    for fam, r in rew.items():
        ES.log("%-12s dense %.2f %%  re-weighted to the sparse sites %.2f %%  sparse %.2f %%" % (fam, r["dense_median"], r["dense_reweighted_median"], r["sparse_median"]))
    os.makedirs(a.out_dir, exist_ok=True); f = a.out_dir + "/site_reweighting.json"
    json.dump(out, open(f, "w"), indent=1); ES.log("not explained by the site: %s -> %s" % (out["not_explained_by_site"], f))


if __name__ == "__main__":
    main()
