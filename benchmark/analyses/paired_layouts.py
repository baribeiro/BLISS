"""Paired layouts: the same defects placed 10 deg or 25 deg apart inside the same footprint, B3 models.

    python benchmark/analyses/paired_layouts.py [--compact cut_dense] [--clustered layout_10deg] [--wide layout_25deg]
                                  [--families figconv,mgn,transformer,transolver,deeponet] [--seeds 0,1,2]
                                  [--in-dir outputs] [--n-boot 2000]
                                  [--dense-ids all | FILE] [--exclude-sets SETS]

The count-matched shells of count_matched.py hold their defects in one compact patch, so their footprint on the shell
differs from that of a sparse shell. Three sets with the same source shell, the same number of defects, the same depths
and widths and the same seed defect separate the local spacing from the footprint:
  compact    one patch, nearest neighbours 10 deg apart (the count-matched shells; small footprint)
  wide       defects 25 deg apart                        (sparse spacing; large footprint)
  clustered  clusters of 3 to 6 defects 10 deg apart, the clusters at least 25 deg apart, the footprint matched per
             shell to the paired wide shell               (dense spacing; large footprint)
Shells of the three sets with the same `source` are the paired layouts of one shell. Clustered against wide changes the
local spacing only.

INPUTS
  <edited-dir>/<set>.npz                             prepared arrays of the three sets, with the list of defects
                                                     (contract in edited_shells.py); the labels and the footprints
                                                     are read from them
  <in-dir>/edited_pred_<fam>_b3_<set>_s<seed>.npz    predictions of the B3 finals (edited_shells.py). For the compact
                                                     set only the shells paired to the wide set are scored, whichever
                                                     shells the prediction file holds.
  <runs>/final_runs/<fam>_B3_n8192/*_{val,test}_pred.npz    published predictions of the same models, as count_matched.py
  --dense-ids                                        the dense reference set, as count_matched.py. Default "all", the
                                                     full B3 test set of splits/B3.json. A json list of ids selects
                                                     another set, read relative to BLISS_ROOT.
  --exclude-sets                                     optional: sets of edited shells whose sources are also left out.
                                                     Empty by default, no file is read.
  <in-dir>/count_matched.json                        optional: E_cs of every count-matched shell, copied for reference

DEFINITIONS
  Labels (solver): over the clustered shells paired in all three sets, the median kappa of each layout, the median
  of the per-shell ratios kappa_clustered / kappa_wide and kappa_clustered / kappa_compact, and their Spearman
  correlations. The label change between the paired layouts is the first ratio.
  Footprints (per set, from the defect centres; the seed defect is the first defect of a shell): the largest angle
  from the seed (median, range, 5th and 95th percentile over shells), the median angle from the seed, the
  nearest-neighbour spacing over all defects, the largest polar angle; for the clustered set the difference of its
  largest angle from the seed to that of the paired wide shell.
  Models (per family and layout): e = |kappa_pred / kappa_true - 1| * 100; E = mean over seeds of the median over
  shells; signed bias = the same without the absolute value; field R^2 and site within 10 deg as edited_shells.py.
  S = (E - E_sp) / (E_de - E_sp) with E_sp, E_de the sparse validation and dense reference errors of count_matched.py.
  Interval: shell bootstrap, 2,000 resamples, one generator with seed 0 for the whole run; families in the order of
  --families and, inside a family, the layouts in the order clustered, wide, compact; in each resample the shells are
  drawn with replacement within each seed in the order E_sp, E_de, E; 2.5 and 97.5 percentiles.
Output: <out-dir>/paired_layouts.json.

Functions
---------
restrict
    Predictions of a set restricted to some shells.
layout_stats
    Errors, bias, field and site of one layout.
labels
    Solver labels of the paired layouts.
nearest_neighbour
    Angle from every defect to its nearest neighbour.
footprints
    Footprint statistics of every layout from the defect centres.
main
    Score the paired layouts and write paired_layouts.json.
"""
import argparse, json, os, sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edited_shells as ES   # noqa: E402

ORDER = ("clustered", "wide", "compact")          # order of the bootstrap stream inside a family


def restrict(D, keep):
    """Predictions of a set restricted to some shells.

    Parameters
    ----------
    D : dict
        Seed -> prediction (edited_shells.load_pred).
    keep : set of int
        Shell ids to keep.

    Returns
    -------
    D : dict
        The same, with the per-shell arrays masked.
    """
    out = {}
    for s, v in D.items():
        m = np.array([int(i) in keep for i in v["ids"]])
        out[s] = dict(v, kp=v["kp"][m], field=v["field"][m], site=v["site"][m], ids=v["ids"][m], KT=v["KT"][m], source=v["source"][m])
    return out


def layout_stats(D):
    """Errors, bias, field and site of one layout.

    Parameters
    ----------
    D : dict
        Seed -> prediction.

    Returns
    -------
    r : dict
        n_seeds, n_shells, E, E_per_seed, signed_bias, field_r2, site10 and the files read.
    """
    if not D: return dict(n_seeds=0, n_shells=0, E=None, E_per_seed={}, signed_bias=None, field_r2=None, site10=None, files=[])
    fr, st = ES.seed_mean_scores([v["field"] for v in D.values()], [v["site"] for v in D.values()])
    return dict(n_seeds=len(D), n_shells=int(len(next(iter(D.values()))["KT"])),
                E=ES.seed_mean_median([ES.relerr(v["kp"], v["KT"]) for v in D.values()]),
                E_per_seed={int(s): float(np.median(ES.relerr(v["kp"], v["KT"]))) for s, v in sorted(D.items())},
                signed_bias=ES.seed_mean_median([100 * (v["kp"] / v["KT"] - 1) for v in D.values()]),
                field_r2=fr, site10=st, files=sorted({v["src"] for v in D.values()}))


def labels(C, W, P):
    """Solver labels of the paired layouts.

    Parameters
    ----------
    C, W, P : dict
        Prepared arrays of the clustered, wide and compact sets (edited_shells.load_set).

    Returns
    -------
    lab : dict
        Medians, ratios and rank correlations of kappa over the paired shells.
    pairs : list of tuple
        (clustered id, wide id, compact id) of every clustered shell.
    """
    kd = lambda S: {int(i): float(k) for i, k in zip(S["ids"], S["kappa"]) if np.isfinite(k)}
    by = lambda S: {int(s): int(i) for i, s in zip(S["ids"], S["source"])}
    Kc, Kw, Kp = kd(C), kd(W), kd(P); sw, sp = by(W), by(P)
    pairs = [(int(c), sw[int(s)], sp[int(s)]) for c, s in sorted(zip(C["ids"], C["source"]))]
    ok = [(c, w, p) for c, w, p in pairs if c in Kc and w in Kw and p in Kp]
    kc = np.array([Kc[c] for c, _, _ in ok]); kw = np.array([Kw[w] for _, w, _ in ok]); kp = np.array([Kp[p] for _, _, p in ok])
    return dict(n_clustered_with_kappa=len(Kc), n_paired=len(ok), kappa_clustered_median=float(np.median(kc)),
                kappa_clustered_p5_p95=[float(np.percentile(kc, 5)), float(np.percentile(kc, 95))],
                kappa_wide_paired_median=float(np.median(kw)), kappa_compact_paired_median=float(np.median(kp)),
                ratio_clustered_over_wide_median=float(np.median(kc / kw)), ratio_clustered_over_compact_median=float(np.median(kc / kp)),
                spearman_clustered_vs_wide=float(spearmanr(kc, kw)[0]), spearman_clustered_vs_compact=float(spearmanr(kc, kp)[0])), pairs


def nearest_neighbour(U):
    """Angle from every defect to its nearest neighbour.

    Parameters
    ----------
    U : numpy.ndarray, shape (K, 3)
        Unit centres of the defects of one shell.

    Returns
    -------
    a : numpy.ndarray, shape (K,)
        Angles in degrees.
    """
    A = np.degrees(np.arccos(np.clip(U @ U.T, -1, 1))); np.fill_diagonal(A, 1e9); return A.min(1)


def footprints(C, W, P, pairs):
    """Footprint statistics of every layout from the defect centres.

    Parameters
    ----------
    C, W, P : dict
        Prepared arrays of the clustered, wide and compact sets, with their defects.
    pairs : list of tuple
        (clustered id, wide id, compact id) from labels().

    Returns
    -------
    foot : dict
        Statistics per layout.
    """
    ang = lambda U, c: np.degrees(np.arccos(np.clip(U @ c, -1, 1)))
    keep = {p for _, _, p in pairs}; foot = {}; fmax = {}
    for name, S, sel in (("compact", P, keep), ("clustered", C, None), ("wide", W, None)):
        D = {k: v for k, v in S["defects"].items() if sel is None or k in sel}
        fp = np.array([ang(U, U[0]).max() for U in D.values()]); fmax[name] = dict(zip(D, fp))
        md = np.array([np.median(ang(U[1:], U[0])) for U in D.values()])
        nnall = np.concatenate([nearest_neighbour(U) for U in D.values()])
        pol = np.degrees(np.arccos(np.clip(np.concatenate(list(D.values()))[:, 2], -1, 1)))
        foot[name] = dict(n=len(D), footprint_max_from_seed_median=float(np.median(fp)), footprint_max_range=[float(fp.min()), float(fp.max())],
                          footprint_max_p5_p95=[float(np.percentile(fp, 5)), float(np.percentile(fp, 95))],
                          median_dist_from_seed_median=float(np.median(md)), median_dist_from_seed_range=[float(md.min()), float(md.max())],
                          nn_median=float(np.median(nnall)), nn_p5_p95=[float(np.percentile(nnall, 5)), float(np.percentile(nnall, 95))],
                          nn_min=float(nnall.min()), nn_max=float(nnall.max()), max_polar=float(pol.max()))
    d = np.array([fmax["clustered"][c] - fmax["wide"][w] for c, w, _ in pairs])
    foot["clustered"]["footprint_minus_paired_wide"] = dict(median=float(np.median(d)), max_abs=float(np.abs(d).max()))
    return foot


def main():
    """Score the paired layouts and write paired_layouts.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compact", default="cut_dense", help="set with one compact patch at 10 deg (the count-matched shells)")
    ap.add_argument("--clustered", default="layout_10deg", help="set with clusters at 10 deg inside the wide footprint")
    ap.add_argument("--wide", default="layout_25deg", help="set with defects 25 deg apart")
    ap.add_argument("--families", default=",".join(ES.FAMILIES)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--in-dir", default=ES.H, help="directory of the edited_pred files")
    ap.add_argument("--runs", default=ES.RUNS, help="directory of the trained models (holds final_runs/)")
    ap.add_argument("--dense-ids", default="all", help='"all" for the full B3 test set of splits/B3.json, or a json list of the ids of the dense reference set')
    ap.add_argument("--exclude-sets", default="", help="optional: sets of edited shells whose sources are also left out of the dense reference set")
    ap.add_argument("--edited-dir", default=ES.EDITED); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=ES.H)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    C, W, P = (ES.load_set(n, a.edited_dir) for n in (a.clustered, a.wide, a.compact))
    lab, pairs = labels(C, W, P); ES.log("labels", json.dumps(lab))
    foot = footprints(C, W, P, pairs) if all("defects" in S for S in (C, W, P)) else {}
    for k, v in foot.items(): ES.log("footprint %-9s n %d  max from seed %.1f deg  nearest neighbour %.1f deg" % (k, v["n"], v["footprint_max_from_seed_median"], v["nn_median"]))
    src_w = set(int(s) for s in W["source"]); keep = {int(i) for i, s in zip(P["ids"], P["source"]) if int(s) in src_w}
    te = ES.dense_test_ids(a.dense_ids, a.exclude_sets, a.edited_dir)
    cm = a.in_dir + "/count_matched.json"; G = json.load(open(cm)) if os.path.exists(cm) else {}
    rng = np.random.default_rng(0); fams = {}
    for fam in a.families.split(","):
        L = dict(clustered=ES.load_pred(a.in_dir, fam, "b3", a.clustered, seeds), wide=ES.load_pred(a.in_dir, fam, "b3", a.wide, seeds),
                 compact=restrict(ES.load_pred(a.in_dir, fam, "b3", a.compact, seeds), keep))
        ref = L["clustered"] or L["wide"]
        if not ref:
            ES.log("skip %s (no prediction on %s or %s)" % (fam, a.clustered, a.wide)); continue
        E = {s: ES.reference_errors(fam, s, te, a.runs) for s in sorted(set().union(*L.values()))}
        E_sp = ES.seed_mean_median([E[s][0] for s in ref]); E_de = ES.seed_mean_median([E[s][1] for s in ref])
        r = dict(E_sp=E_sp, E_de=E_de, **{k: layout_stats(L[k]) for k in ("compact", "clustered", "wide")})
        if fam in G: r.update(E_count_matched_all=G[fam]["E_cs"], field_r2_count_matched_all=G[fam]["field_r2_cs"], site10_count_matched_all=G[fam]["site10_cs"])
        for name in ORDER:
            D = L[name]
            if not D or r[name]["E"] is None: continue
            r[name]["S"] = (r[name]["E"] - E_sp) / (E_de - E_sp)
            ss = sorted(D); boot = []
            for _ in range(a.n_boot):
                b = lambda src: ES.seed_mean_median([rng.choice(x, len(x)) for x in src])
                bs = b([E[s][0] for s in ss]); bd = b([E[s][1] for s in ss])
                bx = b([ES.relerr(D[s]["kp"], D[s]["KT"]) for s in ss]); boot.append((bx - bs) / (bd - bs))
            r[name]["S_ci"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        fams[fam] = r
        ES.log("%-12s E_sp %.2f  %s  E_de %.2f" % (fam, E_sp, "  ".join(
            "%s %s" % (k, "%.2f (S %.2f, %d seeds)" % (r[k]["E"], r[k]["S"], r[k]["n_seeds"]) if r[k]["E"] is not None else "-")
            for k in ("compact", "clustered", "wide")), E_de))
    os.makedirs(a.out_dir, exist_ok=True); f = a.out_dir + "/paired_layouts.json"
    json.dump(dict(labels=lab, footprints=foot, families=fams, sets=dict(compact=a.compact, clustered=a.clustered, wide=a.wide)), open(f, "w"), indent=1)
    ES.log("wrote " + f)


if __name__ == "__main__":
    main()
