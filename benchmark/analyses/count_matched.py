"""Defect count against spacing: dense test shells cut to the sparse count, and the spacing share of the B3 error.

    python benchmark/analyses/count_matched.py [--set cut_dense] [--families figconv,mgn,transformer,transolver,deeponet]
                                  [--seeds 0,1,2] [--in-dir outputs] [--n-boot 2000]
                                  [--dense-ids all | FILE] [--exclude-sets SETS]

B3 trains on sparse shells (11 to 18 defects, at least 25 deg apart) and tests on dense shells (69 to 81 defects, 10 deg
apart), so two things change together: the number of defects and their spacing. The count-matched set removes the
first: each of its shells keeps the K nearest defects of a dense test shell, K drawn in the sparse range, and was
simulated again. It has the sparse count at the dense spacing.

INPUTS
  <in-dir>/edited_pred_<fam>_b3_<set>_s<seed>.npz    predictions of the B3 finals on the count-matched shells
                                                      (edited_shells.py; the prepared arrays of the edited shells
                                                      are its input, see the contract in its docstring)
  <runs>/final_runs/<fam>_B3_n8192/*_{val,test}_pred.npz   published predictions of the same models (transfer.py)
  --dense-ids                                         the dense reference set. Default "all", the full B3 test set
                                                      of splits/B3.json. A json list of ids selects another set,
                                                      read relative to BLISS_ROOT.
  <edited-dir>/<set>.npz of --exclude-sets            optional: the source shells of these sets are also left out of
                                                      the dense reference set. Empty by default, no file is read.

DEFINITIONS (per family; a model = one seed)
  e = |kappa_pred / kappa_true - 1| * 100 per shell; E = mean over seeds of the median over shells.
  E_sp   on the sparse validation shells of B3 (sparse count, sparse spacing)
  E_cs   on the count-matched shells            (sparse count, dense spacing)
  E_de   on the dense reference set             (dense count, dense spacing)
  Spacing share S = (E_cs - E_sp) / (E_de - E_sp): 0 if the count-matched shells are as easy as the sparse ones (the
  error comes with the count), 1 if they are as hard as the dense ones (the error comes with the spacing).
  Interval: shell bootstrap, 2,000 resamples, one generator with seed 0 for the whole run; in each resample the shells
  of each set are drawn with replacement within each seed, in the order E_sp, E_cs, E_de, and S is recomputed; 2.5 and
  97.5 percentiles. Families are processed in the order of --families, which fixes the random stream.
  Summary: the median of S over the families, the interval of that median (the median over families taken in each
  resample). Criterion reported: spacing is the main driver iff median S >= 0.5 and the lower end of its interval is
  above 0.25.
  Also reported on the count-matched shells: field R^2 (mean over seeds of the median over shells) and site within
  10 deg (mean over seeds of the share of shells), as defined in edited_shells.py.
  The count-matched shells hold their defects in one compact patch; paired_layouts.py controls for that footprint.
Output: <out-dir>/count_matched.json.

Functions
---------
spacing_share
    Spacing share of one family with its bootstrap draws.
main
    Score the count-matched shells and write count_matched.json.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edited_shells as ES   # noqa: E402

S_MIN, S_LOW_MIN = 0.5, 0.25          # criterion on the median spacing share and on the lower end of its interval


def spacing_share(rows, n_boot, rng):
    """Spacing share of one family with its bootstrap draws.

    Parameters
    ----------
    rows : list of dict
        One dict per seed with the per-shell errors e_sp, e_cs, e_de.
    n_boot : int
        Resamples.
    rng : numpy.random.Generator
        Generator shared by the families of a run.

    Returns
    -------
    E : dict
        E_sp, E_cs, E_de, S and S_ci.
    boot : list of float
        S of every resample.
    """
    med = lambda k: ES.seed_mean_median([r[k] for r in rows])
    Es, Ec, Ed = med("e_sp"), med("e_cs"), med("e_de")
    boot = []
    for _ in range(n_boot):
        b = lambda k: ES.seed_mean_median([rng.choice(r[k], len(r[k])) for r in rows])
        bs, bc, bd = b("e_sp"), b("e_cs"), b("e_de"); boot.append((bc - bs) / (bd - bs))
    return dict(E_sp=Es, E_cs=Ec, E_de=Ed, S=(Ec - Es) / (Ed - Es), S_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]), boot


def main():
    """Score the count-matched shells and write count_matched.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="cut_dense", help="set of count-matched shells")
    ap.add_argument("--families", default=",".join(ES.FAMILIES)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--in-dir", default=ES.H, help="directory of the edited_pred files")
    ap.add_argument("--runs", default=ES.RUNS, help="directory of the trained models (holds final_runs/)")
    ap.add_argument("--dense-ids", default="all", help='"all" for the full B3 test set of splits/B3.json, or a json list of the ids of the dense reference set')
    ap.add_argument("--exclude-sets", default="", help="optional: sets of edited shells whose sources are also left out of the dense reference set")
    ap.add_argument("--edited-dir", default=ES.EDITED); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", default=ES.H)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    te = ES.dense_test_ids(a.dense_ids, a.exclude_sets, a.edited_dir)
    rng = np.random.default_rng(0); out, boots, n_cs = {}, {}, 0
    for fam in a.families.split(","):
        P = ES.load_pred(a.in_dir, fam, "b3", a.set, seeds)
        if not P:
            ES.log("skip %s (no prediction on %s)" % (fam, a.set)); continue
        rows = []
        for s in sorted(P):
            e_sp, e_de = ES.reference_errors(fam, s, te, a.runs)
            rows.append(dict(seed=s, e_cs=ES.relerr(P[s]["kp"], P[s]["KT"]), e_sp=e_sp, e_de=e_de, field_cs=P[s]["field"], site_cs=P[s]["site"]))
            n_cs = int(np.isfinite(P[s]["KT"]).sum())
        r, boots[fam] = spacing_share(rows, a.n_boot, rng)
        fr, st = ES.seed_mean_scores([x["field_cs"] for x in rows], [x["site_cs"] for x in rows])
        out[fam] = dict(n_seeds=len(rows), n_sp=int(len(rows[0]["e_sp"])), n_cs=int(len(rows[0]["e_cs"])), n_de=int(len(rows[0]["e_de"])), **r,
                        field_r2_cs=fr, site10_cs=st)
        ES.log("%-12s E_sp %.2f  E_cs %.2f  E_de %.2f   S %.3f [%.3f, %.3f]   field R2 %.3f  site10 %.3f  (%d seeds)"
               % (fam, r["E_sp"], r["E_cs"], r["E_de"], r["S"], r["S_ci"][0], r["S_ci"][1], fr, st, len(rows)))
    if not out: raise SystemExit("no predictions under %s: run edited_shells.py --sets %s --variant b3" % (a.in_dir, a.set))
    fams = list(out); medS = float(np.median([out[f]["S"] for f in fams]))
    bootmed = np.median(np.array([boots[f] for f in fams]), axis=0); lo, hi = float(np.percentile(bootmed, 2.5)), float(np.percentile(bootmed, 97.5))
    summary = dict(families=fams, median_S=medS, median_S_ci=[lo, hi], spacing_is_main_driver=bool(medS >= S_MIN and lo > S_LOW_MIN),
                   n_cs=n_cs, n_dense_test=len(te), n_boot=a.n_boot, set=a.set)
    os.makedirs(a.out_dir, exist_ok=True); f = a.out_dir + "/count_matched.json"
    json.dump(dict(summary=summary, **out), open(f, "w"), indent=1)
    ES.log("median S %.3f [%.3f, %.3f] over %d families; spacing is the main driver: %s -> %s" % (medS, lo, hi, len(fams), summary["spacing_is_main_driver"], f))


if __name__ == "__main__":
    main()
