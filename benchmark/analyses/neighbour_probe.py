"""Neighbour probe: the deepest defect alone, with its neighbours, and the full shell; predicted against solver effect.

    python benchmark/analyses/neighbour_probe.py [--variants n8192,b3] [--families figconv,mgn,transformer,transolver,deeponet]
                                  [--seeds 0,1,2] [--in-dir outputs]
                                  [--sets dense:full=nb_dense_full,dense:single=nb_dense_single,dense:nb=nb_dense_nb,sparse:full=nb_sparse_full,sparse:single=nb_sparse_single]

Does a model read the interaction between defects, or only the deepest one? Each source shell was simulated again in
up to three versions that share its deepest defect:
  single   the deepest defect alone
  nb       the deepest defect and its neighbours within one buckling wavelength (4 to 10 defects: a sparse count at
           the dense spacing)
  full     every defect of the source shell
on 60 dense test shells of B1 (all three versions) and 60 sparse shells (full and single). The solver gives kappa of
every version, so the effect of adding defects is known per shell. The B1 finals (variant n8192, trained with dense
shells) and the B3 finals (variant b3, trained on sparse shells only) predict every version.

INPUTS
  <in-dir>/edited_pred_<fam>_<variant>_<set>_s<seed>.npz   predictions on every set of --sets (edited_shells.py; the
                                                           prepared arrays of the edited shells are its input, see
                                                           the contract in its docstring)
  --sets   regime:version=set, comma-separated. Versions of one regime are matched shell by shell through `source`.

DEFINITIONS (per family, variant and seed; nothing is averaged over seeds in the file)
  err_<regime>_<version>       median over shells of |kappa_pred - kappa_true| / kappa_true * 100
  fieldR2_<regime>_<version>   median over shells of the field R^2, site10_<regime>_<version> the share of shells whose
                               predicted site is within 10 deg of the solver's (edited_shells.py); NaN without fields
  <regime>_<a>/<b>             effect of going from version b to version a, for full/single, nb/single and full/nb:
       per shell, solver ratio s = kappa_true(a) / kappa_true(b) and predicted ratio p = kappa_pred(a) / kappa_pred(b),
       over the shells where s is finite;
       sim_median = median s, pred_median = median p, corr = Pearson correlation of p with s,
       sign = share of shells where p - 1 and s - 1 have the same sign.
Output: <out-dir>/neighbour_probe.json, {"<variant>_<family>": [one record per seed]}, and a table of seed means.

Functions
---------
parse_sets
    Sets of edited shells by regime and version.
aligned
    Stored predictions of the versions of one regime, on the same shells in the same order.
effect
    Predicted against solver effect of adding defects.
score_seed
    Record of one family, variant and seed.
main
    Score the neighbour probe and write neighbour_probe.json.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edited_shells as ES   # noqa: E402

SETS = "dense:full=nb_dense_full,dense:single=nb_dense_single,dense:nb=nb_dense_nb,sparse:full=nb_sparse_full,sparse:single=nb_sparse_single"
EFFECTS = (("full", "single"), ("nb", "single"), ("full", "nb"))


def parse_sets(text):
    """Sets of edited shells by regime and version.

    Parameters
    ----------
    text : str
        regime:version=set, comma-separated.

    Returns
    -------
    sets : dict
        (regime, version) -> set name, in the order given.
    """
    out = {}
    for item in [x for x in text.split(",") if x]:
        key, name = item.split("="); reg, ver = key.split(":"); out[(reg, ver)] = name
    return out


def aligned(P):
    """Stored predictions of the versions of one regime, on the same shells in the same order.

    Parameters
    ----------
    P : dict
        Version -> prediction of one seed (edited_shells.load_pred).

    Returns
    -------
    P : dict
        The same, every version reordered to the sources of the first one.
    """
    ref = next(iter(P.values()))["source"]; out = {}
    for ver, v in P.items():
        pos = {int(s): j for j, s in enumerate(v["source"])}
        assert len(pos) == len(v["source"]) and set(pos) == set(int(s) for s in ref), "versions of a regime must hold the same source shells"
        j = np.array([pos[int(s)] for s in ref]); out[ver] = {k: (x[j] if isinstance(x, np.ndarray) else x) for k, x in v.items()}
    return out


def effect(pa, pb, ta, tb):
    """Predicted against solver effect of adding defects (version b -> version a).

    Parameters
    ----------
    pa, pb : numpy.ndarray
        Predicted kappa of the two versions, per shell.
    ta, tb : numpy.ndarray
        Solver kappa of the two versions.

    Returns
    -------
    r : dict
        sim_median, pred_median, corr and sign.
    """
    ps = pa / pb; ss = ta / tb; ok = np.isfinite(ss)
    return dict(sim_median=float(np.median(ss[ok])), pred_median=float(np.median(ps[ok])), corr=float(np.corrcoef(ps[ok], ss[ok])[0, 1]),
                sign=float(np.mean(np.sign(ps[ok] - 1) == np.sign(ss[ok] - 1))))


def score_seed(P, seed):
    """Record of one family, variant and seed.

    Parameters
    ----------
    P : dict
        (regime, version) -> prediction of this seed.
    seed : int
        Training seed.

    Returns
    -------
    r : dict
        Errors, field and site per version and the effects per regime.
    """
    import warnings
    r = {"seed": seed}; regs = {}
    for (reg, ver), v in P.items(): regs.setdefault(reg, {})[ver] = v
    regs = {reg: aligned(V) for reg, V in regs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN field scores when the solver fields are not given
        for reg, ver in P:
            v = regs[reg][ver]
            r["err_%s_%s" % (reg, ver)] = float(100 * np.nanmedian(np.abs(v["kp"] - v["KT"]) / v["KT"]))
            r["fieldR2_%s_%s" % (reg, ver)] = float(np.nanmedian(v["field"])); r["site10_%s_%s" % (reg, ver)] = float(np.nanmean(v["site"]))
    for reg, V in regs.items():
        for a, b in EFFECTS:
            if a in V and b in V: r["%s_%s/%s" % (reg, a, b)] = effect(V[a]["kp"], V[b]["kp"], V[a]["KT"], V[b]["KT"])
    return r


def main():
    """Score the neighbour probe and write neighbour_probe.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", default="n8192,b3", help="n8192 = B1 finals, b3 = B3 finals")
    ap.add_argument("--families", default=",".join(ES.FAMILIES)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sets", default=SETS, help="regime:version=set of edited shells, comma-separated")
    ap.add_argument("--in-dir", default=ES.H, help="directory of the edited_pred files"); ap.add_argument("--out-dir", default=ES.H)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]; sets = parse_sets(a.sets); out = {}
    for variant in a.variants.split(","):
        for fam in a.families.split(","):
            L = {key: ES.load_pred(a.in_dir, fam, variant, name, seeds) for key, name in sets.items()}
            have = sorted(set.intersection(*[set(v) for v in L.values()]))
            if not have:
                ES.log("skip %s %s (no seed with a prediction on every set)" % (variant, fam)); continue
            res = [score_seed({key: L[key][s] for key in sets}, s) for s in have]
            out["%s_%s" % (variant, fam)] = res
            m = lambda k, q=None: float(np.mean([(x[k][q] if q else x[k]) for x in res]))
            ES.log("%-5s %-12s %s" % (variant, fam, "  ".join("%s solver %.3f pred %.3f corr %.2f sign %.2f" % (
                k, m(k, "sim_median"), m(k, "pred_median"), m(k, "corr"), m(k, "sign")) for k in res[0] if "/" in k)))
    if not out: raise SystemExit("no predictions under %s: run edited_shells.py on the sets of --sets, variants %s" % (a.in_dir, a.variants))
    os.makedirs(a.out_dir, exist_ok=True); f = a.out_dir + "/neighbour_probe.json"
    json.dump(out, open(f, "w"), indent=1); ES.log("wrote " + f)


if __name__ == "__main__":
    main()
