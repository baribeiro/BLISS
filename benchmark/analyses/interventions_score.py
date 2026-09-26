"""Intervention tests (App K): the pre-registered statistics of the benchmark plan, Part K (r2, frozen sha256
52832869...) as amended by K-A1 (r3.3, frozen sha256 a4a11776...). Implements exactly those definitions; nothing
here is tuned on the results.

    python benchmark/analyses/interventions_score.py [--in-dir outputs] [--n-boot 2000]
    python benchmark/analyses/interventions_score.py --synthetic       # plumbing test on generated csvs; never reads the real predictions

INPUTS (benchmark/analyses/interventions_predict.py outputs; one csv per family and seed):
  B1 models   sweep_eval_<fam>_n8192_s<s>.csv          P1, P2, P3 records + source core rows (P4 rows are ignored:
                                                        65/80 P4 sources are in B1 train, K rule)
              sweep_eval_<fam>_n8192_rot_s<s>.csv      analytic rotations (interventions_inputs.py) -> jitter J_m
  B3 models   sweep_eval_<fam>_b3_p3p4dense_s<s>.csv   P3 + dense P4 (b10) records + source core rows
              sweep_eval_<fam>_b3_rot_s<s>.csv         analytic rotations of P3 and dense-P4 sources
  truth fields (P1 field equivariance and its interpolation calibration): sweep_inputs.npz (U8192) and
              sweep_eval_<fam>_n8192_s<s>_fields.npz (pred_U).

DEFINITIONS (K-A1 replaces r2's detection rule; everything else is r2):
  unit = source; source bootstrap, 2,000 resamples, seed 0; a model's prediction per record = the mean over its
  seeds (reported), each seed also scored separately; no ranking. kappa changes are absolute (kappa units) and
  also printed in percent of the reference arm's kappa.
  J_m (per model and population S): median over sources in S and rotations {30,60,90} of |kappa_pred(rot) -
  kappa_pred(source)|, analytic rotations. S = P3 sources for P2/P3 (B1) and P3 (B3); the 40 dense P4 sources
  for P4 (B3).
  P1 (B1): (i) per-record |dkappa_pred| vs the solver's |dkappa_true| (medians); (ii) field equivariance =
       buckling R^2 of the rotated shell's prediction rotated back (inverse-distance k=4 on the 8,192 nodes, vectors
       rotated back) against the source's prediction; the same operator applied to the solver's own fields gives the
       interpolation calibration row; (iii) site follows rotation (predicted site within 4 deg of the source's
       predicted site rotated by the angle), split by whether the solver's site followed (same rule on the truth).
       No "detected".
  P2 (B1; the 87 analysed sources: reference arm db+0.0 present and >= 7 of 9 arms): per record dk = kappa(arm) -
       kappa(db+0.0 arm), truth and prediction. Pooled least-squares slope (with intercept) of dk_pred on dk_true over
       all non-reference records; source-bootstrap 95 % interval; DETECTED iff the interval excludes 0 AND the
       per-seed pooled slopes share its sign in all seeds. Distance of the slope from 1 printed. Per-source Spearman
       rho and slope: descriptive (median, share positive). Site not scored.
  P3 (B1 and B3; reference arm sep 25.0) and P4 (B3 only; 40 dense sources; reference arm f100 = all far dimples
       kept): per source and arm, change = dk_pred(arm) - dk_pred(ref) with dk_pred = kappa_pred(arm) -
       kappa_pred(ref) (i.e. the paired change against the reference arm). DETECTED iff the source-bootstrap 95 %
       interval of the median signed change lies entirely outside [-J_m, +J_m] AND the per-seed medians share its
       sign in all seeds. MAGNITUDE recovered iff the median ratio change_pred/change_true over sources with
       |change_true| above the solver floor (0.09 % of the reference kappa) lies in [0.5, 2]. P3 headline arm 10 deg;
       share of sources recovering >= half the true drop. P3 site: share of sources whose predicted site at 10 deg is
       > 2 deg from every defect centre (truth reference 96 %). P4 site "moves": predicted site > 4 deg from the
       predicted site on the f100 arm; the same on the truth (solver reference 92 of 98 dropping arms).
  Also printed (r2 Part K): the identical-prediction check of the P2/P4 reference arm against the source's core record;
       the P3/P4 sign agreement rate (share of sources above the solver floor whose predicted change has the true sign);
       P4 site moves on all removal arms and on the solver's dropping arms (true drop > 1 %), model and solver under
       the same 4-deg rule (ADC-02); the P4 headline arm b10_f050 (K-A3).
  MDE (per sweep, model): J_m + the half-width of the bootstrap interval of the median change, i.e. the smallest
       median change that could have been declared detected with this many sources.
Output: <in-dir>/score_sweeps.json and score_sweeps_table.csv.

Classes
-------
Geo
    Mesh and frame of the release, with rotations about the hemisphere axis.

Functions
---------
log
    Print a message with the time.
read_csv
    Read a sweep prediction CSV with typed columns.
boot_median
    Median with a bootstrap 95 % interval.
slope
    Least-squares slope of y on x.
spearman
    Spearman rank correlation (NaN when undefined).
rotate_back
    Field predicted on the rotated shell, read at the rotated positions of the source nodes, vectors
    rotated back.
load_family
    Sweep prediction CSVs of one family, per seed.
jitter
    J per seed and for the seed-mean prediction; pop = set of sources.
paired
    {(source, arm): dict(true, pred_s{seed}, pred_mean, kref_true)} of changes against the reference
    arm.
identical_check
    Reference arm vs the source's core record (the same shell): max |kappa_pred difference| over sources
    and seeds.
detect_shift
    Detection and magnitude criteria of one arm: paired changes against the model's P1 jitter.
main
    Score the intervention tests P1-P4 from the sweep predictions and write outputs/score_sweeps.json.
field_equivariance
    P1 (ii): buckling R^2 of the rotated-back prediction vs the source prediction; calibration on the
    truth.
make_synthetic
    Fake interventions_predict outputs (random numbers) with the real column layout, sources and arms,
    for a plumbing test.
"""
import argparse, csv, glob, json, math, os, re, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"
sys.path.insert(0, B)
FLOOR = 0.0009               # solver floor, fraction of kappa (0.05-0.09 %)
SITE_MOVE, SITE_ROT, SITE_DIMPLE = 4.0, 4.0, 2.0


def log(*a): 
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_csv(f):
    """Read a sweep prediction CSV with typed columns.

    Parameters
    ----------
    f : str
        Path.

    Returns
    -------
    rows : list of dict
        Rows.
    """
    rows = list(csv.DictReader(open(f)))
    for r in rows:
        for k in ("id", "source", "pred_node", "true_node"): r[k] = int(r[k])
        for k in ("kappa_true", "kappa_pred"): r[k] = float(r[k])
    return rows


def boot_median(v, n_boot, rng_seed=0):
    """Median with a bootstrap 95 % interval.

    Parameters
    ----------
    v : array_like
        Values.
    n_boot : int
        Resamples.
    rng_seed : int, default=0
        Seed.

    Returns
    -------
    median, lo, hi : float
        Median and interval.
    """
    v = np.asarray(v, float); rng = np.random.default_rng(rng_seed)
    d = np.median(v[rng.integers(0, len(v), (n_boot, len(v)))], 1)
    return float(np.median(v)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def slope(x, y):
    """Least-squares slope of y on x.

    Parameters
    ----------
    x, y : array_like
        Data.

    Returns
    -------
    slope : float
        Slope.
    """
    x, y = np.asarray(x, float), np.asarray(y, float); xm = x.mean()
    return float(((x - xm) * (y - y.mean())).sum() / max(((x - xm) ** 2).sum(), 1e-30))


def spearman(x, y):
    """Spearman rank correlation (NaN when undefined).

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
    r = spearmanr(x, y).correlation
    return float(r) if np.isfinite(r) else np.nan


class Geo:
    """Mesh and frame of the release, with rotations about the hemisphere axis."""
    def __init__(s):
        """Load the unit node directions and the frame of the mesh."""
        from bliss.geometry import frame, pole_axis
        Pf = np.load(B + "/BLISS-1.0/shared.npz")["P"].astype(np.float64); s.Pf = Pf / np.linalg.norm(Pf, axis=1, keepdims=True)
        s.E = np.stack(frame(pole_axis(s.Pf * 25.4)[0]))
    def rot(s, V, deg):           # rotate vectors (..., 3) by deg about the pole (mesh frame)
        """Rotate vectors about the hemisphere axis.

        Parameters
        ----------
        V : numpy.ndarray
            Vectors, shape (..., 3).
        deg : float
            Angle in degrees.

        Returns
        -------
        R : numpy.ndarray
            Rotated vectors.
        """
        a = math.radians(deg); c, sn = math.cos(a), math.sin(a)
        L = V @ s.E.T; L = np.stack([c * L[..., 0] - sn * L[..., 1], sn * L[..., 0] + c * L[..., 1], L[..., 2]], -1)
        return L @ s.E
    def ang(s, i, j): 
        """Angle between two nodes (deg).

        Parameters
        ----------
        i, j : int
            Node indices.

        Returns
        -------
        angle : float
            Angle.
        """
        return float(np.degrees(np.arccos(np.clip(s.Pf[i] @ s.Pf[j], -1, 1))))
    def ang_v(s, v, j): 
        """Angle between a vector and a node (deg).

        Parameters
        ----------
        v : numpy.ndarray
            Unit vector.
        j : int
            Node index.

        Returns
        -------
        angle : float
            Angle.
        """
        return float(np.degrees(np.arccos(np.clip(v @ s.Pf[j], -1, 1))))


def rotate_back(Urot, deg, P, tree):
    """Field predicted on the rotated shell, read at the rotated positions of the source nodes, vectors rotated back.

    Parameters
    ----------
    Urot : numpy.ndarray
        Field predicted on the rotated shell.
    deg : float
        Rotation angle.
    P : numpy.ndarray
        Unit node directions.
    tree : scipy.spatial.cKDTree
        Tree of P.

    Returns
    -------
    U : numpy.ndarray
        Field rotated back onto the source frame.
    """
    d, i = tree.query(GEO.rot(P, deg), k=4); w = 1 / np.maximum(d, 1e-12); w /= w.sum(1, keepdims=True)
    return GEO.rot((Urot[i] * w[..., None]).sum(1), -deg)


def load_family(in_dir, fam, variant, tag):
    """Sweep prediction CSVs of one family, per seed.

    Parameters
    ----------
    in_dir : str
        Directory.
    fam : str
        Family.
    variant : str
        Variant.
    tag : str
        Output tag.

    Returns
    -------
    seeds : dict
        Seed -> rows.
    """
    fs = sorted(glob.glob("%s/sweep_eval_%s_%s%s_s[0-9].csv" % (in_dir, fam, variant, tag)))
    return {int(re.search(r"_s(\d)\.csv$", f).group(1)): read_csv(f) for f in fs}


def jitter(rot_seeds, pop):
    """J per seed and for the seed-mean prediction; pop = set of sources.

    Parameters
    ----------
    rot_seeds : dict
        P1 rows per seed.
    pop : set
        Source shells.

    Returns
    -------
    J : dict
        Jitter per seed and for the seed mean.
    """
    out = {}
    per = {}
    for s, rows in rot_seeds.items():
        src = {r["source"]: r["kappa_pred"] for r in rows if r["sweep"] == "core"}
        per[s] = {(r["source"], r["arm"]): abs(r["kappa_pred"] - src[r["source"]]) for r in rows if r["sweep"] == "ROT" and r["source"] in pop and r["source"] in src}
        out["s%d" % s] = float(np.median(list(per[s].values()))) if per[s] else np.nan
    seeds = sorted(rot_seeds)
    mean = {}
    for s in seeds:
        rows = rot_seeds[s]; src = {r["source"]: r["kappa_pred"] for r in rows if r["sweep"] == "core"}
        for r in rows:
            if r["sweep"] == "ROT" and r["source"] in pop: mean.setdefault((r["source"], r["arm"]), []).append(r["kappa_pred"] - src[r["source"]])
    out["seed_mean"] = float(np.median([abs(np.mean(v)) for v in mean.values()])) if mean else np.nan
    out["n"] = len(mean)
    return out


def paired(seeds, sweep, ref_arm, arm_ok=lambda a: True):
    """{(source, arm): dict(true, pred_s{seed}, pred_mean, kref_true)} of changes against the reference arm.

    Parameters
    ----------
    seeds : dict
        Rows per seed.
    sweep : str
        Sweep name.
    ref_arm : str
        Reference arm.
    arm_ok : callable
        Filter of the arms.

    Returns
    -------
    changes : dict
        (source, arm) -> true and predicted changes.
    """
    out = {}
    for s, rows in seeds.items():
        ref = {r["source"]: r for r in rows if r["sweep"] == sweep and r["arm"] == ref_arm}
        for r in rows:
            if r["sweep"] != sweep or r["arm"] == ref_arm or not arm_ok(r["arm"]) or r["source"] not in ref: continue
            e = out.setdefault((r["source"], r["arm"]), dict(true=r["kappa_true"] - ref[r["source"]]["kappa_true"],
                                                               kref=ref[r["source"]]["kappa_true"], preds={}))
            e["preds"][s] = r["kappa_pred"] - ref[r["source"]]["kappa_pred"]
    for e in out.values(): e["pred"] = float(np.mean(list(e["preds"].values())))
    return out


def identical_check(seeds, sweep, ref_arm):
    """Reference arm vs the source's core record (the same shell): max |kappa_pred difference| over sources and seeds.

    Parameters
    ----------
    seeds : dict
        Rows per seed.
    sweep : str
        Sweep name.
    ref_arm : str
        Reference arm.

    Returns
    -------
    maxdiff : float
        Largest |kappa difference|.
    """
    d = []
    for s, rows in seeds.items():
        core = {r["source"]: r["kappa_pred"] for r in rows if r["sweep"] == "core"}
        d += [abs(r["kappa_pred"] - core[r["source"]]) for r in rows if r["sweep"] == sweep and r["arm"] == ref_arm and r["source"] in core]
    return dict(n=len(d), max_abs_dkappa_pred=float(max(d)) if d else None)


def detect_shift(ch, arm, J, n_boot, seeds):
    """Detection and magnitude criteria of one arm: paired changes against the model's P1 jitter.

    Parameters
    ----------
    ch : dict
        Paired changes.
    arm : str
        Arm.
    J : dict
        Jitter.
    n_boot : int
        Resamples.
    seeds : list
        Seeds.

    Returns
    -------
    res : dict or None
        Medians, intervals, detection and magnitude flags.
    """
    sub = {k: v for k, v in ch.items() if k[1] == arm}
    if not sub: return None
    pm = [v["pred"] for v in sub.values()]; tr = [v["true"] for v in sub.values()]; kr = np.array([v["kref"] for v in sub.values()])
    med, lo, hi = boot_median(pm, n_boot)
    per_seed = {s: float(np.median([v["preds"][s] for v in sub.values() if s in v["preds"]])) for s in seeds}
    sign_ok = all(np.sign(x) == np.sign(med) and x != 0 for x in per_seed.values())
    detected = bool((lo > J or hi < -J) and sign_ok)
    big = np.abs(tr) > FLOOR * kr; ratio = np.array(pm)[big] / np.array(tr)[big]
    rmed = float(np.median(ratio)) if big.any() else np.nan
    half = [v["pred"] / v["true"] >= 0.5 for v in sub.values() if abs(v["true"]) > FLOOR * v["kref"]]
    sgn = [np.sign(v["pred"]) == np.sign(v["true"]) for v in sub.values() if abs(v["true"]) > FLOOR * v["kref"]]
    return dict(n_sources=len(sub), median_change_pred=med, ci=[lo, hi], median_change_true=float(np.median(tr)),
                sign_agreement_rate=float(np.mean(sgn)) if sgn else np.nan, n_sign_agreement=len(sgn),
                median_change_pred_pct=float(np.median(np.array(pm) / kr * 100)), median_change_true_pct=float(np.median(np.array(tr) / kr * 100)),
                per_seed_median=per_seed, sign_agrees_all_seeds=sign_ok, J=J, detected=detected,
                n_above_floor=int(big.sum()), median_ratio=rmed, magnitude_recovered=bool(0.5 <= rmed <= 2) if np.isfinite(rmed) else False,
                share_recovering_half=float(np.mean(half)) if half else np.nan, MDE=float(J + 0.5 * (hi - lo)))


def main():
    """Score the intervention tests P1-P4 from the sweep predictions and write outputs/score_sweeps.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=H); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true"); a = ap.parse_args()
    global GEO; GEO = Geo()
    if a.synthetic: a.in_dir = make_synthetic(); log("synthetic inputs in", a.in_dir)
    from scipy.spatial import cKDTree
    Z = np.load("%s/sweep_inputs.npz" % a.in_dir) if os.path.exists("%s/sweep_inputs.npz" % a.in_dir) else None
    fams = sorted({re.match(r"sweep_eval_(.+?)_(n8192|b3)", os.path.basename(f)).group(1) for f in glob.glob(a.in_dir + "/sweep_eval_*_s[0-9].csv")})
    Ddef = {}
    for f in (B + "/cells/p3/defects.csv",):
        for r in csv.DictReader(open(f)):
            ph, th = float(r["phi"]), float(r["theta"])
            Ddef.setdefault(int(r["id"]), []).append(np.array([math.sin(ph) * math.cos(th), math.sin(ph) * math.sin(th), math.cos(ph)]) @ GEO.E)
    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), in_dir=a.in_dir, synthetic=a.synthetic, n_boot=a.n_boot,
               frozen=dict(part_K_r2="52832869033e787f4d142c117a960ad8cbc9cb8bac89db5a5415088b8c952da2",
                           K_A1="a4a11776da4144134030d1cb38831464e756fc498be575e6e8dbfc5b5618615a"), models={})
    table = []
    for fam in fams:
        for variant, tag in (("n8192", ""), ("b3", "_p3p4dense")):
            seeds = load_family(a.in_dir, fam, variant, tag); rot = load_family(a.in_dir, fam, variant, "_rot")
            if not seeds: continue
            if not rot: log("SKIP %s %s: no analytic-rotation predictions (J_m undefined)" % (fam, variant)); continue
            S = sorted(seeds); any_rows = seeds[S[0]]
            p3src = {r["source"] for r in any_rows if r["sweep"] == "P3"}; p4src = {r["source"] for r in any_rows if r["sweep"] == "P4"}
            J3 = jitter(rot, p3src); J4 = jitter(rot, p4src) if p4src else None
            m = dict(seeds=S, J_P3_sources=J3, J_P4_dense_sources=J4); key = "%s|%s" % (fam, "B1" if variant == "n8192" else "B3")
            if variant == "n8192":
                # ---- P1 ----
                p1 = {}
                for s, rows in seeds.items():
                    core = {r["source"]: r for r in rows if r["sweep"] == "core"}
                    for r in rows:
                        if r["sweep"] != "P1" or r["source"] not in core: continue
                        c = core[r["source"]]; ang = float(r["arm"].replace("rot", ""))
                        e = p1.setdefault(r["id"], dict(source=r["source"], angle=ang, dtrue=abs(r["kappa_true"] - c["kappa_true"]), dpred={}, follow={}))
                        e["dpred"][s] = abs(r["kappa_pred"] - c["kappa_pred"])
                        e["follow"][s] = GEO.ang_v(GEO.rot(GEO.Pf[c["pred_node"]], ang), r["pred_node"]) <= SITE_ROT
                        e["true_follow"] = GEO.ang_v(GEO.rot(GEO.Pf[c["true_node"]], ang), r["true_node"]) <= SITE_ROT
                if p1:
                    dp = [np.mean(list(e["dpred"].values())) for e in p1.values()]; dt = [e["dtrue"] for e in p1.values()]
                    tf = np.array([e["true_follow"] for e in p1.values()]); pf = np.array([np.mean(list(e["follow"].values())) for e in p1.values()])
                    m["P1"] = dict(n=len(p1), median_abs_dkappa_pred=float(np.median(dp)), median_abs_dkappa_true=float(np.median(dt)),
                                   site_follows_rate=float(pf.mean()), site_follows_when_solver_followed=float(pf[tf].mean()) if tf.any() else None,
                                   site_follows_when_solver_did_not=float(pf[~tf].mean()) if (~tf).any() else None,
                                   n_solver_followed=int(tf.sum()))
                    m["P1"]["field_equivariance"] = field_equivariance(a.in_dir, fam, seeds, Z, cKDTree)
                # ---- P2 ----
                m["P2_reference_identical_check"] = identical_check(seeds, "P2", "db+0.0")
                ch2 = paired(seeds, "P2", "db+0.0")
                arms_per = {}
                for (src, arm) in ch2: arms_per.setdefault(src, set()).add(arm)
                ok_src = {s_ for s_, v in arms_per.items() if len(v) + 1 >= 7}
                ch2 = {k: v for k, v in ch2.items() if k[0] in ok_src}
                if ch2:
                    X = np.array([v["true"] for v in ch2.values()]); Y = np.array([v["pred"] for v in ch2.values()]); srcs = np.array([k[0] for k in ch2])
                    us = np.unique(srcs); idx = {u: np.where(srcs == u)[0] for u in us}; rng = np.random.default_rng(0); bs = []
                    for _ in range(a.n_boot):
                        pick = np.concatenate([idx[u] for u in rng.choice(us, len(us))]); bs.append(slope(X[pick], Y[pick]))
                    sl = slope(X, Y); lo, hi = np.quantile(bs, [0.025, 0.975])
                    per_seed = {s: slope(X, np.array([v["preds"].get(s, np.nan) for v in ch2.values()])) for s in S}
                    sign_ok = all(np.sign(x) == np.sign(sl) for x in per_seed.values())
                    rho = [spearman(X[idx[u]], Y[idx[u]]) for u in us]; sls = [slope(X[idx[u]], Y[idx[u]]) for u in us]
                    m["P2"] = dict(n_sources=int(len(us)), n_records=int(len(X)), pooled_slope=sl, ci=[float(lo), float(hi)], per_seed_slope=per_seed,
                                   sign_agrees_all_seeds=sign_ok, detected=bool((lo > 0 or hi < 0) and sign_ok), distance_from_1=float(sl - 1),
                                   per_source_median_spearman=float(np.nanmedian(rho)), per_source_median_slope=float(np.median(sls)),
                                   share_positive_slope=float(np.mean(np.array(sls) > 0)))
            # ---- P3 (B1 and B3) ----
            ch3 = paired(seeds, "P3", "sep25.0")
            if ch3:
                m["P3"] = {arm: detect_shift(ch3, arm, J3["seed_mean"], a.n_boot, S) for arm in sorted({k[1] for k in ch3})}
                off = []
                for s, rows in seeds.items():
                    for r in rows:
                        if r["sweep"] == "P3" and r["arm"] == "sep10.0" and r["id"] in Ddef:
                            off.append((r["source"], s, min(GEO.ang_v(c, r["pred_node"]) for c in Ddef[r["id"]]) > SITE_DIMPLE,
                                        min(GEO.ang_v(c, r["true_node"]) for c in Ddef[r["id"]]) > SITE_DIMPLE))
                if off:
                    m["P3_site_10deg"] = dict(share_pred_off_dimples=float(np.mean([o[2] for o in off])), share_true_off_dimples=float(np.mean([o[3] for o in off])),
                                              per_seed={s: float(np.mean([o[2] for o in off if o[1] == s])) for s in S})
            # ---- P4 (B3 only, dense) ----
            if variant == "b3":
                m["P4_reference_identical_check"] = identical_check(seeds, "P4", "b10_f100")
                ch4 = paired(seeds, "P4", "b10_f100", lambda arm: arm.startswith("b10"))
                if ch4 and J4:
                    m["P4_headline_arm"] = "b10_f050"          # K-A3: 50 % of the far dimples kept
                    m["P4"] = {arm: detect_shift(ch4, arm, J4["seed_mean"], a.n_boot, S) for arm in sorted({k[1] for k in ch4})}
                    mv = {}
                    for s, rows in seeds.items():
                        ref = {r["source"]: r for r in rows if r["sweep"] == "P4" and r["arm"] == "b10_f100"}
                        for r in rows:
                            if r["sweep"] == "P4" and r["arm"] != "b10_f100" and r["source"] in ref:
                                rf = ref[r["source"]]; drop = (r["kappa_true"] - rf["kappa_true"]) / rf["kappa_true"] < -0.01
                                e = mv.setdefault(r["arm"], dict(pred=[], true=[], drop=[]))
                                e["pred"].append(GEO.ang(r["pred_node"], rf["pred_node"]) > SITE_MOVE); e["drop"].append(drop)
                                e["true"].append(GEO.ang(r["true_node"], rf["true_node"]) > SITE_MOVE)
                    def rates(vs):
                        """Share of shells whose failure site changes, predicted and true.

                        Parameters
                        ----------
                        vs : list of dict
                            Per-seed site changes.

                        Returns
                        -------
                        rates : dict
                            Predicted and true rates.
                        """
                        P_ = np.concatenate([v["pred"] for v in vs]); T_ = np.concatenate([v["true"] for v in vs]); D_ = np.concatenate([v["drop"] for v in vs])
                        return dict(pred_rate_all=float(P_.mean()), true_rate_all=float(T_.mean()), n_all=int(len(P_) // len(S)),
                                    pred_rate_dropping=float(P_[D_].mean()) if D_.any() else None,
                                    true_rate_dropping=float(T_[D_].mean()) if D_.any() else None, n_dropping=int(D_.sum() // len(S)))
                    m["P4_site_moves"] = {arm: rates([v]) for arm, v in mv.items()}
                    m["P4_site_moves"]["all_removal_arms"] = rates(list(mv.values()))
            res["models"][key] = m
            for sw in ("P3", "P4"):
                for arm, d in (m.get(sw) or {}).items():
                    if d: table.append(dict(model=key, sweep=sw, arm=arm, n=d["n_sources"], J=d["J"], median_change_pred=d["median_change_pred"],
                                            ci_lo=d["ci"][0], ci_hi=d["ci"][1], median_change_true=d["median_change_true"], detected=d["detected"],
                                            median_ratio=d["median_ratio"], magnitude_recovered=d["magnitude_recovered"], MDE=d["MDE"]))
            if "P2" in m: table.append(dict(model=key, sweep="P2", arm="pooled", n=m["P2"]["n_sources"], J="", median_change_pred=m["P2"]["pooled_slope"],
                                           ci_lo=m["P2"]["ci"][0], ci_hi=m["P2"]["ci"][1], median_change_true=1.0, detected=m["P2"]["detected"],
                                           median_ratio="", magnitude_recovered="", MDE=""))
            log("%-18s scored: %s" % (key, ", ".join(k for k in m if k.startswith("P"))))
    json.dump(res, open(a.in_dir + "/score_sweeps.json", "w"), indent=1, default=float)
    if table:
        with open(a.in_dir + "/score_sweeps_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    log("wrote %s/score_sweeps.json (%d model rows)" % (a.in_dir, len(res["models"])))


def field_equivariance(in_dir, fam, seeds, Z, cKDTree):
    """P1 (ii): buckling R^2 of the rotated-back prediction vs the source prediction; calibration on the truth.

    Parameters
    ----------
    in_dir : str
        Directory.
    fam : str
        Family.
    seeds : list
        Seeds.
    Z : mapping
        Rotated inputs.
    cKDTree : type
        scipy.spatial.cKDTree.

    Returns
    -------
    res : dict or None
        Buckling R^2 per seed.
    """
    from bliss.metrics import score
    if Z is None: return None
    NODES = Z["NODES8192"]; P = GEO.Pf[NODES]; tree = cKDTree(P); zpos = {int(i): j for j, i in enumerate(Z["id"])}
    out = {}
    for s in sorted(seeds):
        f = "%s/sweep_eval_%s_n8192_s%d_fields.npz" % (in_dir, fam, s)
        if not os.path.exists(f): continue
        F = np.load(f); fpos = {int(i): j for j, i in enumerate(F["ids"])}; pu = F["pred_U"].astype(np.float64)
        rows = seeds[s]; core = {r["source"]: r["id"] for r in rows if r["sweep"] == "core"}
        src_p, back_p, src_t, back_t = [], [], [], []
        for r in rows:
            if r["sweep"] != "P1" or r["source"] not in core: continue
            deg = float(r["arm"].replace("rot", "")); i, c = fpos[r["id"]], fpos[core[r["source"]]]
            src_p.append(pu[c]); back_p.append(rotate_back(pu[i], deg, P, tree))
            src_t.append(Z["U8192"][zpos[core[r["source"]]]].astype(np.float64)); back_t.append(rotate_back(Z["U8192"][zpos[r["id"]]].astype(np.float64), deg, P, tree))
        if not src_p: continue
        k = np.ones(len(src_p))
        sp_ = score(np.array(src_p), np.array(back_p), k, k, P)["buckling_r2"]; st_ = score(np.array(src_t), np.array(back_t), k, k, P)["buckling_r2"]
        out["s%d" % s] = dict(buckling_r2_pred=float(sp_), buckling_r2_truth_calibration=float(st_), n=len(src_p))
    return out


def make_synthetic():
    """Fake interventions_predict outputs (random numbers) with the real column layout, sources and arms, for a plumbing test.

    Returns
    -------
    d : str
        Directory with the synthetic CSVs.
    """
    import tempfile
    d = tempfile.mkdtemp(prefix="score_sweeps_synth_"); rng = np.random.default_rng(1)
    runs = {c: list(csv.DictReader(open("%s/cells/%s/runs.csv" % (B, c)))) for c in ("p1", "p2", "p3", "p5")}
    N = len(GEO.Pf); cols = ["id", "sweep", "source", "arm", "note", "input_from", "kappa_true", "kappa_pred", "pred_node", "true_node"]
    def arm_of(note):
        """Arm name of a run from its note.

        Parameters
        ----------
        note : str
            Note in runs.csv.

        Returns
        -------
        arm : str
            Arm.
        """
        t = note.split("_"); return "_".join(x for x in t[1:] if not re.match(r"^(w\d+|l\d+|nfix\d+|lv[0-9.]+)$", x))
    base = []
    for sw, c in (("P1", "p1"), ("P2", "p2"), ("P3", "p3"), ("P4", "p5")):
        for r in runs[c]:
            base.append(dict(id=int(r["shell"]), sweep=sw, source=int(r["note"][3:].split("_")[0]), arm=arm_of(r["note"]), note=r["note"], input_from="field"))
    srcs = sorted({b["source"] for b in base}); base += [dict(id=s, sweep="core", source=s, arm="source", note="", input_from="canonical") for s in srcs]
    ktrue = {b["id"]: 0.45 + 0.02 * rng.standard_normal() for b in base}; tnode = {b["id"]: int(rng.integers(N)) for b in base}
    for variant, tag, keep in (("n8192", "", lambda b: True), ("b3", "_p3p4dense", lambda b: b["sweep"] in ("P3", "core") or (b["sweep"] == "P4" and b["arm"].startswith("b10")))):
        for s in (0, 1, 2):
            with open("%s/sweep_eval_synth_%s%s_s%d.csv" % (d, variant, tag, s), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
                for b in base:
                    if keep(b): w.writerow(dict(b, kappa_true=ktrue[b["id"]], kappa_pred=ktrue[b["id"]] + 0.005 * rng.standard_normal(),
                                                pred_node=int(rng.integers(N)), true_node=tnode[b["id"]]))
            with open("%s/sweep_eval_synth_%s_rot_s%d.csv" % (d, variant, s), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); rid = 900000
                for src in srcs:
                    w.writerow(dict(id=src, sweep="core", source=src, arm="source", note="", input_from="canonical", kappa_true=ktrue[src],
                                    kappa_pred=ktrue[src], pred_node=0, true_node=0))
                    for ang in (30, 60, 90):
                        rid += 1; w.writerow(dict(id=rid, sweep="ROT", source=src, arm="rot%d" % ang, note="", input_from="analytic_rotation",
                                                  kappa_true="nan", kappa_pred=ktrue[src] + 0.002 * rng.standard_normal(), pred_node=0, true_node=-1))
    return d


if __name__ == "__main__":
    main()
