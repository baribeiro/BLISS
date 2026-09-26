"""Reference rows of the benchmark, recomputed on the v2 splits (BLISS-1.0/splits B1, B2, B3; test sets).

    python benchmark/references/site_baselines.py [--nproc 8] [--n-boot 2000] [--n-draws 256] [--smoke 40]

Data: the released canonical records (BLISS-1.0/canonical/<id>.npz: w, U_peak, kappa, kappa_catalogue,
r_d2_d1), read here, on the standard 8,192-node subset (NODES = sorted rng(0).choice(76805, 8192), the subset
prepare_data.py fixed and every field model reads; P = BLISS-1.0/shared.npz P[NODES]). Dominance q from the
released metadata/truth_dominance.csv (full mesh), as score_ci(q="auto") uses it. The v2 splits have no twin
exclusions (excluded_test_counts = 0 for all three), so every test shell is scored.

  (a) meanfield      pred_U = per-node mean of U over the split's TRAINING shells, pred_K = training mean kappa;
                     bliss.metrics.score_ci (score() point values, 95 % bootstrap over shells).
  (b) contraction    share of the variance of U carried by each shell's uniform radial contraction u_r P
                     (u_r = mean over nodes of U . P, the term score() removes):
                     share = 1 - sum|U - u_r P|^2 / sum|U - mean(U)|^2, mean(U) the SCALAR mean of the set
                     (the field_r2 reference). Also with the per-node mean field as reference (share_vs_nodemean).
  (c) loc nulls      angular distance from a null site to the true site (argmax |D_true| on the 8,192 nodes,
                     D = U - u_r P, exactly score()'s loc): uniform = area-uniform point on the hemisphere cap
                     (mesh frame of geometry.frame_of); random_defect = centre of a defect of the shell drawn
                     uniformly from its defect table; n_draws per shell, seed 0; median and fraction < 10 deg
                     over all (shell, draw) pairs. meanfield = score()'s loc for (a).
  (d) deepest_w      zero-parameter rule: site = argmin_nodes w (the deepest inward point of the surface).
  (e) weakest_link   kappa_WL = kappa_1(delta_max), the lambda = 1 column of the single-defect curve
                     (metadata/single_defect_kd.csv, built from the single-defect shells), linear interpolation; shells
                     with delta_max beyond the curve are counted and left out. Against the released kappa
                     (first limit point) and, for continuity with the old number, kappa_catalogue; by regime
                     (b25: ids < 4758, b10: ids >= 4758, the tools/weakest_link.py rule).
Writes outputs/reference_baselines_v2.json (--smoke writes ./reference_baselines_v2_smoke.json on a few test shells).

Functions
---------
log
    Print a message with the time.
load
    Read a set of shells in parallel.
defects
    Defect list of every shell from metadata/defects.csv.
centres
    Unit vectors of the defect centres.
ang
    Angle between unit vectors (deg).
stats
    Median site error and share within 10 deg, with bootstrap intervals, on all shells and on the clear
    and near-tie subsets.
main
    Score the site references (deepest point, random site, random defect, weakest link) on B1, B2 and
    B3.
"""
import argparse, csv, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); REL = B + "/BLISS-1.0"; H = B + "/outputs"
sys.path.insert(0, B)
from bliss.metrics import score_ci, bootstrap_ci, released_q, excluded_test_counts   # noqa: E402
from bliss.geometry import frame_of                                                          # noqa: E402
SPLITS = {"B1": "random", "B2": "delta", "B3": "regime"}
KD1 = os.environ.get("BLISS_KD1", os.path.join(B, "metadata", "single_defect_kd.csv"))   # kappa of the single-defect shells by depth and width
NODES = np.sort(np.random.default_rng(0).choice(76805, 8192, replace=False))


def log(*a):
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _read(i):
    """Read the arrays of one core shell at the 8,192 nodes.

    Parameters
    ----------
    i : int
        Shell identifier.

    Returns
    -------
    arrays : tuple
        The arrays and scalars used by this script.
    """
    z = np.load("%s/canonical/%d.npz" % (REL, i))
    return (z["w"][NODES].astype(np.float32), z["U_peak"][NODES].astype(np.float32), float(z["kappa"]),
            float(z["r_d2_d1"]))


def load(ids, nproc):
    """Read a set of shells in parallel.

    Parameters
    ----------
    ids : sequence of int
        Shell identifiers.
    nproc : int
        Worker processes.

    Returns
    -------
    data : dict
        Stacked arrays W, U, K and R.
    """
    from multiprocessing import Pool
    with Pool(nproc) as p:
        res = p.map(_read, [int(i) for i in ids], chunksize=16)
    return dict(ids=np.asarray(ids, int), W=np.stack([r[0] for r in res]), U=np.stack([r[1] for r in res]),
                K=np.array([r[2] for r in res]), R=np.array([r[3] for r in res]))


def defects():
    """Defect list of every shell from metadata/defects.csv.

    Returns
    -------
    D : dict
        Shell id -> list of (delta, theta, phi).
    """
    D = {}
    for r in csv.DictReader(open(B + "/metadata/defects.csv")):
        D.setdefault(int(r["id"]), []).append((float(r["delta"]), float(r["theta"]), float(r["phi"])))
    return D


def centres(defs, E, conv):
    """Unit vectors of the defect centres.

    Parameters
    ----------
    defs : list
        Defects as (delta, theta, phi).
    E : numpy.ndarray
        Frame of the mesh (e1, e2, axis).
    conv : str
        Angle convention of the defect table.

    Returns
    -------
    c : numpy.ndarray
        Centres, shape (n_defects, 3).
    """
    th = np.array([d[1] for d in defs]); ph = np.array([d[2] for d in defs])
    v = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], -1)
    return v @ E if conv == "mesh_frame" else v                       # E rows = (e1, e2, axis)


def ang(a, b):
    """Angle between unit vectors (deg).

    Parameters
    ----------
    a, b : numpy.ndarray
        Unit vectors.

    Returns
    -------
    angle : numpy.ndarray
        Angles in degrees.
    """
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def stats(x, n_boot, dom=None, amb=None):
    """Median site error and share within 10 deg, with bootstrap intervals, on all shells and on the clear and near-tie subsets.

    Parameters
    ----------
    x : numpy.ndarray
        Site error per shell (deg).
    n_boot : int
        Bootstrap resamples.
    dom : numpy.ndarray, default=None
        Mask of clear winners.
    amb : numpy.ndarray, default=None
        Mask of near-ties.

    Returns
    -------
    stats : dict
        Statistics per subset.
    """
    x = np.asarray(x, float)
    def one(v):
        """Statistics of one subset.

        Parameters
        ----------
        v : numpy.ndarray
            Site errors (deg).

        Returns
        -------
        s : dict or None
            n, median with interval, share within 10 deg with interval.
        """
        if v.size == 0: return None
        md = bootstrap_ci(v, n_boot=n_boot, statistic="median"); w10 = bootstrap_ci((v < 10).astype(float), n_boot=n_boot, statistic="mean")
        return dict(n=int(v.shape[0]), median_deg=md[0], median_ci=[md[1], md[2]], within_10=w10[0], within_10_ci=[w10[1], w10[2]])
    out = dict(all=one(x))
    if dom is not None: out["dominant"] = one(x[dom])
    if amb is not None: out["ambiguous"] = one(x[amb])
    return out


def main():
    """Score the site references (deepest point, random site, random defect, weakest link) on B1, B2 and B3."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-draws", type=int, default=256); ap.add_argument("--smoke", type=int, default=0)
    a = ap.parse_args(); t0 = time.time()
    sh = np.load(REL + "/shared.npz"); Pf = sh["P"].astype(np.float64); P = Pf[NODES]
    E = np.stack(frame_of(Pf)); ax = E[2]
    c8 = np.load(H + "/canon_nodes_8192.npz")["NODES"]; assert np.array_equal(c8, NODES), "node subset differs from prepare_data's"
    sp = {k: json.load(open("%s/splits/%s.json" % (REL, k))) for k in SPLITS}
    all_ids = sorted(set().union(*[set(s["train"]) | set(s["val"]) | set(s["test"]) for s in sp.values()]))
    if a.smoke:                                         # a few test shells + a training sample for the means
        rng = np.random.default_rng(0)
        keep = set(sp["B1"]["test"][:a.smoke]) | set(sp["B2"]["test"][:a.smoke]) | set(sp["B3"]["test"][:a.smoke])
        keep |= set(rng.choice(sp["B1"]["train"], 4 * a.smoke, replace=False).tolist()) | set(rng.choice(sp["B3"]["train"], 4 * a.smoke, replace=False).tolist())
        all_ids = sorted(keep)
    log("reading %d canonical records (%d procs)" % (len(all_ids), a.nproc))
    C = load(all_ids, a.nproc); pos = {int(i): j for j, i in enumerate(C["ids"])}
    log("read in %.0fs" % (time.time() - t0))
    # the arrays the field models were trained on are these arrays
    cz = np.load(H + "/canon_nodes_8192.npz"); cpos = {int(i): j for j, i in enumerate(cz["ids"])}
    chk = [pos[i] for i in list(pos)[:50]]
    assert np.array_equal(cz["W"][[cpos[int(C["ids"][j])] for j in chk]], C["W"][chk]), "canonical w != canon_nodes_8192 W"
    D = defects()
    # defect-centre convention: the one under which the deepest defect's centre sits on argmin w
    conv_err = {}
    for conv in ("global", "mesh_frame"):
        e = []
        for j in range(min(200, len(C["ids"]))):
            d = D[int(C["ids"][j])]; c = centres([max(d)], E, conv)[0]
            e.append(ang(c, P[C["W"][j].argmin()]))
        conv_err[conv] = float(np.median(e))
    CONV = min(conv_err, key=conv_err.get)
    log("defect-centre convention: %s (median angle deepest centre -> argmin w: %s)" % (CONV, conv_err))
    T = [r for r in csv.reader(open(KD1, encoding="utf-8-sig"))]
    dl = np.array([float(r[0]) for r in T[1:]]); col = [j for j, h in enumerate(T[0]) if "=1)" in h.replace(" ", "")][0]
    k1 = np.array([float(r[col]) for r in T[1:]]); o = np.argsort(dl); dl, k1 = dl[o], k1[o]
    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), smoke=a.smoke, data=REL + "/canonical (w, U_peak at NODES)",
               nodes="8192 (prepare_data.py subset, rng(0))", q_source="metadata/truth_dominance.csv (released, full mesh)",
               n_boot=a.n_boot, n_draws=a.n_draws, defect_centre_convention=CONV, convention_check_deg=conv_err,
               kd1_curve="%s column %r" % (KD1, T[0][col]), splits={})
    rng = np.random.default_rng(0)
    for S, name in SPLITS.items():
        tr = np.array([pos[i] for i in sp[S]["train"] if i in pos]); te = np.array([pos[i] for i in sp[S]["test"] if i in pos])
        ids = C["ids"][te]; U, K, W = C["U"][te], C["K"][te], C["W"][te]
        q = released_q(ids); dom, amb = (q < 0.7), (q > 0.9)
        out = dict(n_train_used=int(len(tr)), n_test=int(len(te)), twin_exclusions=excluded_test_counts("v2/" + name),
                   n_dominant=int(dom.sum()), n_ambiguous=int(amb.sum()))
        # (a) mean field
        Um = C["U"][tr].mean(0); Km = C["K"][tr].mean()
        pred = dict(ids=ids, true_U=U, pred_U=np.broadcast_to(Um, U.shape), true_K=K, pred_K=np.full_like(K, Km), P=P.astype(np.float32), r=C["R"][te])
        sc = score_ci(pred, n_boot=a.n_boot, q=q, r=C["R"][te], drop_leaked=False)
        out["meanfield"] = dict(metrics={k: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for k, v in sc["metrics"].items()},
                                score={k: v for k, v in sc["score"].items() if not isinstance(v, dict)})
        # (b) contraction share
        U64 = U.astype(np.float64); ur = (U64 * P).sum(-1).mean(1); res_c = ((U64 - ur[:, None, None] * P) ** 2).sum()
        out["contraction"] = dict(share=float(1 - res_c / ((U64 - U64.mean()) ** 2).sum()),
                                  share_vs_nodemean=float(1 - res_c / ((U64 - U64.mean(0)) ** 2).sum()),
                                  definition="1 - sum|U - u_r P|^2 / sum|U - mean|^2 over the test shells, u_r = mean_nodes(U.P)")
        # true site, exactly score()'s: argmax |D| on the 8,192 nodes
        Dt = U64 - ur[:, None, None] * P; it = np.linalg.norm(Dt, axis=-1).argmax(1); Pt = P[it]
        # (c) nulls
        cb = rng.uniform(0, 1, (len(te), a.n_draws)); sb = np.sqrt(1 - cb ** 2); ph = rng.uniform(-np.pi, np.pi, (len(te), a.n_draws))
        V = cb[..., None] * ax + sb[..., None] * (np.cos(ph)[..., None] * E[0] + np.sin(ph)[..., None] * E[1])
        uni = ang(V, Pt[:, None, :])
        rd = np.stack([ang(centres(D[int(i)], E, CONV)[rng.integers(0, len(D[int(i)]), a.n_draws)], Pt[j]) for j, i in enumerate(ids)])
        rep = lambda m: np.repeat(m, a.n_draws)
        out["loc_nulls"] = dict(uniform_cap=stats(uni.ravel(), a.n_boot, rep(dom), rep(amb)),
                                random_defect_centre=stats(rd.ravel(), a.n_boot, rep(dom), rep(amb)),
                                meanfield=dict(all_median_deg=sc["score"]["loc_deg_median"], all_within_10=sc["score"]["loc_within_10"],
                                               dominant_median_deg=sc["score"]["loc_dominant_median"], dominant_within_10=sc["score"]["loc_dominant_within_10"]),
                                note="bootstrap CIs of the two drawn nulls resample (shell, draw) pairs, so they understate the shell-level spread")
        # (d) deepest point of w
        dw = ang(P[W.argmin(1)], Pt)
        out["deepest_w"] = stats(dw, a.n_boot, dom, amb)
        rb = C["R"][te]; out["deepest_w"]["by_r"] = {"%.2f-%.2f" % (lo, hi): dict(n=int(((rb >= lo) & (rb < hi)).sum()),
                                                      median_deg=float(np.median(dw[(rb >= lo) & (rb < hi)])) if ((rb >= lo) & (rb < hi)).any() else None,
                                                      within_10=float((dw[(rb >= lo) & (rb < hi)] < 10).mean()) if ((rb >= lo) & (rb < hi)).any() else None)
                                                      for lo, hi in ((0, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01))}
        # (e) weakest link
        dm = np.array([max(x[0] for x in D[int(i)]) for i in ids]); ok = dm <= dl.max()
        kwl = np.interp(dm, dl, k1); wl = {}
        for reg, m in (("all", ok), ("b25", ok & (ids < 4758)), ("b10", ok & (ids >= 4758))):
            if not m.any(): continue
            e = np.abs(kwl[m] - K[m]) / K[m] * 100
            ci = bootstrap_ci(e, n_boot=a.n_boot, statistic="median")
            wl[reg] = dict(n=int(m.sum()), median_rel_err_pct=ci[0], ci=[ci[1], ci[2]],
                           median_signed_err_pct=float(np.median((kwl[m] - K[m]) / K[m] * 100)),
                           kappa_r2=float(1 - ((kwl[m] - K[m]) ** 2).sum() / ((K[m] - K[m].mean()) ** 2).sum()))
        wl["n_beyond_curve"] = int((~ok).sum()); out["weakest_link"] = wl
        res["splits"][S] = out
        log("%s (%d test): meanfield field_r2 %.3f buckling_r2 %.3f bdr2 %.3f | contraction %.1f %% | loc null uniform %.1f, defect %.1f, meanfield %.1f | deepest-w %.1f deg (%.0f %% <10; dominant %.0f %%) | WL %.2f %%"
            % (S, len(te), sc["score"]["field_r2"], sc["score"]["buckling_r2"], sc["score"]["buckling_dimple_r2"], 100 * out["contraction"]["share"],
               out["loc_nulls"]["uniform_cap"]["all"]["median_deg"], out["loc_nulls"]["random_defect_centre"]["all"]["median_deg"],
               sc["score"]["loc_deg_median"], out["deepest_w"]["all"]["median_deg"], 100 * out["deepest_w"]["all"]["within_10"],
               100 * (out["deepest_w"]["dominant"] or {}).get("within_10", np.nan), wl.get("all", {}).get("median_rel_err_pct", np.nan)))
    f = "reference_baselines_v2_smoke.json" if a.smoke else H + "/reference_baselines_v2.json"
    json.dump(res, open(f, "w"), indent=1, default=float); log("wrote %s [%.0fs]" % (f, time.time() - t0))


if __name__ == "__main__":
    main()
