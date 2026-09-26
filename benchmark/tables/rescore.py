"""Rescoring of every stored v2 prediction with the fixed scorer, plus the secondary metrics and diagnostics.

    python benchmark/tables/rescore.py [--n-boot 1000] [--families ...] [--smoke]

Inputs (read only): the B1/B2/B3 finals of the eight field families, three seeds each
(runs/final_runs/<fam>_<B>_n8192/*_test_pred.npz), the w-based references (heavy/wref_v2/, from
shape_baselines.py) and the mean-field row (built here from the training split).

  D-M3  bliss.metrics.score_ci(split="B1|B2|B3") on each artefact (released q, no twin dropping), and
        score_by_regime (pooled / sparse / dense / q bins); mean and range over the three seeds.
        compare(): paired shell bootstrap of the headline differences for every pair of families on each
        protocol, seed-0 artefacts (a pair is also compared seed by seed on buckling_r2, the field headline).
  D-N2  J.2 secondary metrics per artefact: RMSE (mm; t = 0.23 mm) and mean and median per-shell relative L2 error
        ||U_pred-U||/||U||, on the full field and after removing each shell's uniform contraction (D), all shells;
        kappa tail: p90 and mean of |relative error|, median signed error, unconservative rate (kappa_hat > 1.01 kappa).
  D-N3  J.3 diagnostics: rim residual (RMSE of D on the nodes within 10 deg of the rim, polar angle > 80 deg,
        over the RMSE on all nodes); u_bar-kappa consistency (the truth's linear u_bar = a kappa + b fitted on the
        test truth; median |u_bar_pred - (a kappa_pred + b)| over the median |u_bar_true|).
  D-N6  re-rank of the two B1 finalists (c0, c1) on B1-validation subsets admissible for B2 and B3 (B1 val shells
        that are in that protocol's train+val), mean over seeds of val buckling_dimple_r2, when both finalists'
        validation predictions are stored; the files used are listed.
Writes heavy/rescore_v2.json (and rescore_v2_smoke.json with --smoke: 2 families, 60 shells, 50 resamples).

Functions
---------
log
    Print a message with the time.
artefacts
    Test prediction files of one family and protocol, per seed.
secondary
    Secondary metrics of one prediction file: RMSE, relative L2, rim residual and kappa consistency.
main
    Rescore every stored prediction with the scorer and write outputs/rescore.json.
"""
import argparse, glob, itertools, json, os, re, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"; RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, B)
from bliss.metrics import score_ci, score_by_regime, compare, score, released_q   # noqa: E402
FAMS = ["deeponet", "figconv", "mgn", "sfno", "transformer", "transolver", "mlp", "mlp80"]
PROT = ("B1", "B2", "B3")
VAL_DIRS = ["%s/final_runs/{fam}_B1_n8192" % RUNS, "%s/remote3_final/heavy" % RUNS, "%s/remote2_final/heavy" % RUNS,
            "%s/remote1_final/heavy" % RUNS, "%s/remote4_final/heavy" % RUNS, H]


def log(*a): 
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def artefacts(fam, prot):
    """Test prediction files of one family and protocol, per seed.

    Parameters
    ----------
    fam : str
        Model family.
    prot : str
        Protocol.

    Returns
    -------
    files : dict
        Seed -> path.
    """
    d = "%s/final_runs/%s_%s_n8192" % (RUNS, fam, prot); out = {}
    for f in sorted(glob.glob(d + "/*_test_pred.npz")):
        m = re.search(r"_s(\d+)_test_pred\.npz$", f); out[int(m.group(1)) if m else 0] = f
    return out


def secondary(z, P):
    """Secondary metrics of one prediction file: RMSE, relative L2, rim residual and kappa consistency.

    Parameters
    ----------
    z : mapping
        Prediction arrays.
    P : numpy.ndarray
        Unit node directions.

    Returns
    -------
    out : dict
        The secondary metrics.
    """
    Ut, Up = np.asarray(z["true_U"], np.float64), np.asarray(z["pred_U"], np.float64)
    dev = lambda U: U - (U * P).sum(-1).mean(1)[:, None, None] * P
    Dt, Dp = dev(Ut), dev(Up); out = {}
    for nm, a, b in (("full", Ut, Up), ("deviation", Dt, Dp)):
        e = np.linalg.norm((b - a).reshape(len(a), -1), axis=1); n = np.linalg.norm(a.reshape(len(a), -1), axis=1)
        out["rmse_mm_" + nm] = float(np.sqrt(((b - a) ** 2).sum(-1).mean()))
        out["rel_l2_" + nm] = float(np.mean(e / np.maximum(n, 1e-30)))
        out["rel_l2_median_" + nm] = float(np.median(e / np.maximum(n, 1e-30)))
    # rim residual: the rim is where the polar angle from the hemisphere axis exceeds 80 deg
    from bliss.geometry import frame_of
    ax = frame_of(P)[2]; rim = np.degrees(np.arccos(np.clip(P @ ax, -1, 1))) > 80
    r_all = np.sqrt(((Dp - Dt) ** 2).sum(-1).mean()); r_rim = np.sqrt(((Dp - Dt)[:, rim] ** 2).sum(-1).mean())
    out["rim_rmse_over_all"] = float(r_rim / r_all); out["rim_nodes"] = int(rim.sum())
    ub_t = (Ut * P).sum(-1).mean(1); ub_p = (Up * P).sum(-1).mean(1); kt, kp = np.asarray(z["true_K"]), np.asarray(z["pred_K"])
    a_, b_ = np.polyfit(kt, ub_t, 1)
    out["ubar_kappa_truth_fit"] = [float(a_), float(b_)]
    rk = (kp - kt) / kt                                     # kappa tail (MLC-07b): an over-predicted knockdown is the unsafe error
    out["kappa_rel_err_p90_pct"] = float(np.quantile(np.abs(rk), 0.9) * 100); out["kappa_rel_err_mean_pct"] = float(np.mean(np.abs(rk)) * 100)
    out["kappa_signed_err_median_pct"] = float(np.median(rk) * 100); out["kappa_unconservative_rate"] = float(np.mean(kp > 1.01 * kt))
    out["ubar_kappa_inconsistency"] = float(np.median(np.abs(ub_p - (a_ * kp + b_))) / np.median(np.abs(ub_t)))
    return out


def main():
    """Rescore every stored prediction with the scorer and write outputs/rescore.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--families", default=",".join(FAMS))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); t0 = time.time(); fams = a.families.split(",")
    if a.smoke: fams, a.n_boot = fams[:2], 50
    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), n_boot=a.n_boot, scorer="bliss.metrics (D-M3 version)", rows={}, compare={}, rerank={})
    cz = np.load(H + "/canon_nodes_8192.npz"); pos = {int(i): j for j, i in enumerate(cz["ids"])}
    P = cz["P"].astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True)

    def maybe_smoke(f):
        """Reduce a prediction file to 60 shells in --smoke mode.

        Parameters
        ----------
        f : str or mapping
            Prediction file.

        Returns
        -------
        z : str or dict
            The file, or its reduced arrays.
        """
        if not a.smoke: return f
        z = dict(np.load(f)) if isinstance(f, str) else dict(f); n = 60; n0 = len(z["true_K"])
        for k in ("ids", "true_U", "pred_U", "true_K", "pred_K", "r", "ang"):
            if k in z and np.ndim(z[k]) and len(z[k]) == n0: z[k] = z[k][:n]
        return z

    def one(name, prot, f):
        """Headline, regime and secondary metrics of one prediction file.

        Parameters
        ----------
        name : str
            Row name.
        prot : str
            Protocol.
        f : str
            Prediction file.

        Returns
        -------
        row : dict
            The metrics.
        """
        z = maybe_smoke(f); zz = np.load(f) if isinstance(f, str) and not a.smoke else z
        sc = score_ci(z, n_boot=a.n_boot, split=prot)
        rg = score_by_regime(z, n_boot=max(50, a.n_boot // 4), split=prot)
        return dict(file=f if isinstance(f, str) else "smoke", n=sc["n_shells"],
                    metrics={k: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for k, v in sc["metrics"].items()},
                    by_regime={k: dict(n=v["n_shells"], metrics={m: v["metrics"][m]["point"] for m in v["metrics"]}) for k, v in rg.items()},
                    secondary=secondary(zz, P))

    # learned families, three seeds
    for fam in fams:
        for prot in PROT:
            arts = artefacts(fam, prot); rows = {}
            for s, f in sorted(arts.items()):
                rows[s] = one(fam, prot, f)
                log("%-11s %s s%d  buckling_r2 %.4f  kappa err %.3f %%  two-way %s" % (fam, prot, s, rows[s]["metrics"]["buckling_r2"]["point"],
                    rows[s]["metrics"]["kappa_rel_err"]["point"], rows[s]["metrics"].get("two_way_winner", {}).get("point")))
            if not rows: log("MISSING %s %s" % (fam, prot)); continue
            agg = {}
            for m in rows[min(rows)]["metrics"]:
                v = [r["metrics"][m]["point"] for r in rows.values() if m in r["metrics"]]
                agg[m] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_seeds=len(v))
            for m in rows[min(rows)]["secondary"]:
                v = [r["secondary"][m] for r in rows.values() if not isinstance(r["secondary"][m], list)]
                if v: agg["secondary:" + m] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), n_seeds=len(v))
            res["rows"]["%s|%s" % (fam, prot)] = dict(seeds=rows, over_seeds=agg)

    # references: mean field and the w-based rows
    for prot in PROT:
        sp = json.load(open("%s/BLISS-1.0/splits/%s.json" % (B, prot)))
        tr = np.array([pos[i] for i in sp["train"]]); te = np.array([pos[i] for i in sp["test"]])
        U = cz["U"]; K = cz["K"]
        mf = dict(ids=cz["ids"][te], true_U=U[te], pred_U=np.broadcast_to(U[tr].mean(0), U[te].shape), true_K=K[te],
                  pred_K=np.full(len(te), K[tr].mean()), P=cz["P"], r=cz["R"][te])
        res["rows"]["meanfield|%s" % prot] = dict(seeds={0: one("meanfield", prot, mf)})
        for f in sorted(glob.glob("%s/wref_v2/*_%s_test_pred.npz" % (H, prot))):
            nm = os.path.basename(f)[:-len("_%s_test_pred.npz" % prot)]
            res["rows"]["%s|%s" % (nm, prot)] = dict(seeds={0: one(nm, prot, f)})
            log("%-15s %s buckling_r2 %.4f kappa err %.3f %%" % (nm, prot, res["rows"]["%s|%s" % (nm, prot)]["seeds"][0]["metrics"]["buckling_r2"]["point"],
                res["rows"]["%s|%s" % (nm, prot)]["seeds"][0]["metrics"]["kappa_rel_err"]["point"]))

    # paired comparisons, seed 0, every pair of families
    for prot in PROT:
        s0 = {f: artefacts(f, prot).get(0) for f in fams}; s0 = {f: v for f, v in s0.items() if v}
        for fa, fb in itertools.combinations(sorted(s0), 2):
            c = compare(maybe_smoke(s0[fa]), maybe_smoke(s0[fb]), n_boot=a.n_boot)
            res["compare"]["%s-%s|%s" % (fa, fb, prot)] = c["metrics"]
        log("compare %s: %d pairs" % (prot, len(list(itertools.combinations(s0, 2)))))

    # D-N6: re-rank of the two B1 finalists on admissible validation subsets
    SP = {p: json.load(open("%s/BLISS-1.0/splits/%s.json" % (B, p))) for p in PROT}
    for fam in fams:
        found = {}
        for c in (0, 1):
            for s in (0, 1, 2):
                nm = "%s_v2random_fbuck8192_c%d_hp2%s_val_pred.npz" % (fam, c, "_s%d" % s if s else "")
                cands = [d.format(fam=fam) + "/" + nm for d in VAL_DIRS if os.path.exists(d.format(fam=fam) + "/" + nm)]
                if cands: found[(c, s)] = cands
        if len(found) < 6: res["rerank"][fam] = dict(status="not stored: %d of 6 finalist val files found" % len(found)); continue
        out = dict(files={"c%d_s%d" % k: v for k, v in found.items()}, subsets={})
        for sub_nm, prot in (("B1_all", None), ("B2_admissible", "B2"), ("B3_admissible", "B3")):
            vals = {}
            for c in (0, 1):
                v = []
                for s in (0, 1, 2):
                    z = np.load(found[(c, s)][0]); ids = z["ids"].astype(int)
                    m = np.ones(len(ids), bool) if prot is None else np.isin(ids, SP[prot]["train"] + SP[prot]["val"])
                    sc = score(z["true_U"][m], z["pred_U"][m], z["true_K"][m], z["pred_K"][m], P, None, released_q(ids[m]))
                    v.append(sc["buckling_dimple_r2"])
                vals["c%d" % c] = dict(mean=float(np.mean(v)), per_seed=[float(x) for x in v], n=int(m.sum()))
            vals["winner"] = max(("c0", "c1"), key=lambda k: vals[k]["mean"]); out["subsets"][sub_nm] = vals
        out["reported_winner"] = "c%d" % json.load(open("%s/final_runs/%s_B1_n8192/phase2_%s_v2random_n8192_buckling_hp2.json" % (RUNS, fam, fam)))["winner_rank"]
        res["rerank"][fam] = out; log("rerank %s: %s" % (fam, {k: v["winner"] for k, v in out["subsets"].items()}))
    f = "rescore_v2_smoke.json" if a.smoke else H + "/rescore_v2.json"
    json.dump(res, open(f, "w"), indent=1, default=float); log("wrote %s [%.0fs]" % (f, time.time() - t0))


if __name__ == "__main__":
    main()
