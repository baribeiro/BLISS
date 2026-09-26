"""AB2-03: the solver's neighbouring Riks increments as predictions of the limit-point field, on the v2 test lists.

    python benchmark/references/solver_increments.py [--nproc 8] [--n-boot 1000] [--smoke]

One set of solver rows for Sec 3 / F.2 / Table 11 / L (the two older sets are retired): per protocol B1, B2, B3
(BLISS-1.0/splits test lists, all shells, no twin rule), the canonical records' U_prev ("k-1") and U_next ("k+1")
as the predicted field of U_peak, on the standard 8,192-node subset, scored by bliss.metrics.score_ci
(split="B1|B2|B3", released q). kappa = the truth (the kappa columns are not defined for these rows).
Writes heavy/increment_reference_v2.json (the v1 increment_reference.json is left as it is).

Functions
---------
main
    Score the solver's neighbouring increments (U_prev, U_next) as predictions of U_peak.
"""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"; REL = B + "/BLISS-1.0"
sys.path.insert(0, B)
from bliss.metrics import score_ci   # noqa: E402
NODES = np.load(H + "/canon_nodes_8192.npz")["NODES"]


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
    return z["U_peak"][NODES], z["U_prev"][NODES], z["U_next"][NODES], float(z["kappa"]), float(z["r_d2_d1"])


def main():
    """Score the solver's neighbouring increments (U_prev, U_next) as predictions of U_peak."""
    ap = argparse.ArgumentParser(); ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true"); a = ap.parse_args(); t0 = time.time()
    from multiprocessing import Pool
    P = np.load(REL + "/shared.npz")["P"][NODES]; out = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), n_boot=a.n_boot, rows={})
    for prot in ("B1", "B2", "B3"):
        te = json.load(open("%s/splits/%s.json" % (REL, prot)))["test"]
        if a.smoke: te = te[:40]
        with Pool(a.nproc) as p: rs = p.map(_read, te, chunksize=16)
        Ut = np.stack([r[0] for r in rs]); k = np.array([r[3] for r in rs]); rr = np.array([r[4] for r in rs])
        for tag, j in (("k-1", 1), ("k+1", 2)):
            pred = np.stack([r[j] for r in rs])
            r = score_ci(dict(ids=np.array(te), P=P, true_U=Ut, pred_U=pred, true_K=k, pred_K=k, r=rr), n_boot=a.n_boot, split=prot)
            out["rows"]["%s|%s" % (prot, tag)] = dict(n=r["n_shells"], n_dominant=r["n_dominant"], n_ambiguous=r["n_ambiguous"], n_two_way=r["n_two_way"],
                                                     metrics={m: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for m, v in r["metrics"].items() if not m.startswith("kappa")})
            print("%s solver %s n=%d: buckling_r2 %.3f  bdr2 %.3f  loc<10 dominant %.3f  two-way %.3f  [%.0fs]" % (
                prot, tag, r["n_shells"], r["metrics"]["buckling_r2"]["point"], r["metrics"]["buckling_dimple_r2"]["point"],
                r["metrics"]["loc_dominant_within_10"]["point"], r["metrics"].get("two_way_winner", {}).get("point", np.nan), time.time() - t0), flush=True)
    f = "increment_reference_v2_smoke.json" if a.smoke else H + "/increment_reference_v2.json"
    json.dump(out, open(f, "w"), indent=1, default=float); print("wrote", f)


if __name__ == "__main__":
    main()
