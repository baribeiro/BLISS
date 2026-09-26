"""K-A1 jitter inputs: sources rotated ANALYTICALLY about the pole, w rebuilt on the release mesh, no solve.

    python benchmark/analyses/interventions_inputs.py [--smoke]

Populations (K-A1): the 100 P3 sources (= the P1/P2 sources) and the 40 dense P4 sources (cells/p5 notes with
arm b10). For each source and each angle a in {30, 60, 90} deg the defect table (metadata/defects.csv) is rotated
exactly as tools/newcell.py sample_p1 rotates it (theta -> theta + a mod 2 pi) and w is rebuilt with the generator
of tools/newcell.py deck (the one that built every sweep deck): w = sum_i -delta_i t exp(-(beta_i / (lambda_i /
pref))^2), pref = (12 (1 - nu^2))^(1/4) sqrt(R/t), beta_i the angle from the node's direction to the defect centre
in the mesh frame (geometry.frame of the pole axis), on the 76,805 release nodes (BLISS-1.0/shared.npz P).
The continuum truth is invariant under this rotation, so kappa_true(rotated) = kappa_true(source) by construction
and no solve is needed; the truth columns are therefore left empty (NaN) for the rotated rows.

Checks (printed, and stored in the npz): the rebuild with the UNROTATED table vs the canonical w of the source, and
the rebuild of the rotated table vs the simulated P1 arms' w (cells/p1 fields via heavy/sweep_inputs.npz), both
max |dw| / t. The thickness that reproduces them (newcell's T_SHELL 0.23 or t = R/110) is chosen by that check.
Output heavy/rot_inputs.npz in the heavy/sweep_inputs.npz format (sweep = "ROT" rows + the sources' core rows),
plus heavy/rot_inputs_defects.csv (the rotated tables, for the defect-table models), read by
  python benchmark/analyses/interventions_predict.py infer --cache heavy/rot_inputs.npz --out-tag _rot ...

Functions
---------
rebuild
    Rebuild w from a defect list on the release mesh, with the deck generator's formula.
main
    Build the rotated P1 inputs of every source and angle, and their rotated defect tables.
"""
import argparse, csv, math, os, re, sys
import numpy as np
B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"
sys.path.insert(0, B + "/benchmark/analyses")
from bliss.geometry import frame, pole_axis   # noqa: E402
R, NU = 25.4, 0.5
ANGLES = (30, 60, 90)


def rebuild(P, defs, t, E):
    """Rebuild w from a defect list on the release mesh, with the deck generator's formula.

    Parameters
    ----------
    P : numpy.ndarray
        Node positions (mm).
    defs : list
        Defects as (delta, lambda, theta, phi).
    t : float
        Wall thickness (mm).
    E : tuple of numpy.ndarray
        Frame (e1, e2, pole) of the defect angles.

    Returns
    -------
    w : numpy.ndarray
        Radial deviation per node (mm).
    """
    e1, e2, pole = E; eta = R / t; pref = (12.0 * (1.0 - NU * NU)) ** 0.25 * math.sqrt(eta); w = np.zeros(len(P))
    for delta, lam, th, ph in defs:
        c = math.sin(ph) * math.cos(th) * e1 + math.sin(ph) * math.sin(th) * e2 + math.cos(ph) * pole
        beta = np.arccos(np.clip(P @ c, -1.0, 1.0))
        w += -delta * t * np.exp(-(beta / (lam / pref)) ** 2)
    return w.astype(np.float32)


def main():
    """Build the rotated P1 inputs of every source and angle, and their rotated defect tables."""
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    import interventions_predict as ES
    Pf = np.load(B + "/BLISS-1.0/shared.npz")["P"].astype(np.float64); Pf /= np.linalg.norm(Pf, axis=1, keepdims=True)
    name, _ = pole_axis(Pf * R); E = frame(name)
    D = {}
    for r in csv.DictReader(open(B + "/metadata/defects.csv")):
        D.setdefault(int(r["id"]), []).append((float(r["delta"]), float(r["lambda"]), float(r["theta"]), float(r["phi"])))
    src3 = sorted({int(re.match(r"src(\d+)", r["note"]).group(1)) for r in csv.DictReader(open(B + "/cells/p3/runs.csv"))})
    src4 = sorted({int(re.match(r"src(\d+)", r["note"]).group(1)) for r in csv.DictReader(open(B + "/cells/p5/runs.csv")) if "_b10_" in r["note"]})
    pops = {s: [] for s in set(src3) | set(src4)}
    for s in src3: pops[s].append("P3src")
    for s in src4: pops[s].append("P4dense")
    srcs = sorted(pops)
    if a.smoke: srcs = [s for s in srcs if s in (2279, 6677)] or srcs[:2]
    print("sources: %d P3, %d dense P4, %d distinct; building %d" % (len(src3), len(src4), len(pops), len(srcs)))
    # thickness convention: the one that reproduces the canonical w of the sources
    chk = {}
    for t in (0.23, R / 110.0):
        e = [np.abs(rebuild(Pf, D[s], t, E) - np.load("%s/BLISS-1.0/canonical/%d.npz" % (B, s))["w"]).max() / t for s in srcs[:10]]
        chk[t] = float(max(e))
    T = min(chk, key=chk.get); print("rebuild of the unrotated sources vs canonical w, max |dw|/t:", chk, "-> t =", T)
    assert chk[T] < 1e-3, "the generator does not reproduce the canonical w"
    # rotated rebuild vs the simulated P1 arms (same sources, same angles)
    Z = np.load(H + "/sweep_inputs.npz"); p1 = {(int(s), a_): i for i, (s, a_, sw) in enumerate(zip(Z["source"], Z["arm"], Z["sweep"])) if sw == "P1"}
    rows, W, defrows, p1err = [], [], [], []
    NODES = Z["NODES8192"]; rid = 900000; EANG = ES._frame()[2]          # geometry.frame_of, as interventions_predict reports angles
    for s in srcs:
        c = np.load("%s/BLISS-1.0/canonical/%d.npz" % (B, s))
        U = c["U_peak"]; j = ES._site(U, Pf); b, th = ES._angles(Pf[j], EANG)
        rows.append(dict(id=s, sweep="core", cell="", source=s, arm="source", note="+".join(pops[s]), input_from="canonical", status="ok",
                         kappa_true=float(c["kappa"]), kappa_true_field=float(c["kappa"]), first_lp_inc=int(c["peak_frame"]), true_node=j,
                         true_beta_deg=b, true_theta_deg=th, true_node_sub=int(NODES[ES._site(U[NODES], Pf[NODES])]), U_max_mm=float(np.abs(U).max())))
        W.append(c["w"].astype(np.float32))
        for ang in ANGLES:
            v = [(d, l, (th_ + math.radians(ang)) % (2 * math.pi), ph) for d, l, th_, ph in D[s]]
            w = rebuild(Pf, v, T, E); rid += 1
            if (s, "rot%d" % ang) in p1: p1err.append(float(np.abs(w - Z["W"][p1[(s, "rot%d" % ang)]]).max() / T))
            rows.append(dict(id=rid, sweep="ROT", cell="", source=s, arm="rot%d" % ang, note="+".join(pops[s]), input_from="analytic_rotation",
                             status="no truth (continuum-invariant rotation)", kappa_true=np.nan, kappa_true_field=np.nan, first_lp_inc=-1,
                             true_node=-1, true_beta_deg=np.nan, true_theta_deg=np.nan, true_node_sub=-1, U_max_mm=np.nan))
            W.append(w); defrows += [[rid, "%.12g" % d, l, "%.12g" % th_, "%.12g" % ph] for d, l, th_, ph in v]
    print("rotated rebuild vs simulated P1 arms: n %d, max |dw|/t %.2e" % (len(p1err), max(p1err) if p1err else np.nan))
    out = H + "/rot_inputs%s.npz" % ("_smoke" if a.smoke else ""); dcsv = out[:-4] + "_defects.csv"
    with open(dcsv, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["id", "delta", "lambda", "theta", "phi"]); wr.writerows(defrows)
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez(out, W=np.stack(W), U8192=np.zeros((len(rows), len(NODES), 3), np.float32), NODES8192=NODES, defects_csv=dcsv,
             thickness=T, check_unrotated_vs_canonical=chk[T], check_rotated_vs_p1_max=max(p1err) if p1err else np.nan, **cols)
    print("wrote %s (%d rows: %d rotated + %d core) and %s" % (out, len(rows), sum(r["sweep"] == "ROT" for r in rows), len(srcs), dcsv))


if __name__ == "__main__":
    main()
