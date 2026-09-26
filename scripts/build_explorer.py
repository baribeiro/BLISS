"""Build the 3D dataset explorer data of the BLISS project page from the review subset.

    python scripts/build_explorer.py --sample <review subset>

Writes, under static/data/:
- geometry.json        the 8,192 scoring nodes (unit vectors) and a triangulation of the hemisphere through them
- catalog.json         one row per core shell of the subset (defects, deepest defect, kappa, q, regime, split parts),
                       the P3 records, and the single- and two-defect shells
- shells/<id>.json     per shell: w and the displacement at the peak U (mm) at the 8,192 nodes
"""
import argparse
import csv
import json
import os
import pathlib
import sys

import numpy as np
from scipy.spatial import Delaunay

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "static" / "data"


def standard_nodes(n_mesh=76805, n=8192):
    """The fixed 8,192-node subset every metric of the benchmark is computed on."""
    return np.sort(np.random.default_rng(0).choice(n_mesh, n, replace=False))


def frame_of(p):
    """Frame (e1, e2, axis) of the hemisphere, as in the benchmark code."""
    for v in (np.array([0, 0, 1.]), np.array([0, 1., 0]), np.array([1., 0, 0])):
        if np.degrees(np.arccos(np.clip(p @ v, -1, 1))).max() < 91:
            ax = v
    e1 = np.array([1., 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1., 0])
    e1 = e1 - (e1 @ ax) * ax; e1 /= np.linalg.norm(e1)
    return e1, np.cross(ax, e1), ax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True)
    S = ap.parse_args().sample
    (OUT / "shells").mkdir(parents=True, exist_ok=True)
    sh = np.load(os.path.join(S, "shared.npz")); P = sh["P"].astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True)
    R = float(sh["R"]); nodes = standard_nodes(len(P)); e1, e2, ax = frame_of(P)
    B = np.stack([e1, e2, ax]); Q = P[nodes] @ B.T                       # local frame, pole along z
    tri = Delaunay(Q[:, :2]).simplices
    json.dump(dict(R=R, x=np.round(Q[:, 0], 5).tolist(), y=np.round(Q[:, 1], 5).tolist(), z=np.round(Q[:, 2], 5).tolist(),
                   i=tri[:, 0].tolist(), j=tri[:, 1].tolist(), k=tri[:, 2].tolist()),
              open(OUT / "geometry.json", "w"), separators=(",", ":"))
    def write_shell(z, i):
        """Per-shell file: w, the displacement at the peak (U) and after it (U + mode), and the load path."""
        U = z["U_peak"][nodes] @ B.T; M = (z["U_peak"][nodes] + z["mode"][nodes]) @ B.T
        out = dict(w=np.round(z["w"][nodes], 4).tolist(), lpf=np.round(z["lpf"], 5).tolist(), peak=int(z["peak_frame"]))
        for k, A in (("u", U), ("m", M)):
            for c, ax_ in enumerate("xyz"):
                out[k + ax_] = np.round(A[:, c], 4).tolist()
        json.dump(out, open(OUT / "shells" / ("%d.json" % i), "w"), separators=(",", ":"))

    kd = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(S, "metadata", "shells.csv")))}
    dp = {}
    for r in csv.DictReader(open(os.path.join(S, "metadata", "defect_parameters.csv"))):
        dp.setdefault(int(r["id"]), []).append(float(r["delta"]))
    q = {int(r["shell"]): float(r["q"]) for r in csv.DictReader(open(os.path.join(S, "metadata", "truth_dominance.csv")))}
    parts = {}
    for b in ("B1", "B2", "B3"):
        sp = json.load(open(os.path.join(S, "splits", b + ".json")))
        for part in ("train", "val", "test"):
            for i in sp[part]:
                parts.setdefault(int(i), {})[b] = part
    reasons = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(S, "sample_ids.csv")))}
    core = []
    for f in sorted(os.listdir(os.path.join(S, "canonical"))):
        i = int(f[:-4]); z = np.load(os.path.join(S, "canonical", f))
        write_shell(z, i)
        n = len(dp[i])
        core.append(dict(id=i, n=n, dmax=round(max(dp[i]), 3), kappa=round(float(z["kappa"]), 4), q=round(q[i], 3),
                         regime="sparse" if n <= 18 else "dense", parts=parts.get(i, {}), reason=reasons[i]["reason"]))
    p3 = []
    for r in csv.DictReader(open(os.path.join(S, "sweeps", "P3", "sources.csv"))):
        i = int(r["id"]); z = np.load(os.path.join(S, "sweeps", "P3", "%d.npz" % i))
        write_shell(z, i)
        p3.append(dict(id=i, source=int(r["source_id"]), spacing=reasons[i]["reason"].split("spacing ")[1].split(" ")[0],
                       kappa=round(float(z["kappa"]), 4)))
    # fine display grid of the hemisphere (pole along z), with the nearest scoring node of every vertex
    nlat, nlon = 72, 288
    V, T = [[0.0, 0.0, 1.0]], []
    for a in range(1, nlat + 1):
        b = np.radians(90.0 * a / nlat)
        for o in range(nlon):
            t = 2 * np.pi * o / nlon
            V.append([np.sin(b) * np.cos(t), np.sin(b) * np.sin(t), np.cos(b)])
    V = np.array(V)
    for o in range(nlon):
        T.append([0, 1 + o, 1 + (o + 1) % nlon])
    for a in range(1, nlat):
        r0, r1 = 1 + (a - 1) * nlon, 1 + a * nlon
        for o in range(nlon):
            p, q = o, (o + 1) % nlon
            T += [[r0 + p, r1 + p, r1 + q], [r0 + p, r1 + q, r0 + q]]
    T = np.array(T)
    near = np.argmax(V @ Q.T, axis=1)                                     # nearest of the 8,192 scoring nodes
    json.dump(dict(x=np.round(V[:, 0], 5).tolist(), y=np.round(V[:, 1], 5).tolist(), z=np.round(V[:, 2], 5).tolist(),
                   i=T[:, 0].tolist(), j=T[:, 1].tolist(), k=T[:, 2].tolist(), near=near.tolist()),
              open(OUT / "grid.json", "w"), separators=(",", ":"))
    # defect lists of every block: (delta [t], lambda, theta, phi) in the frame of the mesh (pole along z)
    def defects(path):
        d = {}
        for r in csv.DictReader(open(path)):
            d.setdefault(int(r["id"]), []).append([round(float(r[k]), 5) for k in ("delta", "lambda", "theta", "phi")])
        return d
    dcore = defects(os.path.join(S, "metadata", "defect_parameters.csv"))
    dp3 = defects(os.path.join(S, "sweeps", "P3", "defect_parameters.csv"))
    dref = defects(os.path.join(S, "metadata", "reference_defect_parameters.csv"))
    for c in core:
        c["defects"] = dcore[c["id"]]
    for c in p3:
        c["defects"] = dp3[c["id"]]
    single, double = [], []
    for i, ds in sorted(dref.items()):
        row = kd[i]
        rec = dict(id=i, kappa=round(float(row["knock_down"]), 4), R=float(row["radius"]), eta=float(row["eta"]), defects=ds)
        if reasons[i]["block"] == "single":
            rec.update(delta=ds[0][0], lam=ds[0][1]); single.append(rec)
        else:
            u = [np.array([np.sin(p) * np.cos(t), np.sin(p) * np.sin(t), np.cos(p)]) for _, _, t, p in ds]
            rec.update(sep=round(float(np.degrees(np.arccos(np.clip(u[0] @ u[1], -1, 1)))), 1),
                       same=ds[0][0] == ds[1][0] and ds[0][1] == ds[1][1]); double.append(rec)
    json.dump(dict(core=core, p3=p3, single=single, double=double, eta_core=110.0, R_core=R),
              open(OUT / "catalog.json", "w"), separators=(",", ":"))
    print("core", len(core), "p3", len(p3), "triangles", len(tri))


if __name__ == "__main__":
    main()
