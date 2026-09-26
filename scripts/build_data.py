"""Regenerate the JSON data consumed by the BLISS project page.

    python scripts/build_data.py --sample <review subset> --tables <paper tables dir> --runs <finished runs dir>

Writes, under static/data/:

- explorer.json     the shells of the review subset seen from the pole at the 8,192 scoring nodes (input w and the
                    magnitude of the buckling deviation), the intervention test P3 of its five sources, and the
                    single- and two-defect shells as table rows.
- leaderboard.json  every row of the full results tables of the paper (B1, B2, B3), with the bootstrap intervals.
- predictions.json  for the core shells that are in both the B1 and the B3 test sets of the subset, the solver field
                    and the seed-0 prediction of every field model trained on B1 and on B3.
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import re
import sys

import numpy as np

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "static" / "data"
FAMS = [("figconv", "FigConvNet"), ("mgn", "MeshGraphNet"), ("sfno", "SFNO"), ("transformer", "Transformer"),
        ("transolver", "Transolver"), ("deeponet", "DeepONet"), ("mlp", "MLP-20")]
COLS = ["kappa_mre", "kappa_r2", "unconservative", "buckling_r2", "field_r2", "defect_r2", "rmse", "rmse_noshrink",
        "rel_l2", "rel_l2_noshrink", "clear_site", "near_tie", "sparse_kappa", "sparse_r2", "dense_kappa", "dense_r2"]


def deviation(U, P):
    """Magnitude of the buckling deviation: U minus each shell's uniform radial contraction."""
    U = U.astype(np.float64)
    return np.linalg.norm(U - (U * P).sum(-1).mean(-1)[..., None, None] * P, axis=-1)


def clean(c):
    """Plain text of a LaTeX table cell."""
    c = c.replace("$-$", "-").replace("\\newline", " ").replace("$", "").replace("\\times", "x")
    c = re.sub(r"\\(textbf|emph)\{([^}]*)\}", r"\2", c)
    return re.sub(r"\s+", " ", c).strip()


def cell(c):
    """(value, low, high) of a cell 'v\\,{\\tiny[lo, hi]}', or the raw text."""
    m = re.match(r"^\s*([-\d.]+)\s*\\,\{\\tiny\[([-\d.]+),\s*([-\d.]+)\]\}", c.replace("$-$", "-"))
    if m:
        return [float(m.group(1)), float(m.group(2)), float(m.group(3))]
    t = clean(c)
    try:
        return [float(t)]
    except ValueError:
        return t


def leaderboard(tables):
    """Rows of full_B1/2/3.tex as {protocol: [{name, group, cells}]}."""
    out = {}
    for prot in ("B1", "B2", "B3"):
        rows, group = [], "learned field models"
        for line in open(os.path.join(tables, "full_%s.tex" % prot)):
            s = line.strip()
            if not s.endswith("\\\\") or "&" not in s or s.startswith(("&", "\\", "%")):
                continue
            parts = [p.strip() for p in s[:-2].split("&")]
            if len(parts) not in (1 + len(COLS), 1 + len(COLS) - 4):
                continue
            name = clean(parts[0])
            if name.startswith(("XGBoost", "flat set", "DeepSets", "pairwise")):
                group = "models on the list of defects"
            elif name.startswith(("training mean", "regime", "nearest", "k nearest", "PCA", "gradient", "weakest",
                                  "deepest", "random")):
                group = "references"
            elif name.startswith("solver"):
                group = "solver"
            elif name.startswith(("MLP",)):
                group = "models on the list of defects"
            rows.append(dict(name=name, group=group, cells=dict(zip(COLS, [cell(p) for p in parts[1:]]))))
        out[prot] = rows
    return out


def predictions(sample, runs, nodes_file):
    """Solver field and B1/B3 predictions of every field model on the shells in both test sets."""
    b1 = set(json.load(open(os.path.join(sample, "splits", "B1.json")))["test"])
    b3 = set(json.load(open(os.path.join(sample, "splits", "B3.json")))["test"])
    both = sorted(b1 & b3)
    out = dict(shells=[], models=[n for _, n in FAMS])
    files = {}
    for fam, _ in FAMS:
        for prot, pat in (("B1", "%s_v2random_*_test_pred.npz"), ("B3", "%s_v2regime_pb3buck_test_pred.npz")):
            fs = [f for f in sorted(glob.glob(os.path.join(runs, "%s_%s_n8192" % (fam, prot), pat % fam)))
                  if not re.search(r"_s[12]_test_pred", f)]
            files[(fam, prot)] = np.load(fs[0])
    z0 = files[(FAMS[0][0], "B1")]; P = z0["P"].astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True)
    for sid in both:
        rec = dict(id=int(sid), models={})
        for (fam, prot), z in files.items():
            j = int(np.where(z["ids"] == sid)[0][0])
            if "truth" not in rec:
                d = deviation(z["true_U"][j][None], P)[0]
                rec["truth"] = dict(d=np.round(d, 3).tolist(), kappa=round(float(z["true_K"][j]), 4), site=int(d.argmax()))
            d = deviation(z["pred_U"][j][None], P)[0]
            name = dict(FAMS)[fam]
            rec["models"].setdefault(name, {})[prot] = dict(d=np.round(d, 3).tolist(),
                                                           kappa=round(float(z["pred_K"][j]), 4), site=int(d.argmax()))
        out["shells"].append(rec)
    e1 = np.array([1.0, 0, 0]); ax = np.array([0, 0, 1.0])
    for v in (np.array([0, 0, 1.]), np.array([0, 1., 0]), np.array([1., 0, 0])):
        if np.degrees(np.arccos(np.clip(P @ v, -1, 1))).max() < 91:
            ax = v
    e1 = e1 - (e1 @ ax) * ax if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0]) - (np.array([0, 1.0, 0]) @ ax) * ax
    e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
    out["x"] = np.round(P @ e1, 4).tolist(); out["y"] = np.round(P @ e2, 4).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", required=True, help="the review subset (release format)")
    ap.add_argument("--tables", required=True, help="directory with full_B1.tex, full_B2.tex, full_B3.tex")
    ap.add_argument("--runs", required=True, help="directory of the finished runs (<fam>_<B>_n8192/*_test_pred.npz)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lb = leaderboard(a.tables)
    json.dump(dict(columns=COLS, protocols=lb), open(OUT / "leaderboard.json", "w"), separators=(",", ":"))
    print("leaderboard:", {k: len(v) for k, v in lb.items()})
    pr = predictions(a.sample, a.runs, None)
    json.dump(pr, open(OUT / "predictions.json", "w"), separators=(",", ":"))
    print("predictions:", [s["id"] for s in pr["shells"]])


if __name__ == "__main__":
    main()
