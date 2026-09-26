"""Build the parametric master table: one row per shell, from every campaign, in one place.

The set models, the weakest link and the tabular models read this same table, so that a difference between
them is the model and not the table.

    python benchmark/data/prepare_parametric.py [--out outputs/parametric_master.parquet]

ONE ROW PER SHELL, with
  id, origin, held_out, R, t, R_over_t, nu, p_classical, N, kappa
  per-defect lists: delta (in units of t), lam, beta0_deg, x, y, z   (x, y, z = unit vector)

The position is a Cartesian unit vector rather than (theta, phi), which are discontinuous at the pole and at
the 2*pi cut.

ORIGINS. `core_b25` and `core_b10` are the sparse and dense families of the 4,399 core shells; `single` and
`double` are the single- and two-defect shells.

held_out marks the intervention tests P1-P4, which never enter training in any model.

Functions
---------
beta0
    Angular half-width of a defect (deg).
"""
import argparse
import csv
import math
import os
import sys

import numpy as np
import pandas as pd

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=B + "/outputs/parametric_master.parquet")
A = ap.parse_args()

M = B + "/metadata"


def _lista(nome):
    """The family files are lists of ids without a header.

    Parameters
    ----------
    nome : str
        File name without extension.

    Returns
    -------
    ids : set of int
        Shell identifiers.
    """
    return {int(l.split(",")[0]) for l in open("%s/%s.csv" % (M, nome)) if l.strip()}


FAM = [("core_b10", _lista("multi_defects_phi_min_10deg")),
       ("core_b25", _lista("multi_defects_phi_min_25deg")),
       ("single", _lista("single_defect")),
       ("double", _lista("two_defects"))]

defeitos = {}
for r in csv.DictReader(open(M + "/defects.csv")):
    defeitos.setdefault(int(r["id"]), []).append(
        (float(r["delta"]), float(r["lambda"]), float(r["theta"]), float(r["phi"])))

casca = {}
for r in csv.DictReader(open(M + "/shells.csv")):
    if not r["knock_down"]:
        continue
    casca[int(r["id"])] = dict(kappa=float(r["knock_down"]), R=float(r["radius"]),
                               nu=float(r["nu"]), eta=float(r["eta"]))

# espessura e pressao classica medidas, quando existem; senao derivadas (t = R/eta, Zoelly)
med = {}
if os.path.exists(M + "/reproduction.csv"):
    for r in csv.DictReader(open(M + "/reproduction.csv")):
        i = int(r["shell"])
        med[i] = (float(r["thickness"]) if r.get("thickness") else None,
                  float(r["p_classical"]) if r.get("p_classical") else None)

canon = {int(r["sample_id"]) for r in csv.DictReader(open(B + "/BLISS-1.0/metadata/sample_ids.csv"))}


def beta0(lam, nu, eta):
    """Angular half-width of a defect (deg).

    Parameters
    ----------
    lam : float
        Width parameter lambda.
    nu : float
        Poisson's ratio.
    eta : float
        Slenderness R/t.

    Returns
    -------
    beta0 : float
        Half-width in degrees.
    """
    return lam * (12 * (1 - nu ** 2)) ** -0.25 * eta ** -0.5 * 180.0 / math.pi


linhas = []
sem_familia = []
for i, c in sorted(casca.items()):
    if i not in defeitos:
        continue
    orig = next((n for n, s in FAM if i in s), None)
    if orig is None:
        sem_familia.append(i)
        orig = "core_b10" if len(defeitos[i]) >= 40 else "core_b25"
    t, pc = med.get(i, (None, None))
    if t is None:
        t = c["R"] / c["eta"]
    if pc is None:
        # O E FALTAVA. A formula de Zoelly e p_c = 2E/sqrt(3(1-nu^2)) * (t/R)^2, e sem o modulo
        # para geometria identica, 99% delas do regime denso: um separador perfeito do regime
        E_MPA = 1.25                     # neo-Hookean, ver appendix/datasheet.tex
        pc = 2.0 * E_MPA / math.sqrt(3 * (1 - c["nu"] ** 2)) * (t / c["R"]) ** 2
    ds = sorted(defeitos[i], key=lambda x: -x[0])
    de = [x[0] for x in ds]
    lam = [x[1] for x in ds]
    xs, ys, zs = [], [], []
    for _, _, th, ph in ds:
        sp = math.sin(ph)
        xs.append(sp * math.cos(th)); ys.append(sp * math.sin(th)); zs.append(math.cos(ph))
    linhas.append(dict(
        id=i, origin=orig, held_out=False, in_release=i in canon,
        R=c["R"], t=t, R_over_t=c["R"] / t, nu=c["nu"], eta=c["eta"], p_classical=pc,
        N=len(ds), kappa=c["kappa"],
        delta=de, lam=lam, beta0_deg=[beta0(l, c["nu"], c["eta"]) for l in lam],
        x=xs, y=ys, z=zs))

df = pd.DataFrame(linhas)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
df.to_parquet(A.out, index=False)

print("escrito %s" % A.out)
print("  %d cascas, %d colunas" % (len(df), df.shape[1]))
print("\n%-12s %6s %8s %10s %14s %16s" % ("origem", "n", "na rel.", "N", "kappa", "beta0 (graus)"))
for o, g in df.groupby("origin"):
    b = np.concatenate(g["beta0_deg"].to_numpy())
    print("%-12s %6d %8d %10s %14s %16s"
          % (o, len(g), int(g["in_release"].sum()), "%d..%d" % (g.N.min(), g.N.max()),
             "%.3f..%.3f" % (g.kappa.min(), g.kappa.max()),
             "%.2f..%.2f" % (b.min(), b.max())))
if sem_familia:
    print("\n%d shells without a declared family, assigned by their number of defects" % len(sem_familia))
print("\nThe intervention tests P1-P4 are not in this table. They are added later,")
print("always with held_out=True, and never enter training.")
