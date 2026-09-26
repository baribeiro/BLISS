"""Main benchmark table (Section 5, tab:main), v3 of 2026-09-25: five columns per protocol.

    python benchmark/tables/main_table.py            -> paper_iclr/tables/main_results.tex

Every cell is copied from the printed full tables (paper_iclr/tables/full_B{1,2,3}.tex, themselves generated from
outputs/rescore_v2.json and, for the SFNO 64x128 row, outputs/g64_scores_all.json), so the main table and the full
tables cannot disagree. Columns per protocol: median relative kappa error [%], buckling R^2, RMSE of the field after
removing the uniform shrink [mm], mean relative L2 error after removing the shrink [%], clear-site accuracy [%].
The depth rules have no row in the full tables; their site accuracy is kept from the previous main table
(rascunhos/results_v1/make_results_tables.py), whose file is read here before being overwritten.
Bold: the best learned model per column among the 8,192-node rows (geometry models, SFNO 64x128, the MLPs and XGBoost).

Full-mesh block (2026-09-25): the native track, trained on all 76,805 nodes and scored on the same 8,192 nodes, mean over
the seeds present. kappa, buckling R^2 and clear-site accuracy come from outputs/native_rescore_v2.json (B1, written by
native_rescore.py) and outputs/ladder_native_v2.json["native"] (B2/B3, ladder_native.py); RMSE and rel. L2 after
removing the shrink from outputs/native_secondary_v2.json (native_secondary.py, rescore_v2.secondary()). A row appears
as soon as its B1 artefacts are in native_rescore_v2.json (MeshGraphNet: native_rescore.py --families mgn), and its B2/B3
cells as soon as ladder_native.py has scored them; cells of a protocol without a native run stay empty, and "--" marks
an artefact whose secondary metrics are not computed yet (run native_secondary.py). Rounding as in the full tables
(the value is formatted to the full-table precision first, then converted).

Functions
---------
strip
    Remove the interval annotation from a table cell.
val
    Numeric value of a table cell.
read_full
    Read one full-results table into a dict of rows.
find
    Cells of the row whose label matches a key.
fmt
    Cell text from the full-table cell: percentages for rel. L2 and site, div. and dashes kept.
native_rows
    Rows of the full-mesh block: {fam: (label, [15 cells])}, only for families with a native B1 final.
"""
import json, os, re

import numpy as np

P = os.environ.get("BLISS_TABLES", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper", "tables"))
PROT = ("B1", "B2", "B3")
# column index in the full tables (after the row name): 0 kappa err, 3 buckling R2, 7 RMSE w/o shrink, 9 rel L2 w/o shrink
# (mean / median), 10 clear-site accuracy
COLS = [(0, "k"), (3, "r2"), (7, "rmse"), (9, "l2"), (10, "site")]


def strip(c):
    """Remove the interval annotation from a table cell.

    Parameters
    ----------
    c : str
        Cell text.

    Returns
    -------
    c : str
        Cell text without the interval.
    """
    c = re.sub(r"\\,\{\\tiny\[[^\]]*\]\}", "", c).strip()
    return c


def val(c):
    """Numeric value of a table cell.

    Parameters
    ----------
    c : str
        Cell text.

    Returns
    -------
    v : float or None
        The value, or None if the cell is not numeric.
    """
    c = c.replace("$-$", "-")
    try:
        return float(c)
    except ValueError:
        return None


def read_full(prot):
    """Read one full-results table into a dict of rows.

    Parameters
    ----------
    prot : str
        Protocol.

    Returns
    -------
    rows : dict
        Row label -> cells.
    """
    rows = {}
    for line in open("%s/full_%s.tex" % (P, prot)):
        if "&" not in line or line.startswith(("%", "\\", " &")):
            continue
        cells = [x.strip() for x in line.rstrip().rstrip("\\").split("&")]
        rows[cells[0]] = [strip(c) for c in cells[1:]]
    return rows


FULL = {p: read_full(p) for p in PROT}
OLD = open(P + "/main_results.tex").read()
depth = re.search(r"^depth rules\$\^\\ddagger\$ & (.*?) \\\\", OLD, re.M)
if depth:
    depth_site = [c.strip() for c in depth.group(1).replace("&&", "&").split("&")]
    depth_site = depth_site[2::4] if len(depth_site) == 12 else depth_site[4::5]  # site of B1, B2, B3 (old 4-col or new 5-col file)
# Rows left out of the main table to keep the conclusion on page 9 (they stay in the full tables).
# Rows left out with --drop (comma-separated labels as printed, e.g. "gradient boosting on $w$"); none by default.
import argparse
_ap = argparse.ArgumentParser(); _ap.add_argument("--drop", default=""); DROP = {x.strip() for x in _ap.parse_args().drop.split(",") if x.strip()}
STRETCH = 0.80                               # row spacing of the main table (page budget)

# (label in the main table, name in the full tables, learned?)
ROWS = ["references", ("training mean", "training mean", False),
        ("weakest-link rule", "weakest link from $w$", False),
        ("depth rules$^\\ddagger$", None, False),
        ("nearest neighbour on $w$", "nearest neighbour ($k=1$)", False),
        ("gradient boosting on $w$", "gradient boosting on $w$", False),
        ("solver, one step before", "solver, increment $k-1$", False),
        ("solver, one step after", "solver, increment $k+1$", False),
        "8{,}192 nodes",
        ("FigConvNet", "FigConvNet", True), ("MeshGraphNet", "MeshGraphNet", True), ("SFNO", "SFNO", True),
        ("SFNO-$64{\\times}128$$^\\ast$", "SFNO-$64\\times128$", True),
        ("Transformer", "Transformer", True), ("Transolver", "Transolver", True), ("DeepONet", "DeepONet", True),
        "NATIVE", "list of defects (parametric input)",
        ("MLP-20$^\\dagger$", "MLP-20", True), ("MLP-80$^\\dagger$", "MLP-80", True),
        ("XGBoost$^\\dagger$", "XGBoost", True)]


def find(prot, key):
    """Cells of the row whose label matches a key.

    Parameters
    ----------
    prot : str
        Protocol.
    key : str
        Row label or its prefix.

    Returns
    -------
    cells : list
        The cells.
    """
    hits = [k for k in FULL[prot] if k == key] or [k for k in FULL[prot] if k.startswith(key)]
    assert len(hits) == 1, (prot, key, hits)
    return FULL[prot][hits[0]]


def fmt(ci, kind):
    """Cell text from the full-table cell: percentages for rel. L2 and site, div. and dashes kept.

    Parameters
    ----------
    ci : str
        Cell of the full table.
    kind : str
        Kind of metric.

    Returns
    -------
    text : str
        Cell text for the main table.
    """
    if ci in ("--", "div.", "div. / div."):
        return "div." if ci.startswith("div") else "--"
    if kind == "l2":
        m = val(ci.split("/")[0].strip())
        return "--" if m is None else "%d" % round(100 * m)
    if kind == "site":
        v = val(ci)
        return "--" if v is None else "%d" % round(100 * v)
    return ci


H = os.environ.get("BLISS_HEAVY", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs"))
NATIVE = [("figconv", "FigConvNet"), ("mgn", "MeshGraphNet"), ("sfno", "SFNO"), ("transolver", "Transolver")]


def native_rows():
    """Rows of the full-mesh block: {fam: (label, [15 cells])}, only for families with a native B1 final.

    Returns
    -------
    rows : dict
        Family -> (label, cells).
    """
    b1 = json.load(open(H + "/native_rescore_v2.json")); nat = json.load(open(H + "/ladder_native_v2.json"))["native"]
    sec = json.load(open(H + "/native_secondary_v2.json")) if os.path.exists(H + "/native_secondary_v2.json") else {}
    out = {}
    for fam, name in NATIVE:
        if not isinstance(b1.get(fam), dict) or not b1[fam]["seeds"]:
            continue
        cells, prots = [], []
        for p in PROT:
            seeds = b1[fam]["seeds"] if p == "B1" else nat.get("%s|%s" % (fam, p), {})
            if not seeds:
                cells += [""] * 5; continue
            prots.append(p)
            m = lambda k: np.mean([x["metrics"][k]["point"] for x in seeds.values()])
            sc = sec.get("%s|%s" % (fam, p), {})
            have = all(s_ in sc and sc[s_]["file"] == x["file"] for s_, x in seeds.items())
            sv = lambda k: np.mean([sc[s_]["secondary"][k] for s_ in seeds]) if have else None
            cells += ["%.2f" % m("kappa_rel_err"), "%.3f" % m("buckling_r2"),
                      "--" if not have else "%.4f" % sv("rmse_mm_deviation"),
                      "--" if not have else "%d" % round(100 * float("%.3f" % sv("rel_l2_deviation"))),
                      "%d" % round(100 * float("%.2f" % m("loc_dominant_within_10")))]
        out[fam] = (name, cells, prots)
    return out


NAT = native_rows()
table = []                                   # (label, learned, [15 cells]) or None
for r in ROWS:
    if r is None or isinstance(r, str):
        table.append(r); continue
    lab, key, learned = r
    if lab in DROP:
        continue
    cells = []
    for i, p in enumerate(PROT):
        if key is None:
            cells += ["--", "--", "--", "--", depth_site[i]]
            continue
        c = find(p, key)
        for j, kind in COLS:
            x = fmt(c[j], kind)
            if kind == "rmse" and ("div" in c[1] or "div" in c[0]) and x not in ("--",):
                x = "div."                  # a diverged run has a meaningless RMSE (MLP-80 on B3)
            cells.append(x)
    table.append((lab, learned, cells))

# bold the best learned model per column
lower = {"k": True, "r2": False, "rmse": True, "l2": True, "site": False}
kinds = [k for _ in PROT for _, k in COLS]
for j, kind in enumerate(kinds):
    vals = [(val(t[2][j]), n) for n, t in enumerate(table) if isinstance(t, tuple) and t[1] and val(t[2][j]) is not None]
    best = (min if lower[kind] else max)(v for v, _ in vals)
    for v, n in vals:
        if v == best:
            table[n][2][j] = "\\textbf{%s}" % table[n][2][j]

neg = lambda s: re.sub(r"(?<![$\w])-(?=\d)", "$-$", s)
full = [n for f, (n, _, pr) in NAT.items() if pr == list(PROT)]
L = ["% Generated by benchmark/tables/main_table.py from tables/full_B{1,2,3}.tex (rescore_v2.json, g64_scores_all.json) and,",
     "% for the full-mesh block, outputs/native_rescore_v2.json, ladder_native_v2.json and native_secondary_v2.json.",
     "\\begin{table}[t]",
     "\\caption{\\textbf{Benchmark results.} B1 plays the role of existing benchmarks, and on B3 every geometry model falls "
     "below the training mean. Learned models give the mean over three seeds, and field metrics are taken after removing the "
     "uniform shrink. Bold marks the best learned model on 8{,}192 nodes. Near-tie winner rates are in "
     "Figure~\\ref{fig:ladder}d and App.~\\ref{app:full}. $^\\dagger$Reads the list of defects. $^\\ast$SFNO on a "
     "$64{\\times}128$ grid. $^\\ddagger$Deepest point of $w$. div.\\ diverged."
     + (" Full-mesh B2 and B3 are reported for %s." % " and ".join(full) if full and len(full) < len(NAT) else "") + "}",
     "\\label{tab:main}\\label{tab:random}",
     "\\centering\\scriptsize\\setlength{\\tabcolsep}{1pt}\\renewcommand{\\arraystretch}{%s}" % STRETCH,
     "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l ccccc ccccc ccccc@{}}",
     "\\toprule",
     " & \\multicolumn{5}{c}{B1 within distribution} & \\multicolumn{5}{c}{B2 deeper defects} & \\multicolumn{5}{c}{B3 sparse to dense} \\\\",
     "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\\cmidrule(lr){12-16}",
     " " + " & MRE & $R^2$ & RMSE & rel.\\ $L^2$ & site" * 3 + " \\\\",
     " " + "".join(" & {\\tiny %s}" % h for h in ("$\\kappa$ [\\%]", "$\\mathbf U$", "$\\mathbf U$ [mm]", "$\\mathbf U$ [\\%]", "clear [\\%]")) * 3 + " \\\\",
     "\\midrule"]
row = lambda lab, c: "%s & %s & %s & %s \\\\" % (lab, " & ".join(neg(x) for x in c[:5]), " & ".join(neg(x) for x in c[5:10]),
                                                 " & ".join(neg(x) for x in c[10:]))
for t in table:
    if t is None:
        L.append("\\midrule")
    elif t == "NATIVE":
        if NAT:
            L += ["\\midrule", "\\multicolumn{16}{@{}l}{\\emph{full mesh, 76{,}805 nodes (scored on the 8{,}192)}} \\\\"]
            L += [row(n, c) for n, c, _ in NAT.values()]
    elif isinstance(t, str):
        L += ([] if L[-1] == "\\midrule" else ["\\midrule"]) + ["\\multicolumn{16}{@{}l}{\\emph{%s}} \\\\" % t]
    else:
        L.append(row(t[0], t[2]))
L += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
open(P + "/main_results.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L[-32:]))
