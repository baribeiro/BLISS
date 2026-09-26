"""Full-results tables (appendix "Full results"), generated from stored artefacts only; every cell is traceable.

    python benchmark/tables/full_tables.py

Sources (read only):
  outputs/rescore_v2.json          learned field models (8 families x B1/B2/B3 x 3 seeds), the mean field, the w-based
                                    references (knn, knn1, pca_ridge, gb_w, regime_mean, weakest_link_w): score_ci with
                                    2,000 shell resamples per seed artefact, and score_by_regime points (pooled/sparse/dense)
  outputs/oracle_B1_topk_c1_xgboost_kn_final.json, outputs/xgb_b1frozen_v2.json   xgboost (B1 config frozen on B2/B3)
  outputs/set_{flat,deepsets,pairwise}_B{1,2,3}_residual_f2ne_kn_final.json      set models (tuned per split)
  outputs/increment_reference_v2.json     solver k-1 / k+1 rows
Outputs (paper_iclr/rascunhos/fullresults_v1/):
  full_B1.tex, full_B2.tex, full_B3.tex   per split, ONE landscape table (sidewaystable) with grouped columns: kappa
                                          (median rel. error, R^2, unconservative rate) | field (buckling R^2, field R^2,
                                          dimple R^2, RMSE full and without shrink, rel. L2 full and without shrink) |
                                          failure site (clear <= 10 deg, two-way) | by regime (B1, B2). Intervals only on
                                          the four headline columns (mean of per-seed 95 % CIs); the rest 3-seed means
  seed_spread.tex                         per family and split: the three per-seed values of the four headlines
  full_tables_sources.json                the file behind every row

Functions
---------
fmt
    Format a number, or '--' when missing.
row_learned
    Table row of one learned family (mean and range over seeds).
unconservative
    Share of shells whose predicted kappa exceeds the true one by more than 1 %.
row_kappa_only
    Table row of a model that predicts kappa only.
table
    LaTeX full-results table of one protocol.
seed_spread
    LaTeX table of the headline metrics of each learned family per training seed.
"""
import glob, json, os
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"; OUT = B + "/paper/tables/full"
os.makedirs(OUT, exist_ok=True)
GEO = [("figconv", "FigConvNet"), ("sfno", "SFNO"), ("transformer", "Transformer"), ("transolver", "Transolver"),
       ("mgn", "MeshGraphNet"), ("deeponet", "DeepONet")]
ORA = [("mlp", "MLP (top-20 defects)"), ("mlp80", "MLP (top-80 defects)")]
REF = [("meanfield", "training mean"), ("regime_mean", "regime-conditional mean"), ("knn1", "nearest neighbour ($k=1$)"),
       ("knn", "$k$ nearest neighbours"), ("pca_ridge", "PCA and ridge"), ("gb_w", "gradient boosting on $w$"),
       ("weakest_link_w", "weakest link from $w$")]
HEAD = [("kappa_rel_err", "$\\kappa$ err [\\%]", "%.2f", 1), ("buckling_r2", "buckling $R^2$", "%.3f", 1),
        ("loc_dominant_within_10", "clear $\\le10^\\circ$", "%.2f", 1), ("two_way_winner", "two-way", "%.2f", 1)]
SEC = [("kappa_r2", "$R^2_\\kappa$", "%.3f"), ("field_r2", "field $R^2$", "%.3f"), ("buckling_dimple_r2", "dimple $R^2$", "%.3f")]
R = json.load(open(H + "/rescore_v2.json"))["rows"]
_W = json.load(open(H + "/references_w_v2.json"))["models"]
REF = [(k, n + (" (tuned: %s)" % ", ".join("%s=%s" % (a, ("%.3g" % b) if isinstance(b, float) else b) for a, b in _W[k]["chosen"].items())
                if k in _W and _W[k].get("trials", 0) > 1 else "")) for k, n in REF]
SRC = {}


def fmt(v, f):
    """Format a number, or '--' when missing.

    Parameters
    ----------
    v : float or None
        Value.
    f : str
        Format.

    Returns
    -------
    text : str
        Formatted value.
    """
    return "--" if v is None or not np.isfinite(v) else (f % v)


# One landscape table per protocol. Column groups; "ci" columns carry the mean of the per-seed 95 % intervals.
COLS = [("$\\kappa$", [("kappa_rel_err", "median rel.\\ err [\\%]", "%.2f", "m", True), ("kappa_r2", "$R^2$", "%.3f", "m", False),
                   ("kappa_unconservative_rate", "unconserv.", "%.2f", "s", False)]),
        ("field", [("buckling_r2", "buckling $R^2$", "%.3f", "m", True), ("field_r2", "field $R^2$", "%.3f", "m", False),
                   ("buckling_dimple_r2", "dimple $R^2$", "%.3f", "m", False), ("rmse_mm_full", "RMSE [mm]", "%.4f", "s", False),
                   ("rmse_mm_deviation", "RMSE w/o shrink", "%.4f", "s", False), ("rel_l2_full", "rel.\\ $L^2$", "%.3f", "s", False),
                   ("rel_l2_deviation", "rel.\\ $L^2$ w/o shrink", "%.3f", "s", False)]),
        ("failure site", [("loc_dominant_within_10", "clear $\\le10^\\circ$", "%.2f", "m", True), ("two_way_winner", "two-way", "%.2f", "m", True)])]
REG = ("by regime", [("sparse", "kappa_rel_err", "sparse $\\kappa$ err", "%.2f"), ("sparse", "buckling_r2", "sparse bkl.\\ $R^2$", "%.3f"),
                     ("dense", "kappa_rel_err", "dense $\\kappa$ err", "%.2f"), ("dense", "buckling_r2", "dense bkl.\\ $R^2$", "%.3f")])


def row_learned(key, name, prot):
    """Table row of one learned family (mean and range over seeds).

    Parameters
    ----------
    key : str
        Family key in rescore.json.
    name : str
        Row label.
    prot : str
        Protocol.

    Returns
    -------
    row : list or None
        Cells of the row.
    """
    r = R.get("%s|%s" % (key, prot))
    if r is None: return None
    seeds = list(r["seeds"].values()); SRC["%s|%s" % (key, prot)] = [s["file"] for s in seeds]
    cells = []
    for _, cols in COLS:
        for m, _, f, where, ci in cols:
            if where == "m":
                v = [s["metrics"][m]["point"] for s in seeds if m in s["metrics"]]
                lo = [s["metrics"][m]["lo"] for s in seeds if m in s["metrics"]]; hi = [s["metrics"][m]["hi"] for s in seeds if m in s["metrics"]]
            else:
                v = [s["secondary"][m] for s in seeds if m in s.get("secondary", {})]; lo = hi = []
            if not v: cells.append("--"); continue
            c = fmt(np.mean(v), f)
            if ci and len(seeds) > 1 and lo: c += "\\,{\\tiny[%s, %s]}" % (fmt(np.mean(lo), f), fmt(np.mean(hi), f))
            cells.append(c)
    parts = []
    for reg, m, _, f in REG[1]:
        v = [s["by_regime"][reg]["metrics"].get(m) for s in seeds if reg in s["by_regime"]]; v = [x for x in v if x is not None]
        parts.append(fmt(np.mean(v), f) if v else "--")
    return name, cells, parts


def unconservative(pred_npz):
    """Share of shells whose predicted kappa exceeds the true one by more than 1 %.

    Parameters
    ----------
    pred_npz : str
        Prediction file.

    Returns
    -------
    rate : float or None
        Unconservative rate.
    """
    if not os.path.exists(pred_npz): return None
    z = np.load(pred_npz); P = z["pred_K_per_seed"] if "pred_K_per_seed" in z.files else z["pred_K"][None]
    return float(np.mean([np.mean(p > 1.01 * z["true_K"]) for p in P]))


def row_kappa_only(name, d, prot, src, pred=None):
    """Table row of a model that predicts kappa only.

    Parameters
    ----------
    name : str
        Row label.
    d : dict
        Metrics.
    prot : str
        Protocol.
    src : list
        Source files.
    pred : str
        Prediction file.

    Returns
    -------
    row : list
        Cells of the row.
    """
    SRC["%s|%s" % (name, prot)] = src
    u = unconservative(pred) if pred else None
    ncell = sum(len(c) for _, c in COLS)
    cells = ["%.2f" % d["test_err_mean"], "%.3f" % d["test_r2_mean"], fmt(u, "%.2f") if u is not None else "--"] + ["--"] * (ncell - 3)
    return name, cells, ["--"] * 4


def table(prot):
    """LaTeX full-results table of one protocol.

    Parameters
    ----------
    prot : str
        Protocol.

    Returns
    -------
    tex : str
        The table.
    """
    rows = [("geometry models (8{,}192 nodes)", [row_learned(k, n, prot) for k, n in GEO]),
            ("oracles on the defect list (field)", [row_learned(k, n, prot) for k, n in ORA])]
    tab = []
    f = H + "/oracle_B1_topk_c1_xgboost_kn_final.json"
    if prot == "B1" and os.path.exists(f):
        tab.append(row_kappa_only("XGBoost (B1 search)", json.load(open(f)), prot, f, H + "/oracle_B1_topk_c1_xgboost_kn_pred.npz"))
    fz = H + "/xgb_b1frozen_v2.json"
    if prot != "B1" and os.path.exists(fz):
        d = json.load(open(fz)).get(prot)
        if isinstance(d, dict): tab.append(row_kappa_only("XGBoost (B1 configuration, frozen)", d, prot, fz, H + "/oracle_%s_topk_xgbfrozen_kn_frozen_B1_pred.npz" % prot))
    for arch, an in (("flat", "flat"), ("deepsets", "DeepSets"), ("pairwise", "pairwise")):
        f = H + "/set_%s_%s_residual_f2ne_kn_final.json" % (arch, prot)
        if os.path.exists(f): tab.append(row_kappa_only("%s set model (tuned on %s)" % (an, prot), json.load(open(f)), prot, f))
    rows.append(("oracles on the defect list ($\\kappa$ only)", tab))
    rows.append(("references", [row_learned(k, n, prot) for k, n in REF]))
    inc = H + "/increment_reference_v2.json"
    if os.path.exists(inc):
        I = json.load(open(inc))["rows"]; sr = []
        for tag in ("k-1", "k+1"):
            x = I.get("%s|%s" % (prot, tag))
            if not x: continue
            mm = x["metrics"]; SRC["solver %s|%s" % (tag, prot)] = inc; cells = []
            for _, cols in COLS:
                for m, _, f, where, _ in cols:
                    cells.append(fmt(mm[m]["point"], f) if where == "m" and m in mm and not m.startswith("kappa") else "--")
            sr.append(("solver, increment $%s$" % tag, cells, ["--"] * 4))
        rows.append(("solver (label resolution)", sr))
    reg = prot != "B3"
    groups = COLS + ([REG] if reg else [])
    ncol = 1 + sum(len(c) for _, c in COLS) + (4 if reg else 0)
    hdr1 = " & " + " & ".join("\\multicolumn{%d}{c}{%s}" % (len(c), g) for g, c in groups) + " \\\\"
    cm, i = [], 2
    for _, c in groups: cm.append("\\cmidrule(lr){%d-%d}" % (i, i + len(c) - 1)); i += len(c)
    hdr2 = " & " + " & ".join(h for _, cols in COLS for _, h, _, _, _ in cols) + ((" & " + " & ".join(h for _, _, h, _ in REG[1])) if reg else "") + " \\\\"
    L = ["% Generated by benchmark/tables/full_tables.py from outputs/rescore_v2.json and the files in full_tables_sources.json.",
         "% Needs \\usepackage{rotating} (sidewaystable) and \\usepackage{booktabs}.",
         "\\begin{sidewaystable}", "\\caption{\\textbf{Full results, %s} (all %s test shells, 8{,}192 scoring nodes). Learned field models: mean "
         "over three training seeds; the headline columns ($\\kappa$ median error, buckling $R^2$, clear winners within $10^\\circ$, "
         "two-way winner) carry the mean of the per-seed 95\\%% bootstrap intervals over shells (2{,}000 resamples), the other columns "
         "are 3-seed means. Unconservative: share of shells with $\\hat\\kappa>1.01\\,\\kappa$. \\emph{w/o shrink}: after removing the uniform "
         "contraction. References and the solver: one run. Dashes: the row has no such output.%s}"
         % (prot, {"B1": "660", "B2": "1{,}234", "B3": "2{,}200"}[prot], " By regime: 3-seed means of score() on the sparse and dense parts." if reg else ""),
         "\\label{tab:full_%s}" % prot.lower(), "\\centering\\tiny\\setlength{\\tabcolsep}{2pt}",
         "\\begin{tabular}{@{}l" + "r" * (ncol - 1) + "@{}}", "\\toprule", hdr1, " ".join(cm), hdr2, "\\midrule"]
    for title, rs in rows:
        rs = [r for r in rs if r]
        if not rs: continue
        L.append("\\multicolumn{%d}{@{}l}{\\emph{%s}}\\\\" % (ncol, title))
        for name, cells, parts in rs:
            L.append(name + " & " + " & ".join(cells) + ((" & " + " & ".join(parts)) if reg else "") + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{sidewaystable}"]
    open("%s/full_%s.tex" % (OUT, prot), "w").write("\n".join(L) + "\n")


def seed_spread():
    """LaTeX table of the headline metrics of each learned family per training seed.

    Returns
    -------
    tex : str
        The table.
    """
    L = ["% Generated by benchmark/tables/full_tables.py from outputs/rescore_v2.json (per-seed score_ci points).",
         "\\begin{table}[!ht]", "\\caption{\\textbf{Seed spread.} The four headline metrics of each learned field model for "
         "training seeds 0 / 1 / 2 on each protocol (pooled test set).}", "\\label{tab:seed_spread}",
         "\\centering\\tiny\\setlength{\\tabcolsep}{2pt}", "\\begin{tabular}{@{}ll" + "r" * len(HEAD) + "@{}}", "\\toprule",
         "model & protocol & " + " & ".join(h for _, h, _, _ in HEAD) + " \\\\", "\\midrule"]
    for k, n in GEO + ORA:
        for prot in ("B1", "B2", "B3"):
            r = R.get("%s|%s" % (k, prot))
            if not r: continue
            cs = []
            for m, _, f, _ in HEAD:
                v = [r["seeds"][s]["metrics"].get(m, {}).get("point") for s in sorted(r["seeds"])]
                cs.append(" / ".join(fmt(x, f) for x in v))
            L.append("%s & %s & %s \\\\" % (n, prot, " & ".join(cs)))
        L.append("\\addlinespace[1pt]")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(OUT + "/seed_spread.tex", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    for p in ("B1", "B2", "B3"): table(p)
    seed_spread()
    json.dump(SRC, open(OUT + "/full_tables_sources.json", "w"), indent=1)
    print("wrote", sorted(os.listdir(OUT)))
