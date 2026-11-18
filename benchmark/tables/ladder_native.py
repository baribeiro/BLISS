"""Resolution ladder on B1 (2,048 against 8,192 nodes) and the full-mesh (76,805-node) finals on B2 and B3, scored like
every row of the tables.

    python benchmark/tables/ladder_native.py [--families figconv,mgn,transformer,transolver] [--runs runs]
                                             [--native-runs runs/native_final] [--out-dir outputs]
                                             [--tables-dir paper/tables/full]

No training and no inference: stored test predictions only.

INPUTS
  ladder     <runs>/final_runs/<fam>_ladder_n2048/<fam>_v2random_sc2048buck[_s<n>]_test_pred.npz: test predictions of
             the B1 configuration trained on 2,048 nodes (transfer.py, train.py --nodes 2048), stored on those nodes
             only, one file per seed. The 2,048 nodes are np.sort(default_rng(1).choice(8192, 2048, replace=False))
             of the 8,192 nodes, the subset train.py draws.
             <runs>/final_runs/<fam>_B1_n8192/<fam>_v2random_fbuck8192_c<winner>_hp2[_s<n>]_test_pred.npz: the
             8,192-node B1 final of the same family and seed, <winner> = winner_rank of
             phase2_<fam>_v2random_n8192_buckling_hp2.json in the same directory (confirm.py).
  full mesh  <native-runs>/<fam>_v2delta_pb2buck[_s<n>]_test_pred.npz (B2) and
             <native-runs>/<fam>_v2regime_pb3buck[_s<n>]_test_pred.npz (B3): test predictions of the finals trained on
             all 76,805 nodes (transfer.py). A file whose fields are not on 76,805 nodes is skipped.
  --runs defaults to BLISS_RUNS or <BLISS_ROOT>/runs, --native-runs to BLISS_NATIVE_RUNS or <runs>/native_final,
  --out-dir to BLISS_HEAVY or <BLISS_ROOT>/outputs, --tables-dir to <BLISS_ROOT>/paper/tables/full and BLISS_ROOT to
  the repository.

DEFINITIONS
  Score: bliss.metrics.score_ci(split=<protocol>, 2,000 shell resamples), point value and 95 % interval of every
  metric.
  Ladder, per family and seed, three scores on the same test shells of B1:
    n2048          the 2,048-node model on its 2,048 nodes;
    n8192_on_2048  the 8,192-node model restricted to the same 2,048 nodes, which compares like with like;
    n8192          the 8,192-node model on its 8,192 nodes, the score of the tables.
  The two files must hold the same shells and the same node positions on the 2,048 nodes.
  Full mesh: pred_U, true_U and P are cut to bliss.metrics.standard_nodes(), the 8,192 nodes every row of the tables is
  scored on, and scored with the split of the protocol.
  Tables: the mean over the seeds of the four headline metrics, with the range over the seeds when there is more
  than one.
Output: <out-dir>/ladder_native_v2.json with the keys "ladder" and "native" (the latter read by main_table.py and
native_secondary.py), <tables-dir>/ladder_B1.tex and <tables-dir>/native_B2B3.tex.

Functions
---------
pts
    Point value and interval of every metric of a score_ci result.
npz_shape
    Shape of one array of an npz file, read from its header only.
ms
    Table cell: mean over seeds, and the range when there is more than one seed.
ladder_table
    LaTeX table of the resolution ladder on B1.
native_table
    LaTeX table of the full-mesh track on B2 and B3.
main
    Score the ladder and the full-mesh B2/B3 finals and write the JSON and the two tables.
"""
import argparse, glob, json, os, re, sys, zipfile
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs"); R = os.environ.get("BLISS_RUNS", B + "/runs")
OUT = B + "/paper/tables/full"
sys.path[:0] = [B, os.path.dirname(os.path.abspath(__file__))]
from bliss.metrics import score_ci, standard_nodes   # noqa: E402
from native_rescore import H4, N_MESH, cut_nodes, seed_of   # noqa: E402
NM = {"figconv": "FigConvNet", "mgn": "MeshGraphNet", "transformer": "Transformer", "transolver": "Transolver", "sfno": "SFNO", "deeponet": "DeepONet"}
FAMILIES = ["figconv", "mgn", "transformer", "transolver"]      # families of the ladder
N_FULL, N_SUB = 8192, 2048                                      # nodes of the tables and of the smaller run of the ladder
SUBSET_SEED = 1                                                 # seed of the node subset that train.py draws with --nodes
N_BOOT = 2000                                                   # shell resamples of the intervals
LADDER = (("n2048", "2{,}048 / 2{,}048"), ("n8192_on_2048", "8{,}192 / same 2{,}048"), ("n8192", "8{,}192 / 8{,}192"))


def pts(sc):
    """Point value and interval of every metric of a score_ci result.

    Parameters
    ----------
    sc : dict
        Output of bliss.metrics.score_ci.

    Returns
    -------
    out : dict
        Metric -> dict(point, lo, hi).
    """
    return {k: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for k, v in sc["metrics"].items()}


def npz_shape(f, key="pred_U"):
    """Shape of one array of an npz file, read from its header only.

    Parameters
    ----------
    f : str
        Path of the npz file.
    key : str, default="pred_U"
        Array name.

    Returns
    -------
    shape : tuple
        Shape of the array.
    """
    with zipfile.ZipFile(f) as z, z.open(key + ".npy") as fh:
        v = np.lib.format.read_magic(fh); return (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)[0]


def ms(d, m, f):
    """Table cell: mean over seeds, and the range when there is more than one seed.

    Parameters
    ----------
    d : list of dict
        Metrics of each seed.
    m : str
        Metric name.
    f : str
        Number format.

    Returns
    -------
    cell : str
        LaTeX cell.
    """
    v = [x[m]["point"] for x in d]
    return (f % np.mean(v)) + ("\\,{\\tiny(%s--%s)}" % (f % min(v), f % max(v)) if len(v) > 1 else "")


def ladder_table(ladder):
    """LaTeX table of the resolution ladder on B1.

    Parameters
    ----------
    ladder : dict
        Family -> {seed: dict(n2048, n8192_on_2048, n8192)}, each the output of pts.

    Returns
    -------
    lines : list of str
        Lines of the table, three rows per family.
    """
    L = ["% Generated by benchmark/tables/ladder_native.py (outputs/ladder_native_v2.json).", "\\begin{table}[!ht]",
         "\\caption{\\textbf{Node-resolution ladder, B1.} The same configuration trained on 2{,}048 or 8{,}192 nodes; the 8{,}192-node model is "
         "also scored on the 2{,}048 nodes of the smaller run, so the first two rows of a model compare like with like. Mean over the "
         "seeds (range when there is more than one).}", "\\label{tab:ladder}", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}l l r r r r@{}}", "\\toprule",
         "model & trained / scored on & $\\kappa$ err [\\%] & buckling $R^2$ & clear $\\le10^\\circ$ & two-way \\\\", "\\midrule"]
    for fam, rows in ladder.items():
        for key, lab in LADDER:
            L.append("%s & %s & %s \\\\" % (NM[fam], lab, " & ".join(ms([r[key] for r in rows.values()], m, f) for m, f in H4)))
        L.append("\\addlinespace[1pt]")
    return L + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]


def native_table(native):
    """LaTeX table of the full-mesh track on B2 and B3.

    Parameters
    ----------
    native : dict
        '<family>|<protocol>' -> {seed: dict(metrics)}, metrics the output of pts.

    Returns
    -------
    lines : list of str
        Lines of the table, one row per family and protocol in sorted order.
    """
    L = ["% Generated by benchmark/tables/ladder_native.py (outputs/ladder_native_v2.json).", "\\begin{table}[!ht]",
         "\\caption{\\textbf{Native track (76{,}805 nodes) on B2 and B3}, predictions cut to the 8{,}192 scoring nodes; mean over seeds (range).}",
         "\\label{tab:native_b2b3}", "\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}l l r r r r r@{}}", "\\toprule",
         "model & protocol & seeds & $\\kappa$ err [\\%] & buckling $R^2$ & clear $\\le10^\\circ$ & two-way \\\\", "\\midrule"]
    for key, rows in sorted(native.items()):
        fam, prot = key.split("|"); L.append("%s & %s & %d & %s \\\\" % (NM.get(fam, fam), prot, len(rows), " & ".join(ms([r["metrics"] for r in rows.values()], m, f) for m, f in H4)))
    return L + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]


def main():
    """Score the ladder and the full-mesh B2/B3 finals and write the JSON and the two tables."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(FAMILIES), help="families of the ladder")
    ap.add_argument("--runs", default=R, help="directory of the trained models (holds final_runs/)")
    ap.add_argument("--native-runs", default=None, help="directory of the full-mesh finals (default: BLISS_NATIVE_RUNS or <runs>/native_final)")
    ap.add_argument("--out-dir", default=H, help="directory of the json written")
    ap.add_argument("--tables-dir", default=OUT, help="directory of the LaTeX tables written")
    a = ap.parse_args(); D = a.native_runs or os.environ.get("BLISS_NATIVE_RUNS", a.runs + "/native_final")
    for p, what in ((a.runs + "/final_runs", "the directory of the 8,192- and 2,048-node finals (--runs or BLISS_RUNS)"),
                    (D, "the directory of the full-mesh finals (--native-runs or BLISS_NATIVE_RUNS)")):
        if not os.path.exists(p): sys.exit("ladder_native.py: missing input %s (%s)" % (p, what))
    SUB = np.sort(np.random.default_rng(SUBSET_SEED).choice(N_FULL, N_SUB, replace=False)); res = dict(ladder={}, native={})
    for fam in a.families.split(","):
        win = json.load(open("%s/final_runs/%s_B1_n%d/phase2_%s_v2random_n%d_buckling_hp2.json" % (a.runs, fam, N_FULL, fam, N_FULL)))["winner_rank"]
        rows = {}
        for f in sorted(glob.glob("%s/final_runs/%s_ladder_n%d/%s_v2random_sc%dbuck*_test_pred.npz" % (a.runs, fam, N_SUB, fam, N_SUB))):
            s = seed_of(f); z = dict(np.load(f)); x = pts(score_ci(z, n_boot=N_BOOT, split="B1"))
            g = "%s/final_runs/%s_B1_n%d/%s_v2random_fbuck%d_c%d_hp2%s_test_pred.npz" % (a.runs, fam, N_FULL, fam, N_FULL, win, "_s%d" % s if s else "")
            y = dict(np.load(g)); assert np.allclose(y["P"][SUB], z["P"]) and np.array_equal(y["ids"], z["ids"]), "node subset / ids differ"
            y2 = dict(y, pred_U=y["pred_U"][:, SUB], true_U=y["true_U"][:, SUB], P=y["P"][SUB])
            b = pts(score_ci(y2, n_boot=N_BOOT, split="B1")); c = pts(score_ci(y, n_boot=N_BOOT, split="B1"))
            rows[s] = dict(n2048=x, n8192_on_2048=b, n8192=c, files=[f, g]); print(fam, s, "2048 %.4f | 8192 on same 2048 %.4f | 8192 %.4f" % (x["buckling_r2"]["point"], b["buckling_r2"]["point"], c["buckling_r2"]["point"]), flush=True)
        res["ladder"][fam] = rows
    for f in sorted(glob.glob(D + "/*_v2*_pb[23]buck*_test_pred.npz")):
        fam, spl = re.search(r"/(\w+?)_v2(delta|regime)_pb", f).groups(); prot = {"delta": "B2", "regime": "B3"}[spl]
        shp = npz_shape(f)
        if shp[1] != N_MESH: print("skip (not on the full mesh):", os.path.basename(f)); continue
        NODES = standard_nodes(); z = np.load(f)
        cut = cut_nodes(z, NODES)
        res["native"].setdefault("%s|%s" % (fam, prot), {})[seed_of(f)] = dict(file=f, metrics=pts(score_ci(cut, n_boot=N_BOOT, split=prot)))
        print("native", fam, prot, seed_of(f), round(res["native"]["%s|%s" % (fam, prot)][seed_of(f)]["metrics"]["buckling_r2"]["point"], 4), flush=True)
        del z, cut
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(a.out_dir + "/ladder_native_v2.json", "w"), indent=1, default=float)
    os.makedirs(a.tables_dir, exist_ok=True)
    open(a.tables_dir + "/ladder_B1.tex", "w").write("\n".join(ladder_table(res["ladder"])) + "\n")
    open(a.tables_dir + "/native_B2B3.tex", "w").write("\n".join(native_table(res["native"])) + "\n"); print("wrote " + a.out_dir + "/ladder_native_v2.json")


if __name__ == "__main__":
    main()
