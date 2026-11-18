"""Full-mesh track on B1: the 76,805-node finals cut to the 8,192 scoring nodes and scored like every row of the tables.

    python benchmark/tables/native_rescore.py [--families figconv,sfno,transolver,mgn] [--native-runs runs/native_final]
                                              [--out-dir outputs] [--tables-dir paper/tables/full]

No training and no inference: stored test predictions only.

INPUTS
  <native-runs>/phase2_<fam>_v2random_n76805_buckling.json   the result of the B1 search of a family on the full mesh
                                                             (confirm.py); its winner_rank names the final.
  <native-runs>/<fam>_v2random_fbuck76805_c<winner>[_s<n>]_test_pred.npz
                                                             test predictions of the B1 finals trained on all 76,805
                                                             nodes (train.py --nodes 76805 through confirm.py), one
                                                             file per seed; a file without the _s suffix is seed 0.
  <out-dir>/canon_nodes_8192.npz                             the scoring nodes (prepare_data.py), checked against
                                                             bliss.metrics.standard_nodes().
  <out-dir>/native_rescore_v2.json                           if it exists, the families that are not named in
                                                             --families are kept from it.
  --native-runs defaults to BLISS_NATIVE_RUNS or <BLISS_RUNS>/native_final, BLISS_RUNS to <BLISS_ROOT>/runs, --out-dir
  to BLISS_HEAVY or <BLISS_ROOT>/outputs, --tables-dir to <BLISS_ROOT>/paper/tables/full and BLISS_ROOT to the
  repository.

DEFINITIONS
  Scoring nodes: bliss.metrics.standard_nodes(), the 8,192 nodes every row of the tables is scored on. The stored
  fields pred_U and true_U and the node directions P are cut to them.
  Score of a file: bliss.metrics.score_ci(split="B1", 2,000 shell resamples), point value and 95 % interval of every
  metric, and score_by_regime(split="B1", 200 resamples), point values of the four headline metrics per regime.
  Score on the full mesh: bliss.metrics.score on all 76,805 nodes with the released site ratio q, stored and printed
  beside the score on the scoring nodes.
  A family without its search result has no full-mesh final and is stored as a message.
  Table: per family, the mean over the seeds of the four headline metrics with their range over the seeds.
Output: <out-dir>/native_rescore_v2.json (read by main_table.py and native_secondary.py) and
<tables-dir>/native_B1.tex.

Functions
---------
seed_of
    Training seed in the name of a prediction file.
cut_nodes
    Arrays of a full-mesh prediction file on a subset of the nodes.
table
    LaTeX table of the full-mesh track on B1.
main
    Score the B1 full-mesh finals and write native_rescore_v2.json and native_B1.tex.
"""
import argparse, glob, json, os, re, sys
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs"); RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
D = os.environ.get("BLISS_NATIVE_RUNS", RUNS + "/native_final"); OUT = B + "/paper/tables/full"
sys.path.insert(0, B)
from bliss.metrics import score_ci, score_by_regime, standard_nodes, score, released_q   # noqa: E402
H4 = [("kappa_rel_err", "%.2f"), ("buckling_r2", "%.3f"), ("loc_dominant_within_10", "%.2f"), ("two_way_winner", "%.2f")]
FAMS = (("figconv", "FigConvNet"), ("sfno", "SFNO"), ("transolver", "Transolver"), ("mgn", "MeshGraphNet"))
N_MESH = 76805      # nodes of the full mesh


N_BOOT, N_BOOT_REGIME = 2000, 200   # shell resamples of the intervals: all shells, and per defect spacing

def seed_of(f):
    """Training seed in the name of a prediction file.

    Parameters
    ----------
    f : str
        Path of a *_test_pred.npz file.

    Returns
    -------
    seed : int
        The seed, 0 when the name has no _s<n> tag.
    """
    m = re.search(r"_s(\d+)_test_pred\.npz$", f)
    return int(m.group(1)) if m else 0


def cut_nodes(z, nodes):
    """Arrays of a full-mesh prediction file on a subset of the nodes.

    Parameters
    ----------
    z : mapping
        Prediction file on the full mesh (ids, true_U, pred_U, true_K, pred_K, P, r).
    nodes : numpy.ndarray
        Indices of the nodes to keep.

    Returns
    -------
    cut : dict
        The same keys, with true_U, pred_U and P on the nodes kept.
    """
    return dict(ids=z["ids"], true_U=z["true_U"][:, nodes], pred_U=z["pred_U"][:, nodes], true_K=z["true_K"], pred_K=z["pred_K"],
                P=z["P"][nodes], r=z["r"])


def table(res, fams=FAMS):
    """LaTeX table of the full-mesh track on B1.

    Parameters
    ----------
    res : dict
        Family -> dict(name, seeds {seed: dict(metrics {metric: dict(point)})}), or a message for a family without a
        full-mesh final.
    fams : sequence of tuple, default=FAMS
        (family, printed name) of the families the caption can name as missing.

    Returns
    -------
    lines : list of str
        Lines of the table: one row per family with a final, the mean over the seeds and the range.
    """
    L = ["% Generated by benchmark/tables/native_rescore.py (outputs/native_rescore_v2.json).", "\\begin{table}[!ht]",
         "\\caption{\\textbf{Native track (76{,}805 nodes), B1}, predictions cut to the 8{,}192 scoring nodes and scored like every "
         "leaderboard row. Mean over the seeds (range).%s}" % "".join(" %s has no native final." % n for f, n in fams if not isinstance(res.get(f), dict)), "\\label{tab:native_b1}",
         "\\centering\\tiny\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{@{}lrrrr@{}}", "\\toprule",
         "model & $\\kappa$ err [\\%] & buckling $R^2$ & clear $\\le10^\\circ$ & near-tie \\\\", "\\midrule"]
    for fam, r in res.items():
        if not isinstance(r, dict) or not r["seeds"]: continue
        cs = []
        for m, f in H4:
            v = [x["metrics"][m]["point"] for x in r["seeds"].values() if m in x["metrics"]]
            cs.append((f % np.mean(v)) + "\\,{\\tiny(%s--%s)}" % (f % min(v), f % max(v)))
        L.append("%s & %s \\\\" % (r["name"], " & ".join(cs)))
    return L + ["\\bottomrule", "\\end{tabular}", "\\end{table}"]


def main():
    """Score the B1 full-mesh finals and write native_rescore_v2.json and native_B1.tex."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", default=",".join(f for f, _ in FAMS), help="families to score; the others are kept from the stored json")
    ap.add_argument("--native-runs", default=D, help="directory of the full-mesh finals")
    ap.add_argument("--out-dir", default=H, help="directory of canon_nodes_8192.npz and of the json written")
    ap.add_argument("--tables-dir", default=OUT, help="directory of the LaTeX table written")
    a = ap.parse_args(); only = a.families.split(",")
    for p, what in ((a.out_dir + "/canon_nodes_8192.npz", "the scoring nodes, written by benchmark/data/prepare_data.py"),
                    (a.native_runs, "the directory of the full-mesh finals (--native-runs or BLISS_NATIVE_RUNS)")):
        if not os.path.exists(p): sys.exit("native_rescore.py: missing input %s (%s)" % (p, what))
    NODES = standard_nodes(); assert np.array_equal(NODES, np.load(a.out_dir + "/canon_nodes_8192.npz")["NODES"])
    JS = a.out_dir + "/native_rescore_v2.json"
    old = json.load(open(JS)) if os.path.exists(JS) else {}
    res = {}
    for fam, name in FAMS:
        if fam not in only and fam in old: res[fam] = old[fam]; continue
        pj = "%s/phase2_%s_v2random_n%d_buckling.json" % (a.native_runs, fam, N_MESH)
        if not os.path.exists(pj): res[fam] = "no native final (%s missing)" % os.path.basename(pj); continue
        rank = json.load(open(pj))["winner_rank"]; rows = {}
        for f in sorted(glob.glob("%s/%s_v2random_fbuck%d_c%d*_test_pred.npz" % (a.native_runs, fam, N_MESH, rank))):
            s = seed_of(f); z = np.load(f); assert z["pred_U"].shape[1] == N_MESH
            cut = cut_nodes(z, NODES)
            sc = score_ci(cut, n_boot=N_BOOT, split="B1"); rg = score_by_regime(cut, n_boot=N_BOOT_REGIME, split="B1")
            Pf = np.asarray(z["P"], np.float64); Pf /= np.linalg.norm(Pf, axis=1, keepdims=True)
            full = score(z["true_U"], z["pred_U"], z["true_K"], z["pred_K"], Pf, None, released_q(z["ids"]))
            rows[s] = dict(file=f, metrics={k: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for k, v in sc["metrics"].items()},
                           by_regime={k: {m_: v["metrics"][m_]["point"] for m_, _ in H4 if m_ in v["metrics"]} for k, v in rg.items()},
                           full_mesh_score={k: v for k, v in full.items() if not isinstance(v, dict)})
            print(fam, s, {k: round(rows[s]["metrics"][k]["point"], 4) for k, _ in H4 if k in rows[s]["metrics"]}, "full-mesh buckling_r2 %.4f" % full["buckling_r2"], flush=True)
            del z, cut
        res[fam] = dict(name=name, winner_rank=rank, phase2=pj, seeds=rows)
    json.dump(res, open(JS, "w"), indent=1, default=float)
    os.makedirs(a.tables_dir, exist_ok=True)
    open(a.tables_dir + "/native_B1.tex", "w").write("\n".join(table(res)) + "\n"); print("wrote native_B1.tex")


if __name__ == "__main__":
    main()
