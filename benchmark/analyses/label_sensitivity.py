"""Sensitivity of the benchmark to the labels: the model families scored against kappa labels re-solved under other
settings (solver increments, element types, material laws, a finer mesh), and the ranking of the families against
each set of labels compared with their ranking against the released labels.

    python benchmark/analyses/label_sensitivity.py --labels alt_labels.csv [--runs-dir runs/final_runs]
           [--protocol B1] [--groups groups.csv --group dense] [--settings a,b,...] [--families ...]

INPUTS (read only)
  --labels   csv with the columns shell, setting, kappa: the knockdown factor of a released shell (its id in the
             release) solved again under the named setting. Producing these labels is outside this script.
  predictions  <runs-dir>/<family>_<protocol>_n8192/*_test_pred.npz (keys ids, pred_K, true_K), one file per seed,
             as written by benchmark/field_models/train.py; true_K are the released labels.
  --groups   optional csv with the columns shell, group (e.g. sparse / dense); --group keeps the shells of one group.

DEFINITIONS
  shells scored = the shells that have a label under every setting (and, with --group, belong to that group). Every
       set of labels is scored on the same shells.
  error of a family against a set of labels = mean over its seeds of the median over those shells of
       |kappa_pred - kappa_label| / kappa_label, in percent (the kappa_rel_err of bliss.metrics on those shells,
       averaged over seeds).
  rank rho = Spearman rank correlation, over the families, of the errors against the labels of a setting and the
       errors against the released labels.
  label change = median over those shells of |kappa_label / kappa_released - 1|, in percent.
Output: outputs/label_sensitivity.json (label_sensitivity_<group>.json with --group; --out) and a printed table.

Functions
---------
read_labels
    Alternative labels from a csv: setting -> {shell: kappa}.
read_groups
    Group of every shell from a csv.
load_predictions
    Predicted kappa of every family and seed and the released labels, by shell id.
common_shells
    Shells with a label under every setting, optionally of one group.
family_error
    Error of a family against one set of labels.
score_labels
    Errors of every family against every set of labels, with the rank correlation against the released ranking.
main
    Score the model families against the alternative labels and write outputs/label_sensitivity.json.
"""
import argparse, csv, glob, json, os, sys
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"
RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ranking import PATTERN, discover_families, spearman   # noqa: E402


def read_labels(path):
    """Alternative labels from a csv: setting -> {shell: kappa}.

    Parameters
    ----------
    path : str
        csv with the columns shell, setting, kappa.

    Returns
    -------
    labels : dict
        setting -> {shell id: kappa}, settings in the order of the file.
    """
    L = {}
    for r in csv.DictReader(open(path)):
        L.setdefault(r["setting"].strip(), {})[int(r["shell"])] = float(r["kappa"])
    return L


def read_groups(path):
    """Group of every shell from a csv.

    Parameters
    ----------
    path : str
        csv with the columns shell, group.

    Returns
    -------
    groups : dict
        shell id -> group.
    """
    return {int(r["shell"]): r["group"].strip() for r in csv.DictReader(open(path))}


def load_predictions(runs_dir, protocol="B1", families=None, pattern=PATTERN):
    """Predicted kappa of every family and seed and the released labels, by shell id.

    Parameters
    ----------
    runs_dir : str
        Directory with one folder per family and protocol.
    protocol : str, default='B1'
        Protocol whose test predictions are read.
    families : list of str, optional
        Families; default every family found.
    pattern : str, default='{family}_{protocol}_n8192'
        Name of the folder of a family and protocol.

    Returns
    -------
    pred : dict
        family -> list (one entry per seed, files in sorted order) of {shell id: predicted kappa}.
    released : dict
        shell id -> released kappa.
    """
    pred, released = {}, {}
    for fam in families or discover_families(runs_dir, protocol, pattern):
        for f in sorted(glob.glob(os.path.join(runs_dir, pattern.format(family=fam, protocol=protocol), "*_test_pred.npz"))):
            z = np.load(f); ids = z["ids"].tolist()
            pred.setdefault(fam, []).append(dict(zip(ids, z["pred_K"].tolist())))
            released = dict(zip(ids, z["true_K"].tolist()))
    return pred, released


def common_shells(labels, settings, groups=None, group=None):
    """Shells with a label under every setting, optionally of one group.

    Parameters
    ----------
    labels : dict
        Output of read_labels.
    settings : list of str
        Settings that must all be present.
    groups : dict, optional
        shell id -> group.
    group : str, optional
        Group to keep.

    Returns
    -------
    shells : list of int
        Sorted shell ids.
    """
    s = set.intersection(*[set(labels[c]) for c in settings])
    if group: s = {i for i in s if groups.get(i) == group}
    return sorted(s)


def family_error(preds, label, shells):
    """Error of a family against one set of labels.

    Parameters
    ----------
    preds : list of dict
        One {shell id: predicted kappa} per seed.
    label : dict
        shell id -> kappa.
    shells : list of int
        Shells scored.

    Returns
    -------
    error : float
        Mean over seeds of the median relative error, in percent.
    """
    return float(np.mean([100 * np.median([abs(p[s] - label[s]) / label[s] for s in shells]) for p in preds]))


def score_labels(pred, released, labels, settings, shells):
    """Errors of every family against every set of labels, with the rank correlation against the released ranking.

    Parameters
    ----------
    pred : dict
        family -> list of {shell id: predicted kappa}, one per seed.
    released : dict
        shell id -> released kappa.
    labels : dict
        setting -> {shell id: kappa}.
    settings : list of str
        Settings to score.
    shells : list of int
        Shells scored (the same for every setting).

    Returns
    -------
    out : dict
        n, released ({family: error}) and, per setting, errors, rank_rho and label_change_med_pct.
    """
    fams = sorted(pred)
    base = {f: family_error(pred[f], released, shells) for f in fams}
    out = {"n": len(shells), "released": base}
    for c in settings:
        e = {f: family_error(pred[f], labels[c], shells) for f in fams}
        rho = spearman([base[f] for f in fams], [e[f] for f in fams])
        dl = 100 * np.median([abs(labels[c][s] / released[s] - 1) for s in shells])
        out[c] = dict(errors=e, rank_rho=float(rho), label_change_med_pct=float(dl))
    return out


def main():
    """Score the model families against the alternative labels and write outputs/label_sensitivity.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="csv with the columns shell, setting, kappa")
    ap.add_argument("--runs-dir", default=RUNS + "/final_runs", help="folders <family>_<protocol>_n8192 with the test predictions")
    ap.add_argument("--pattern", default=PATTERN, help="name of the folder of a family and protocol")
    ap.add_argument("--protocol", default="B1", help="protocol whose test predictions are scored")
    ap.add_argument("--settings", default="", help="comma-separated settings, in the order printed; default every setting of --labels")
    ap.add_argument("--families", default="", help="comma-separated; default every family found")
    ap.add_argument("--groups", default="", help="csv with the columns shell, group")
    ap.add_argument("--group", default="", help="keep the shells of this group of --groups")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.group and not a.groups: ap.error("--group needs --groups")
    L = read_labels(a.labels); settings = a.settings.split(",") if a.settings else list(L)
    pred, rel = load_predictions(a.runs_dir, a.protocol, a.families.split(",") if a.families else None, a.pattern)
    if len(pred) < 2: sys.exit("fewer than two families with %s test predictions under %s" % (a.protocol, a.runs_dir))
    common = common_shells(L, settings, read_groups(a.groups) if a.groups else None, a.group)
    missing = [s for s in common if s not in rel or any(s not in p for ps in pred.values() for p in ps)]
    if missing: sys.exit("%d shells with alternative labels are not in the %s test predictions, e.g. %s" % (len(missing), a.protocol, missing[:5]))
    print("shells with every setting:", len(common))
    if not common: sys.exit("no shell to score")
    out = score_labels(pred, rel, L, settings, common); fams = sorted(pred)
    print("%-12s %s" % ("labels", " ".join("%10s" % f[:10] for f in fams)) + "   rank rho vs released   label change vs released")
    print("%-12s %s" % ("released", " ".join("%9.2f%%" % out["released"][f] for f in fams)))
    for c in settings:
        print("%-12s %s   %+.2f                 %.2f %%" % (c, " ".join("%9.2f%%" % out[c]["errors"][f] for f in fams), out[c]["rank_rho"], out[c]["label_change_med_pct"]))
    dst = a.out or "%s/label_sensitivity%s.json" % (H, "_" + a.group if a.group else "")
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    json.dump(out, open(dst, "w"), indent=1); print("wrote", dst)


if __name__ == "__main__":
    main()
