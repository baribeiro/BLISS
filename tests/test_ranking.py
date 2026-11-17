"""Tests of the ranking and label-sensitivity analyses on synthetic errors (no data download needed)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "analyses"))
import label_sensitivity as LS   # noqa: E402
import ranking as RK   # noqa: E402


def _errors(n_fam=5, n_seeds=3, n_shells=200, flip=False, seed=0):
    """Per-shell errors of well-separated families on two protocols, A and B.

    Parameters
    ----------
    n_fam : int, default=5
        Number of families.
    n_seeds : int, default=3
        Seeds per family.
    n_shells : int, default=200
        Shells per protocol.
    flip : bool, default=False
        Reverse the order of the families on protocol B.
    seed : int, default=0
        Random seed.

    Returns
    -------
    E : dict
        (family, protocol) -> list of arrays, as ranking.load_errors.
    fams : list of str
        Families.
    """
    rng = np.random.default_rng(seed); fams = ["f%d" % i for i in range(n_fam)]; E = {}
    for i, f in enumerate(fams):
        j = n_fam - 1 - i if flip else i
        E[(f, "A")] = [(i + 1) * 0.01 * rng.uniform(0.95, 1.05, n_shells) for _ in range(n_seeds)]
        E[(f, "B")] = [(j + 1) * 0.03 * rng.uniform(0.95, 1.05, n_shells) for _ in range(n_seeds)]
    return E, fams


def test_exact_permutation_extremes():
    """The same order gives rho = tau = 1 with p = 2 / n!; the reversed order gives -1 and a one-sided p of 1 / n!."""
    x = np.array([1.0, 2.0, 3.5, 7.0, 9.0])
    t = RK.exact_permutation(x, 10 * x)
    assert t["n_perm"] == 120 and abs(t["rho"] - 1) < 1e-12 and abs(t["tau"] - 1) < 1e-12
    assert abs(t["p_rho_two_sided"] - 2 / 120) < 1e-12 and abs(t["p_tau_two_sided"] - 2 / 120) < 1e-12
    assert t["p_rho_one_sided_negative"] == 1.0
    t = RK.exact_permutation(x, -x)
    assert abs(t["rho"] + 1) < 1e-12 and abs(t["p_rho_one_sided_negative"] - 1 / 120) < 1e-12


def test_exact_permutation_matches_scipy():
    """The closed-form rho and tau equal scipy's, and the exact p of rho equals scipy's permutation test."""
    from scipy.stats import permutation_test, spearmanr
    rng = np.random.default_rng(1); x, y = rng.normal(size=6), rng.normal(size=6)
    t = RK.exact_permutation(x, y)
    assert abs(t["rho"] - RK.spearman(x, y)) < 1e-12 and abs(t["tau"] - RK.kendall(x, y)) < 1e-12
    ref = permutation_test((y,), lambda v: spearmanr(x, v)[0], permutation_type="pairings", n_resamples=math.inf)
    assert ref.null_distribution.size == t["n_perm"] == 720
    assert abs(np.mean(np.abs(ref.null_distribution) >= abs(t["rho"]) - 1e-12) - t["p_rho_two_sided"]) < 1e-12


def test_point_errors_and_ranks():
    """The error of a family is the mean over seeds of the median over shells, in percent; rank 1 is the lowest."""
    E = {("a", "A"): [np.array([0.01, 0.02, 0.03]), np.array([0.03, 0.04, 0.05])], ("b", "A"): [np.array([0.1, 0.1, 0.1])],
         ("c", "A"): [np.array([0.001, 0.002, 0.003])]}
    pt = RK.point_errors(E, ["a", "b", "c"], ["A"])
    assert abs(pt["A"]["a"] - 3.0) < 1e-12 and abs(pt["A"]["b"] - 10.0) < 1e-12
    assert RK.rank_table(pt, ["a", "b", "c"], ["A"]) == {"a": {"A": 2.0}, "b": {"A": 3.0}, "c": {"A": 1.0}}


def test_bootstraps_on_separated_families():
    """Families far apart keep their order in every resample: the interval is [1, 1], or [-1, -1] when flipped.
    With the families resampled, only the copies of one family can swap, so rho stays positive."""
    E, fams = _errors()
    assert np.allclose(RK.shell_bootstrap(E, fams, [("A", "B")], n_boot=50)[("A", "B")], [1.0, 1.0])
    fb = RK.family_bootstrap(E, fams, "A", "B", n_boot=50)
    assert fb["n_valid"] <= 50 and fb["rho_ci"][0] > 0 and abs(fb["rho_ci"][1] - 1) < 1e-12 and fb["frac_rho_negative"] == 0
    E, fams = _errors(flip=True)
    assert np.allclose(RK.shell_bootstrap(E, fams, [("A", "B")], n_boot=50)[("A", "B")], [-1.0, -1.0])


def test_shell_bootstrap_is_seeded():
    """The same seed gives the same interval, and the interval contains the point estimate here."""
    rng = np.random.default_rng(3); fams = ["f%d" % i for i in range(6)]
    E = {(f, b): [rng.uniform(0.005, 0.03, 80) for _ in range(3)] for f in fams for b in ("A", "B")}
    a = RK.shell_bootstrap(E, fams, [("A", "B")], n_boot=100, seed=0); b = RK.shell_bootstrap(E, fams, [("A", "B")], n_boot=100, seed=0)
    assert a == b and -1 <= a[("A", "B")][0] <= a[("A", "B")][1] <= 1


def test_table_and_prediction_files(tmp_path):
    """A csv table and a folder of prediction files give the same errors and the same comparison."""
    rng = np.random.default_rng(4); fams = ["alpha", "beta_x", "gamma", "delta"]; rows = ["family,protocol,error"]
    for i, f in enumerate(fams):
        for b in ("P1", "P2"):
            for s in range(2):
                d = tmp_path / ("%s_%s_n8192" % (f, b)); d.mkdir(exist_ok=True)
                k = rng.uniform(0.3, 0.7, 50); p = k * (1 + 0.01 * (i + 1) * rng.normal(size=50))
                np.savez(d / ("run%s_test_pred.npz" % ("_s%d" % s if s else "")), ids=np.arange(50), pred_K=p, true_K=k)
                rows.append("%s,%s,%r" % (f, b, 100 * float(np.median(np.abs(p / k - 1)))))
    assert RK.discover_families(str(tmp_path), "P1") == sorted(fams)
    E = RK.load_errors(str(tmp_path), sorted(fams), ["P1", "P2"]); pt = RK.point_errors(E, sorted(fams), ["P1", "P2"])
    (tmp_path / "t.csv").write_text("\n".join(rows) + "\n"); tb = RK.read_table(str(tmp_path / "t.csv"))
    assert list(tb) == ["P1", "P2"] and all(abs(tb[b][f] - pt[b][f]) < 1e-9 for b in tb for f in fams)
    r = RK.compare_protocols(tb, sorted(fams), "P1", "P2")
    assert r["n_families"] == 4 and r["exact"]["n_perm"] == 24 and set(r["leave_one_out"]) == set(fams)
    assert abs(r["rho"] - r["exact"]["rho"]) < 1e-12


def test_label_sensitivity(tmp_path):
    """Labels equal to the released ones keep the errors and the ranking; shells missing a setting are not scored."""
    rng = np.random.default_rng(5); ids = list(range(100, 140)); rel = dict(zip(ids, rng.uniform(0.3, 0.7, len(ids)).tolist()))
    pred = {"f%d" % i: [{s: rel[s] * (1 + 0.01 * (i + 1) * rng.normal()) for s in ids} for _ in range(3)] for i in range(5)}
    rows = ["shell,setting,kappa"] + ["%d,same,%r" % (s, rel[s]) for s in ids] + ["%d,scaled,%r" % (s, 1.02 * rel[s]) for s in ids[:30]]
    (tmp_path / "l.csv").write_text("\n".join(rows) + "\n"); L = LS.read_labels(str(tmp_path / "l.csv"))
    assert list(L) == ["same", "scaled"]
    shells = LS.common_shells(L, ["same", "scaled"]); assert shells == ids[:30]
    grp = {s: "even" if s % 2 == 0 else "odd" for s in ids}
    assert LS.common_shells(L, ["same"], grp, "even") == [s for s in ids if s % 2 == 0]
    out = LS.score_labels(pred, rel, L, ["same", "scaled"], shells)
    assert out["n"] == 30 and abs(out["same"]["rank_rho"] - 1) < 1e-12 and out["same"]["label_change_med_pct"] == 0.0
    assert out["same"]["errors"] == out["released"]
    assert abs(out["scaled"]["label_change_med_pct"] - 2.0) < 1e-9
    e = out["released"]["f0"]
    assert abs(e - np.mean([100 * np.median([abs(p[s] / rel[s] - 1) for s in shells]) for p in pred["f0"]])) < 1e-9
