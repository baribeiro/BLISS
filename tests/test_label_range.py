"""Tests of the label-range control on synthetic kappa (no data download needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "analyses"))
import label_range as LR   # noqa: E402


def test_kappa_range_and_counts():
    """The range is [min, max] of the training kappa, and its two ends belong to it."""
    lo, hi = LR.kappa_range([0.5, 0.4, 0.6])
    assert (lo, hi) == (0.4, 0.6)
    n = LR.range_counts([0.3, 0.4, 0.45, 0.6, 0.61, 0.9], lo, hi)
    assert n == dict(below=1, inside=3, above=2) and sum(n.values()) == 6


def test_range_errors():
    """The error is |pred / true - 1| in percent; the median is taken on all shells and on the shells in range."""
    true = np.array([0.2, 0.4, 0.5, 0.6, 0.8]); pred = true * np.array([1.10, 1.01, 0.98, 1.03, 1.20])
    r = LR.range_errors(pred, true, 0.4, 0.6)
    assert r["n_all"] == 5 and r["n_in_range"] == 3
    assert abs(r["all"] - 3.0) < 1e-9 and abs(r["in_range"] - 2.0) < 1e-9
    c = LR.range_errors(0.5, true, 0.4, 0.6)                    # a constant prediction
    assert abs(c["all"] - 100 * np.median(np.abs(0.5 / true - 1))) < 1e-12 and abs(c["in_range"] - 100 / 6) < 1e-9
    assert LR.range_errors(pred, true, 0.9, 1.0)["in_range"] is None and LR.range_errors(pred, true, 0.9, 1.0)["n_in_range"] == 0


def test_range_errors_mask_from_labels():
    """With labels, the mask follows the labels and the error the truth: two predictors get the same shells."""
    true = np.array([0.399, 0.5, 0.6]); label = np.array([0.401, 0.5, 0.6]); pred = true * np.array([1.10, 1.10, 1.05])
    r = LR.range_errors(pred, true, 0.4, 0.6, label)
    assert r["n_in_range"] == 3 and LR.range_errors(pred, true, 0.4, 0.6)["n_in_range"] == 2
    assert abs(r["in_range"] - 10.0) < 1e-9


def test_family_errors():
    """The family number is the mean over seeds of the per-seed medians; a seed without every shell is left out."""
    rng = np.random.default_rng(0); ids = list(range(10, 60)); kappa = dict(zip(ids, rng.uniform(0.3, 0.8, len(ids)).tolist()))
    preds = {s: {i: kappa[i] * (1 + 0.05 * rng.normal()) for i in ids} for s in (0, 1, 2)}
    preds[3] = {i: kappa[i] for i in ids[:-1]}                   # one shell missing
    r = LR.family_errors(preds, ids, kappa, 0.4, 0.6)
    t = np.array([kappa[i] for i in ids]); k = (t >= 0.4) & (t <= 0.6)
    e = [100 * np.abs(np.array([preds[s][i] for i in ids]) / t - 1) for s in (0, 1, 2)]
    assert r["n_seeds"] == 3 and r["seeds"] == [0, 1, 2] and r["n_all"] == 50 and r["n_in_range"] == int(k.sum())
    assert abs(r["all"] - np.mean([np.median(x) for x in e])) < 1e-12
    assert abs(r["in_range"] - np.mean([np.median(x[k]) for x in e])) < 1e-12
    assert LR.family_errors({3: preds[3]}, ids, kappa, 0.4, 0.6) is None
    truth = {s: {i: 2 * kappa[i] for i in ids} for s in (0, 1, 2)}                  # another truth for the error, same mask
    r2 = LR.family_errors({s: preds[s] for s in (0, 1, 2)}, ids, kappa, 0.4, 0.6, truth)
    assert r2["n_in_range"] == r["n_in_range"] and r2["all"] > 40


def test_prediction_files(tmp_path):
    """One file per seed is found in the folder of the B3 finals; a file without the seed tag is seed 0."""
    d = tmp_path / "final_runs" / "fam_B3_n8192"; d.mkdir(parents=True)
    for name in ("run_test_pred.npz", "run_s2_test_pred.npz", "run_s1_test_pred.npz", "run_val_pred.npz"):
        np.savez(d / name, ids=np.arange(3), pred_K=np.ones(3), true_K=np.ones(3))
    files = LR.prediction_files("fam", str(tmp_path))
    assert list(files) == [0, 1, 2] and files[2].endswith("run_s2_test_pred.npz") and files[0].endswith("run_test_pred.npz")
    assert LR.prediction_files("other", str(tmp_path)) == {}
