"""Tests of the BLISS scorer on synthetic shells (no data download needed)."""
import numpy as np

from bliss.metrics import score


def _shells(n=6, nodes=500, seed=0):
    """Random unit-sphere nodes and smooth displacement fields with one bump per shell.

    Parameters
    ----------
    n : int, default=6
        Number of shells.
    nodes : int, default=500
        Number of nodes.
    seed : int, default=0
        Random seed.

    Returns
    -------
    U : numpy.ndarray
        Fields, shape (n, nodes, 3).
    k : numpy.ndarray
        Knockdown factors.
    P : numpy.ndarray
        Unit node directions.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(size=(nodes, 3))
    P[:, 2] = np.abs(P[:, 2])
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    centres = P[rng.choice(nodes, n, replace=False)]
    U = np.stack([np.exp(-(1 - P @ c) * 40)[:, None] * P for c in centres]) - 0.01 * P[None]
    k = rng.uniform(0.35, 0.65, n)
    return U, k, P


def test_perfect_prediction():
    """A prediction equal to the truth scores R2 = 1, zero kappa error and zero site error."""
    U, k, P = _shells()
    m = score(U, U.copy(), k, k.copy(), P)
    assert abs(m["field_r2"] - 1) < 1e-9
    assert abs(m["buckling_r2"] - 1) < 1e-9
    assert m["kappa_rel_err"] == 0
    assert m["loc_deg_median"] == 0


def test_kappa_error_is_relative_percent():
    """A uniform 1 % bias in kappa gives a median relative error of 1 %."""
    U, k, P = _shells()
    m = score(U, U, k, 1.01 * k, P)
    assert abs(m["kappa_rel_err"] - 1.0) < 1e-6


def test_worse_prediction_scores_lower():
    """Noise in the field lowers the field R2."""
    U, k, P = _shells()
    noisy = U + 0.05 * np.random.default_rng(1).normal(size=U.shape)
    assert score(U, noisy, k, k, P)["field_r2"] < score(U, U, k, k, P)["field_r2"]
