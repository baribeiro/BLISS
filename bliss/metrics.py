"""The benchmark metrics. One function, one dict; every baseline reports these and nothing else.

    from bliss.metrics import score
    score(U_true, U_pred, k_true, k_pred, P)   # U (S,N,3), k (S,), P (N,3) unit-sphere positions

  kappa_rel_err   median |k_pred - k_true| / k_true, in percent. The engineering quantity.
  kappa_r2        R^2 of the knockdown factor over the split
  field_r2        R^2 of U over all nodes and components against the SCALAR mean of the evaluated set.
                  Reported for continuity with the literature; it is the least informative number here,
                  because the pre-buckling contraction is common to every shell and the training-set
                  per-node mean field alone already scores well on it (the reference rows of the paper,
                  benchmark/references/site_baselines.py, give the value per protocol). Read buckling_r2 instead.
  dimple_r2       the same, restricted to nodes where |U_true| > 10 % of that shell's max |U|.
  buckling_r2     R^2 after removing each shell's uniform radial contraction u_bar P, u_bar = the node mean
                  of U.P (the pre-buckling membrane state, set by kappa alone; about half of the variance
                  of U at the limit point, the exact share per protocol is a reference row). What is left
                  is the buckling deviation D, and the trivial mean field scores low here.
  buckling_dimple_r2  the same, on the dimple region.
  loc_deg         angular distance, in degrees, between argmax|D_pred| and argmax|D_true|, D = U minus that
                  shell's uniform radial contraction (the buckling deviation; argmax |D|, not |U|), per shell: median, and fraction within 10 deg. Null references (uniform site on the cap, a random
                  defect centre of the shell, the mean field) are reference rows of the paper, not constants here.
  loc_by_r        the same, binned by r = d2/d1 of the shell (needs r); the difficulty axis.
  loc_dominant    localisation restricted to shells whose TRUE field has one clear dimple (q < 0.7, where
                  q = second dimple / first dimple of the FEM field, metadata/truth_dominance.csv). At a
                  near-tie the FEM field itself is bimodal and its argmax is fragile (it can flip with the
                  mesh), so the headline localisation number is this one; the full-set number is reported
                  alongside. Needs q.
  ambiguity_err   median |q_pred - q_true| over shells with q_true > 0.9: does the model reproduce the
                  two competing dimples with the right amplitude ratio, even when it cannot know which wins.
  same_winner     fraction of q_true > 0.9 shells where the model's GLOBAL argmax is within 10 deg of the
                  solver's. This conflates two failures and should not be read alone: see below.
  two_way_winner  the well-posed question on an ambiguous shell. Take the solver's two competing sites,
                  read the model's predicted amplitude AT those two sites, and ask whether the solver's
                  winner is the larger. Chance is 0.5. This is the headline number for the ambiguous
                  subset, because it does not punish a model whose global peak landed elsewhere.
  peak_elsewhere  fraction of ambiguous shells where the model's peak is near neither candidate: a
                  measure of field accuracy, not of the tie, and the reason same_winner looks like chance.
  resolvable_margin  NOT returned by score(), which sees one split at a time. m* : with m = 1 - q the distance of the true field from a tie, the margin at which
                  the model's winner accuracy first reaches halfway between chance (0.5) and its ceiling
                  on well-separated shells (m > 0.3). A property of the model, defined for any argmax
                  target, in the units of the problem.

  Uncertainty. Every number above is a point estimate on a finite test set. bootstrap_ci() puts a
  percentile confidence interval on any of them by resampling SHELLS, score(..., per_shell=True)
  hands it the per-shell rows, and score_ci() does both for a predictions file and returns every
  headline metric with its 95 % interval. The interval covers the finite test set only: it is not a
  seed interval. To compare two models use compare(), a paired bootstrap of the metric differences on
  the same shells.

Functions
---------
bootstrap_ci
    Percentile bootstrap confidence interval of a benchmark metric. The resampling unit is the SHELL.
loc_error_deg
    Angle between the predicted and the true site of largest displacement, per shell.
dominance
    q = peak2/peak1 of |D| per shell, from the same two-peak rule the ambiguity metrics use.
score
    The benchmark metrics of one split. Returns the dict the module docstring describes.
fmt
    One-line text summary of a metrics dictionary.
released_q
    The released dominance ratio q of each shell id (metadata/truth_dominance.csv, computed on the
score_ci
    Every headline metric of one predictions artefact, with a percentile bootstrap CI over shells.
compare
    Paired shell bootstrap of the metric differences A - B on the SAME test shells.
compare_seeds
    Seed-aware paired comparison (J.4): A and B are lists of artefacts (one per training seed, e.g. 3
    each).
standard_nodes
    The fixed 8,192-node subset every field metric is computed on (sorted default_rng(0) draw).
release_truth
    Ground truth for the given shell ids, read from the release (canonical/<id>.npz): U_peak at `nodes`
    (default the
score_submission
    Score a SUBMISSION: only ids, pred_U (S, N, 3) and pred_K (S,) are read from it; the ground truth is
    loaded
score_by_regime
    score_ci on the pooled test set and on its parts: sparse (ids < 4758, beta_min = 25 deg), dense (ids
    >= 4758,
twins
    {shell: twin} for every shell that has one; empty if the file is absent.
leaked_test_shells
    Test shells of `split` whose twin sits in that split's train or val set. These are dropped at
    scoring.
excluded_test_shells
    Test shells to leave out of every number of `split`: (a) those whose twin is in train or val (a
    leak), and
excluded_test_counts
    Number of test shells left out of a split, by reason.
"""
import os

import numpy as np


def _r2(a, b, ref=None):
    """R^2 of b against a; `ref` is the reference prediction the variance is measured from
    (default: the scalar mean of a). Both flattened, so a boolean mask can be applied beforehand.

    Parameters
    ----------
    a : numpy.ndarray
        True values.
    b : numpy.ndarray
        Predicted values, same shape as a.
    ref : float or numpy.ndarray, default=None
        Reference prediction of the variance; the scalar mean of a if None.

    Returns
    -------
    r2 : float
        Coefficient of determination.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ref = a.mean() if ref is None else np.asarray(ref, np.float64)
    return 1.0 - ((a - b) ** 2).sum() / max(((a - ref) ** 2).sum(), 1e-30)


def _r2_parts(a, b, ref, mask=None):
    """The same R^2 as `_r2`, split per shell: row i is (SSE_i, SST_i) of shell i, against the fixed
    reference `ref`, so that 1 - SSE.sum()/SST.sum() over any set of shells is the R^2 of that set.

    `ref` is held at its full-sample value (the scalar mean of the evaluated set, which is how the
    metric defines it) rather than recomputed inside each bootstrap resample; that keeps the point
    estimate identical to score()'s and makes the interval one about the prediction, not about the
    reference. `mask` is the per-shell boolean node mask of the dimple-region metrics.

    Parameters
    ----------
    a : numpy.ndarray, shape (S, ...)
        True values per shell.
    b : numpy.ndarray, shape (S, ...)
        Predicted values per shell.
    ref : float or numpy.ndarray
        Reference prediction, held at its full-sample value.
    mask : numpy.ndarray, default=None
        Boolean mask of the entries to keep.

    Returns
    -------
    parts : numpy.ndarray, shape (S, 2)
        Row i is (SSE_i, SST_i) of shell i.
    """
    a = np.asarray(a); b = np.asarray(b); out = np.zeros((len(a), 2))
    for i in range(len(a)):
        x, y = np.asarray(a[i], np.float64), np.asarray(b[i], np.float64)
        if mask is not None: x, y = x[mask[i]], y[mask[i]]
        out[i] = ((x - y) ** 2).sum(), ((x - ref) ** 2).sum()
    return out


def _stat_mean(v): 
    """Mean of the per-shell values, along the last axis.

    Parameters
    ----------
    v : numpy.ndarray, shape (..., S, 1)
        Per-shell values.

    Returns
    -------
    mean : numpy.ndarray
        Mean over shells.
    """
    return v[..., 0].mean(-1)


def _stat_median(v): 
    """Median of the per-shell values, along the last axis.

    Parameters
    ----------
    v : numpy.ndarray, shape (..., S, 1)
        Per-shell values.

    Returns
    -------
    median : numpy.ndarray
        Median over shells.
    """
    return np.median(v[..., 0], -1)


def _stat_r2(v): 
    """R^2 from per-shell (SSE, SST) rows, pooled over shells.

    Parameters
    ----------
    v : numpy.ndarray, shape (..., S, 2)
        Per-shell sums of squared errors and total sums of squares.

    Returns
    -------
    r2 : numpy.ndarray
        1 - sum(SSE) / sum(SST).
    """
    return 1.0 - v[..., 0].sum(-1) / np.maximum(v[..., 1].sum(-1), 1e-30)


_STAT = {"mean": _stat_mean, "median": _stat_median, "r2": _stat_r2}


def bootstrap_ci(per_shell_values, n_boot=2000, alpha=0.05, seed=0, statistic="mean"):
    """Percentile bootstrap confidence interval of a benchmark metric. The resampling unit is the SHELL.

    One shell is one independent draw from the population this benchmark samples: its geometry, its
    solve and its label are generated independently of every other shell, while the 8192 nodes inside
    one shell are not independent of each other at all. So the interval is built by resampling shell
    indices with replacement, S of them from the S shells that were scored, and recomputing the whole
    statistic on each resample; nodes are never resampled and never enter the count. The interval is
    therefore about the finite test set, i.e. how much of the reported number is the luck of which
    shells fell in this split, and it says nothing about seed or training variance.

    per_shell_values  one row per shell, as produced by score(..., per_shell=True):
                      a 1-D array for the "mean" and "median" statistics (per-shell relative error,
                      per-shell localisation angle, per-shell 0/1 hit), or an (S, 2) array of the
                      per-shell (SSE, SST) for "r2", which is a ratio of sums over shells and not a
                      mean of per-shell values, so it has to be resampled as the pair.
    statistic         "mean" (accuracies, hit rates), "median" (median relative error, median angle),
                      "r2" (any of the R^2 metrics), or a callable f(values) -> float applied to the
                      resampled rows. Rows with a non-finite entry are dropped before resampling.
    n_boot            resamples (2000 is enough for a 95 % percentile interval).
    seed              seeds a numpy Generator explicitly, so the interval is reproducible.

    Returns (point_estimate, lo, hi): the statistic on the observed shells, and the alpha/2 and
    1-alpha/2 percentiles of the resampled statistic. (nan, nan, nan) if no shell survives, and
    (point, nan, nan) for a single shell, where a bootstrap says nothing.

    Parameters
    ----------
    per_shell_values : numpy.ndarray
        One row per shell, as returned by score(..., per_shell=True).
    n_boot : int, default=2000
        Number of bootstrap resamples.
    alpha : float, default=0.05
        Two-sided level; 0.05 gives a 95 % interval.
    seed : int, default=0
        Seed of the resampling.
    statistic : str or callable, default='mean'
        'mean', 'median', 'r2' or a function of the resampled rows.

    Returns
    -------
    ci : dict
        point, lo, hi, half_width, n_shells and n_boot.
    """
    v = np.asarray(per_shell_values, np.float64)
    if v.ndim == 1: v = v[:, None]
    v = v[np.isfinite(v).all(1)]
    s = len(v)
    if s == 0: return float("nan"), float("nan"), float("nan")
    if callable(statistic):
        f = lambda x: np.array([statistic(x[j]) for j in range(len(x))])
    else:
        f = _STAT[statistic]
    point = float(f(v[None])[0])
    if s < 2: return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot); done = 0
    step = max(1, int(2e7 // s))                      # bound the (step, s, c) index buffer
    while done < n_boot:
        b = min(step, n_boot - done)
        draws[done:done + b] = f(v[rng.integers(0, s, size=(b, s))]); done += b
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def loc_error_deg(U_true, U_pred, P):
    """Angle between the predicted and the true site of largest displacement, per shell.

    Parameters
    ----------
    U_true : numpy.ndarray, shape (S, N, 3)
        True displacement fields.
    U_pred : numpy.ndarray, shape (S, N, 3)
        Predicted displacement fields.
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes.

    Returns
    -------
    angle : numpy.ndarray, shape (S,)
        Angle in degrees between argmax |U_true| and argmax |U_pred|.
    """
    it = np.linalg.norm(U_true, axis=-1).argmax(1); ip = np.linalg.norm(U_pred, axis=-1).argmax(1)
    c = np.clip((P[it] * P[ip]).sum(1), -1, 1)
    return np.degrees(np.arccos(c))


def _ring(P, k=12):
    """The k nearest neighbours of every node, used to find local maxima.

    Parameters
    ----------
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes.
    k : int, default=12
        Number of neighbours.

    Returns
    -------
    nb : numpy.ndarray, shape (N, k)
        Indices of the neighbours of each node.
    """
    from scipy.spatial import cKDTree
    return cKDTree(P).query(P, k=k + 1)[1][:, 1:]


def _two_peaks(D, P, nb, min_sep_deg=8.0):
    """The two competing sites of one field: its argmax and the largest local maximum min_sep_deg away.

    Parameters
    ----------
    D : numpy.ndarray, shape (N,)
        Magnitude of the buckling deviation of one shell.
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes.
    nb : numpy.ndarray, shape (N, k)
        Neighbour indices from _ring.
    min_sep_deg : float, default=8.0
        Minimum angle between the two sites.

    Returns
    -------
    i1, i2 : int
        Node indices of the first and second site (i2 is -1 when there is none).
    """
    a = np.linalg.norm(D, axis=1); loc = np.where(a >= a[nb].max(1))[0]; loc = loc[np.argsort(-a[loc])]
    i1 = loc[0]
    for j in loc[1:]:
        if np.degrees(np.arccos(np.clip(P[j] @ P[i1], -1, 1))) > min_sep_deg: return i1, j
    return i1, None


def dominance(D, P, k=12, min_sep_deg=8.0):
    """q = peak2/peak1 of |D| per shell, from the same two-peak rule the ambiguity metrics use.

    NOTE: on the 8192-node subset this reproduces the released metadata/truth_dominance.csv, which is
    computed on the full 76805-node mesh, to about 2 % of the ambiguous-subset membership. Use the
    released column when you want the canonical partition; this function is for arbitrary node sets.

    Parameters
    ----------
    D : numpy.ndarray, shape (S, N)
        Magnitude of the buckling deviation per shell.
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes.
    k : int, default=12
        Neighbours used for the local maxima.
    min_sep_deg : float, default=8.0
        Minimum angle between the two sites.

    Returns
    -------
    q : numpy.ndarray, shape (S,)
        Ratio of the second to the first site amplitude.
    """
    nb = _ring(P, k); out = []
    for i in range(D.shape[0]):
        a = np.linalg.norm(D[i], axis=1); i1, i2 = _two_peaks(D[i], P, nb, min_sep_deg)
        out.append(0.0 if i2 is None else a[i2] / a[i1])
    return np.array(out)


def score(U_true, U_pred, k_true, k_pred, P, r=None, q=None, per_shell=False):
    """The benchmark metrics of one split. Returns the dict the module docstring describes.

    per_shell=False (default) returns that dict and nothing else: the return shape and every number
    are exactly what they were before this argument existed. per_shell=True returns (dict, parts),
    where parts maps each headline metric to (kind, values) with one row of `values` per SHELL, so
    that a shell-level bootstrap can resample it (see bootstrap_ci). kind is "median", "mean" or
    "r2"; an "r2" row is the pair (SSE, SST) of that shell, and 1 - SSE.sum()/SST.sum() reproduces
    the pooled R^2 in the dict. Subset metrics carry only the shells of their subset, so resampling
    them is conditional on the subset, i.e. the interval is for the metric as defined and reported.

    Parameters
    ----------
    U_true : numpy.ndarray, shape (S, N, 3)
        True displacement fields (mm).
    U_pred : numpy.ndarray, shape (S, N, 3)
        Predicted displacement fields (mm).
    k_true : numpy.ndarray, shape (S,)
        True knockdown factors.
    k_pred : numpy.ndarray, shape (S,)
        Predicted knockdown factors.
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes.
    r : numpy.ndarray, shape (S,), default=None
        Depth ratio of the two deepest defects, for loc_by_r.
    q : numpy.ndarray, shape (S,), default=None
        Released site ratio; computed from U_true if None.
    per_shell : bool, default=False
        If True, also return the per-shell rows for the bootstrap.

    Returns
    -------
    metrics : dict
        Every metric of the module docstring (and the per-shell parts if per_shell is True).
    """
    U_true, U_pred = np.asarray(U_true, np.float32), np.asarray(U_pred, np.float32)
    k_true, k_pred = np.asarray(k_true, np.float64), np.asarray(k_pred, np.float64)
    mag = np.linalg.norm(U_true, axis=-1)
    m = mag > 0.1 * mag.max(1, keepdims=True)
    ur_t = (U_true * P).sum(-1).mean(1); ur_p = (U_pred * P).sum(-1).mean(1)
    Dt = U_true - ur_t[:, None, None] * P; Dp = U_pred - ur_p[:, None, None] * P
    loc = loc_error_deg(Dt, Dp, P)          # on the buckling deviation, as Sec. 3 defines it
    md = np.linalg.norm(Dt, axis=-1); md = md > 0.1 * md.max(1, keepdims=True)
    out = dict(buckling_r2=_r2(Dt, Dp), buckling_dimple_r2=_r2(Dt[md], Dp[md]),kappa_rel_err=float(np.median(np.abs(k_pred - k_true) / k_true) * 100),
               kappa_r2=_r2(k_true, k_pred), field_r2=_r2(U_true, U_pred),
               dimple_r2=_r2(U_true[m], U_pred[m]), loc_deg_median=float(np.median(loc)),
               loc_within_10=float((loc < 10).mean()), n=int(len(k_true)))
    if q is None: q = dominance(Dt, P)
    q = np.asarray(q); dom, amb = q < 0.7, q > 0.9
    tw, elsewhere, tw_shell = [], [], []
    out["loc_dominant_median"] = float(np.median(loc[dom])) if dom.any() else np.nan
    out["loc_dominant_within_10"] = float((loc[dom] < 10).mean()) if dom.any() else np.nan
    out["n_dominant"] = int(dom.sum()); out["n_ambiguous"] = int(amb.sum())
    if amb.any():
        qp = dominance(Dp, P)
        out["ambiguity_err"] = float(np.median(np.abs(qp - q)[amb])); out["same_winner"] = float((loc[amb] < 10).mean())
        tw, elsewhere = [], []
        nb = _ring(P)
        for k in np.where(amb)[0]:
            t1, t2 = _two_peaks(Dt[k], P, nb)
            if t2 is None: continue
            p1, p2 = _two_peaks(Dp[k], P, nb)
            tw_shell.append(int(k))
            tw.append(np.linalg.norm(Dp[k][t1]) > np.linalg.norm(Dp[k][t2]))
            near = lambda a, b: np.degrees(np.arccos(np.clip(P[a] @ P[b], -1, 1))) < 10
            elsewhere.append(not (near(t1, p1) or (p2 is not None and near(t1, p2)) or near(t2, p1) or (p2 is not None and near(t2, p2))))
        if tw:
            out["two_way_winner"] = float(np.mean(tw)); out["peak_elsewhere"] = float(np.mean(elsewhere)); out["n_two_way"] = len(tw)
    if r is not None:
        r = np.asarray(r); out["loc_by_r"] = {}
        for lo, hi in ((0, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)):
            s = (r >= lo) & (r < hi)
            if s.sum():
                out["loc_by_r"]["%.2f-%.2f" % (lo, hi)] = dict(n=int(s.sum()), median_deg=float(np.median(loc[s])),
                                                              within_10=float((loc[s] < 10).mean()))
    if not per_shell: return out
    parts = {"kappa_rel_err": ("median", np.abs(k_pred - k_true) / k_true * 100),
             "kappa_r2": ("r2", _r2_parts(k_true, k_pred, k_true.mean())),
             "field_r2": ("r2", _r2_parts(U_true, U_pred, U_true.mean(dtype=np.float64))),
             "dimple_r2": ("r2", _r2_parts(U_true, U_pred, U_true[m].mean(dtype=np.float64), m)),
             "buckling_r2": ("r2", _r2_parts(Dt, Dp, Dt.mean(dtype=np.float64))),
             "buckling_dimple_r2": ("r2", _r2_parts(Dt, Dp, Dt[md].mean(dtype=np.float64), md)),
             "loc_deg_median": ("median", loc), "loc_within_10": ("mean", (loc < 10).astype(np.float64))}
    if dom.any():
        parts["loc_dominant_median"] = ("median", loc[dom])
        parts["loc_dominant_within_10"] = ("mean", (loc[dom] < 10).astype(np.float64))
    if "ambiguity_err" in out:
        parts["ambiguity_err"] = ("median", np.abs(qp - q)[amb])
        parts["same_winner"] = ("mean", (loc[amb] < 10).astype(np.float64))
    if tw:
        parts["two_way_winner"] = ("mean", np.asarray(tw, np.float64))
        parts["peak_elsewhere"] = ("mean", np.asarray(elsewhere, np.float64))
    return out, parts


def fmt(sc):
    """One-line text summary of a metrics dictionary.

    Parameters
    ----------
    sc : dict
        Output of score().

    Returns
    -------
    text : str
        The headline metrics on one line.
    """
    s = "kappa err %.2f %%  R2 kappa %.3f | field %.3f  dimple %.3f | buckling %.3f  dimple %.3f | loc %.1f deg, %.0f %% <10 | dominant (n=%d): %.0f %% <10 | ambiguous (n=%d): |dq| %.2f, same winner %.0f %%" % (
        sc["kappa_rel_err"], sc["kappa_r2"], sc["field_r2"], sc["dimple_r2"], sc["buckling_r2"], sc["buckling_dimple_r2"], sc["loc_deg_median"], 100 * sc["loc_within_10"],
        sc["n_dominant"], 100 * sc.get("loc_dominant_within_10", np.nan), sc["n_ambiguous"], sc.get("ambiguity_err", np.nan), 100 * sc.get("same_winner", np.nan)) + \
        ("\n    ambiguous subset: two-way winner %.0f %% (n=%d), model peak on neither candidate %.0f %%" % (
            100 * sc["two_way_winner"], sc["n_two_way"], 100 * sc["peak_elsewhere"]) if "two_way_winner" in sc else "")
    for k, v in sc.get("loc_by_r", {}).items():
        s += "\n    r %s  n %4d  median %5.1f deg  %.0f %% <10" % (k, v["n"], v["median_deg"], 100 * v["within_10"])
    return s


HEADLINE = ("kappa_rel_err", "kappa_r2", "field_r2", "dimple_r2", "buckling_r2", "buckling_dimple_r2",
            "loc_deg_median", "loc_within_10", "loc_dominant_median", "loc_dominant_within_10",
            "ambiguity_err", "same_winner", "two_way_winner", "peak_elsewhere")
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# metadata/truth_dominance.csv of the repository if linked there, else the one of the release (BLISS_RELEASE)
TRUTH_Q = os.path.join(_REPO, "metadata", "truth_dominance.csv")
if not os.path.exists(TRUTH_Q):
    TRUTH_Q = os.path.join(os.environ.get("BLISS_RELEASE", os.path.join(_REPO, "BLISS-1.0")), "metadata", "truth_dominance.csv")


def released_q(ids, path=TRUTH_Q):
    """The released dominance ratio q of each shell id (metadata/truth_dominance.csv, computed on the
    full mesh), or None if the file or any of the ids is missing. This is the canonical partition into
    the dominant (q<0.7) and ambiguous (q>0.9) subsets; dominance() recomputes it on whatever node set
    it is given and agrees to about 2 % of the membership.

    Parameters
    ----------
    ids : sequence of int
        Shell identifiers.
    path : str, default=TRUTH_Q
        Path of truth_dominance.csv.

    Returns
    -------
    q : numpy.ndarray or None
        Released site ratio of each shell, or None if unavailable.
    """
    try:
        with open(path) as fh:
            next(fh); d = {int(l.split(",")[0]): float(l.split(",")[1]) for l in fh if l.strip()}
    except (OSError, StopIteration):
        return None
    ids = [int(i) for i in ids]
    return np.array([d[i] for i in ids]) if all(i in d for i in ids) else None


def _load_pred(pred):
    """Open a prediction file, or accept an already opened mapping.

    Parameters
    ----------
    pred : str or mapping
        Path of a .npz prediction file, or a dict-like object with its arrays.

    Returns
    -------
    z : mapping
        The arrays.
    has : callable
        has(key) is True when the key is present.
    """
    z = np.load(pred) if isinstance(pred, str) else pred
    return z, (lambda k: k in (z.files if hasattr(z, "files") else z))


def _prepare(pred, q="auto", r=None, split=None, drop_leaked=False, subset=None):
    """The arrays one scoring call uses, after the released-q lookup, the optional legacy twin rule and an
    optional subset (boolean mask over the file's shells, or an iterable of shell ids).

    Parameters
    ----------
    pred : str or mapping
        Prediction file.
    q : str or numpy.ndarray
        'auto' reads the released q; an array is used as given.
    r : numpy.ndarray, default=None
        Depth ratios.
    split : str, default=None
        Protocol name, for the legacy twin rule.
    drop_leaked : bool
        Apply the legacy twin rule.
    subset : numpy.ndarray or iterable, default=None
        Boolean mask over the shells of the file, or shell ids.

    Returns
    -------
    arrays : tuple
        ids, U_true, U_pred, k_true, k_pred, P, r, q after the selection.
    """
    z, has = _load_pred(pred)
    P = np.asarray(z["P"], np.float64); P = P / np.linalg.norm(P, axis=-1, keepdims=True)
    if r is None and has("r"): r = z["r"]
    ids = np.asarray(z["ids"]).astype(int) if has("ids") else None
    q_source = "given by the caller"
    if isinstance(q, str) and q == "auto":
        if ids is None:
            raise ValueError("score_ci needs the shell ids to use the released dominance q (pass q explicitly otherwise)")
        q = released_q(ids)
        if q is None: raise ValueError("some ids are not in %s" % TRUTH_Q)
        q_source = TRUTH_Q
    keep = np.ones(len(z["true_K"]), bool); n_dropped = 0
    if drop_leaked:
        if split is None or ids is None: raise ValueError("drop_leaked=True needs split and ids")
        bad = excluded_test_shells(split); keep &= np.array([i not in bad for i in ids.tolist()]); n_dropped = int((~keep).sum())
    if subset is not None:
        sub = np.asarray(subset)
        keep &= sub.astype(bool) if sub.dtype == bool else np.isin(ids, sub.astype(int))
    pick = lambda a: None if a is None else np.asarray(a)[keep]
    return dict(U_t=np.asarray(z["true_U"])[keep], U_p=np.asarray(z["pred_U"])[keep], K_t=pick(z["true_K"]), K_p=pick(z["pred_K"]),
                ids=pick(ids), r=pick(r), q=pick(q) if not isinstance(q, str) and q is not None else None, P=P,
                q_source=q_source, n_dropped=n_dropped)


def score_ci(pred, n_boot=2000, alpha=0.05, seed=0, q="auto", r=None, split=None, drop_leaked=False, subset=None):
    """Every headline metric of one predictions artefact, with a percentile bootstrap CI over shells.

        from bliss.metrics import score_ci
        score_ci("pred.npz", split="B1")["metrics"]["kappa_rel_err"]
        -> {"point": .., "lo": .., "hi": .., "half_width": .., "n_shells": .., "n_boot": ..}

    `pred` is the path of a saved prediction file, or anything with the keys a saved one has: ids, true_U,
    pred_U (S,N,3), true_K, pred_K (S,), P (N,3), and optionally r. The ids are required with q="auto", so the
    dominant/ambiguous partition is always the released one (metadata/truth_dominance.csv). `split` ("B1",
    "B2", "B3") is recorded in the output; the released splits need no twin rule, so drop_leaked is False by
    default (True applies the legacy rule of a named split). `subset` restricts the scoring to some shells
    (boolean mask or ids): the R^2 references are then that subset's own scalar means. The point estimates
    are score()'s own numbers; the interval measures the finite test set, not seed or training variance.

    Parameters
    ----------
    pred : str or mapping
        Prediction file with ids, pred_U, pred_K, true_U, true_K and P.
    n_boot : int, default=2000
        Bootstrap resamples.
    alpha : float, default=0.05
        Two-sided level.
    seed : int, default=0
        Seed of the resampling.
    q : str or numpy.ndarray, default='auto'
        Site ratio source.
    r : numpy.ndarray, default=None
        Depth ratios.
    split : str, default=None
        Protocol name.
    drop_leaked : bool, default=False
        Legacy twin rule.
    subset : numpy.ndarray or iterable, default=None
        Shells to score.

    Returns
    -------
    result : dict
        'metrics' maps every headline metric to point, lo, hi, half_width, n_shells and n_boot.
    """
    src = os.path.abspath(pred) if isinstance(pred, str) else None
    d = _prepare(pred, q, r, split, drop_leaked, subset)
    sc, parts = score(d["U_t"], d["U_p"], d["K_t"], d["K_p"], d["P"], d["r"], d["q"], per_shell=True)
    out = dict(source=src, split=split, n_shells=sc["n"], n_dropped_twins=d["n_dropped"], n_dominant=sc["n_dominant"],
               n_ambiguous=sc["n_ambiguous"], n_two_way=sc.get("n_two_way"), q_source=d["q_source"], n_boot=n_boot,
               alpha=alpha, seed=seed, subset=None if subset is None else "restricted", metrics={})
    for name in HEADLINE:                     # one seed for the whole artefact, so metrics scored on the
        if name not in parts: continue        # same shells see the same resamples and stay coherent
        kind, vals = parts[name]
        pt, lo, hi = bootstrap_ci(vals, n_boot=n_boot, alpha=alpha, seed=seed, statistic=kind)
        out["metrics"][name] = dict(point=float(sc[name]), lo=lo, hi=hi, half_width=0.5 * (hi - lo),
                                    n_shells=int(np.isfinite(np.asarray(vals, np.float64)).all(-1).sum())
                                    if np.ndim(vals) > 1 else int(np.isfinite(vals).sum()),
                                    n_boot=n_boot, statistic=kind, point_from_parts=pt)
    out["score"] = sc
    return out


def compare(predA, predB, n_boot=2000, alpha=0.05, seed=0, q="auto", subset=None):
    """Paired shell bootstrap of the metric differences A - B on the SAME test shells.

    Both artefacts are aligned on their common ids. Each resample draws one set of shell indices and applies
    it to both models, so the interval is for the difference itself (the shells' common difficulty cancels).
    Returns {metric: {"diff": A-B, "lo", "hi", "p_a_worse_or_equal"}} where the last entry is the fraction of
    resamples in which A does not beat B (orientation: higher is better except kappa_rel_err, loc medians,
    ambiguity_err and peak_elsewhere, for which lower is better).

    Parameters
    ----------
    predA, predB : str or mapping
        The two prediction files.
    n_boot : int, default=2000
        Bootstrap resamples.
    alpha : float, default=0.05
        Two-sided level.
    seed : int, default=0
        Seed of the resampling.
    q : str or numpy.ndarray, default='auto'
        Site ratio source.
    subset : numpy.ndarray or iterable, default=None
        Shells to compare.

    Returns
    -------
    result : dict
        Per metric, the difference A - B with its interval and the share of resamples where A does
        not beat B.
    """
    za, _ = _load_pred(predA); zb, _ = _load_pred(predB)
    ia, ib = np.asarray(za["ids"]).astype(int), np.asarray(zb["ids"]).astype(int)
    common = np.intersect1d(ia, ib)
    if subset is not None:
        s_ = np.asarray(subset); common = common[np.isin(common, s_ if s_.dtype != bool else ia[s_])]
    da = _prepare(predA, q, subset=common); db = _prepare(predB, q, subset=common)
    oa, ob = np.argsort(da["ids"]), np.argsort(db["ids"])
    for d_, o_ in ((da, oa), (db, ob)):
        for k in ("U_t", "U_p", "K_t", "K_p", "ids", "r", "q"):
            if d_[k] is not None: d_[k] = d_[k][o_]
    assert np.array_equal(da["ids"], db["ids"]) and np.allclose(da["K_t"], db["K_t"]), "the two artefacts do not score the same truth"
    _, pa = score(da["U_t"], da["U_p"], da["K_t"], da["K_p"], da["P"], da["r"], da["q"], per_shell=True)
    _, pb = score(db["U_t"], db["U_p"], db["K_t"], db["K_p"], db["P"], db["r"], db["q"], per_shell=True)
    lower = {"kappa_rel_err", "loc_deg_median", "loc_dominant_median", "ambiguity_err", "peak_elsewhere"}
    rng = np.random.default_rng(seed); out = dict(n_common=int(len(common)), n_boot=n_boot, metrics={})
    for name in HEADLINE:
        if name not in pa or name not in pb: continue
        kind, va = pa[name]; _, vb = pb[name]
        va, vb = np.asarray(va, np.float64), np.asarray(vb, np.float64)
        if va.shape != vb.shape: continue          # a subset metric whose membership differs (cannot happen with a shared q)
        if va.ndim == 1: va, vb = va[:, None], vb[:, None]
        ok = np.isfinite(va).all(1) & np.isfinite(vb).all(1); va, vb = va[ok], vb[ok]; s = len(va)
        if s < 2: continue
        f = _STAT[kind]; pt = float(f(va[None])[0] - f(vb[None])[0])
        draws = np.empty(n_boot); done = 0; step = max(1, int(2e7 // (s * va.shape[1])))
        while done < n_boot:
            b = min(step, n_boot - done); idx = rng.integers(0, s, size=(b, s))
            draws[done:done + b] = f(va[idx]) - f(vb[idx]); done += b
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        worse = (draws >= 0).mean() if name in lower else (draws <= 0).mean()
        out["metrics"][name] = dict(diff=pt, lo=float(lo), hi=float(hi), p_a_worse_or_equal=float(worse), n_shells=int(s), statistic=kind)
    return out


def compare_seeds(predsA, predsB, n_boot=2000, alpha=0.05, seed=0, q="auto", subset=None):
    """Seed-aware paired comparison (J.4): A and B are lists of artefacts (one per training seed, e.g. 3 each).

    Each resample draws ONE set of shell indices; on it, the metric of each model is computed per seed and averaged
    over that model's seeds, and the difference of the two averages is recorded. So the interval covers the finite
    test set with the seeds' mean as the model's prediction quality; the per-seed differences are reported beside.
    "tail_fraction" = share of resamples in which A does not beat B (a bootstrap tail fraction, not a p-value);
    descriptive, no multiplicity correction. All artefacts are aligned on the ids common to all of them.

    Parameters
    ----------
    predsA, predsB : list
        Prediction files of each model, one per training seed.
    n_boot : int, default=2000
        Bootstrap resamples.
    alpha : float, default=0.05
        Two-sided level.
    seed : int, default=0
        Seed of the resampling.
    q : str or numpy.ndarray, default='auto'
        Site ratio source.
    subset : numpy.ndarray or iterable, default=None
        Shells to compare.

    Returns
    -------
    result : dict
        Per metric, the seed-averaged difference A - B with its interval and the per-seed
        differences.
    """
    arts = [_load_pred(x)[0] for x in list(predsA) + list(predsB)]
    common = arts[0]["ids"].astype(int)
    for z in arts[1:]: common = np.intersect1d(common, np.asarray(z["ids"]).astype(int))
    if subset is not None:
        s_ = np.asarray(subset); common = common[np.isin(common, s_)] if s_.dtype != bool else common
    parts = []
    for x in list(predsA) + list(predsB):
        d = _prepare(x, q, subset=common); o = np.argsort(d["ids"])
        for k in ("U_t", "U_p", "K_t", "K_p", "ids", "r", "q"):
            if d[k] is not None: d[k] = d[k][o]
        parts.append(score(d["U_t"], d["U_p"], d["K_t"], d["K_p"], d["P"], d["r"], d["q"], per_shell=True)[1])
    nA = len(predsA); lower = {"kappa_rel_err", "loc_deg_median", "loc_dominant_median", "ambiguity_err", "peak_elsewhere"}
    rng = np.random.default_rng(seed); out = dict(n_common=int(len(common)), n_seeds=[nA, len(predsB)], n_boot=n_boot, metrics={})
    for name in HEADLINE:
        if not all(name in p_ for p_ in parts): continue
        kind = parts[0][name][0]; V = [np.asarray(p_[name][1], np.float64) for p_ in parts]
        V = [v[:, None] if v.ndim == 1 else v for v in V]
        if len({v.shape for v in V}) != 1: continue
        ok = np.all([np.isfinite(v).all(1) for v in V], 0); V = [v[ok] for v in V]; s = len(V[0])
        if s < 2: continue
        f = _STAT[kind]
        stat = lambda idx: np.mean([f(v[idx]) for v in V[:nA]], 0) - np.mean([f(v[idx]) for v in V[nA:]], 0)
        pt = float(stat(np.arange(s)[None])[0])
        draws = np.empty(n_boot); done = 0; step = max(1, int(2e7 // (s * V[0].shape[1] * len(V))))
        while done < n_boot:
            b = min(step, n_boot - done); draws[done:done + b] = stat(rng.integers(0, s, size=(b, s))); done += b
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        per_seed = [float(f(V[i][None])[0] - f(V[nA + j][None])[0]) for i in range(nA) for j in range(len(predsB)) if i == j]
        out["metrics"][name] = dict(diff=pt, lo=float(lo), hi=float(hi), statistic=kind, n_shells=int(s),
                                    tail_fraction=float((draws >= 0).mean() if name in lower else (draws <= 0).mean()),
                                    per_seed_pair_diff=per_seed)
    return out


RELEASE_ROOT = os.environ.get("BLISS_RELEASE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "BLISS-1.0"))


def standard_nodes(n_mesh=76805, n=8192):
    """The fixed 8,192-node subset every field metric is computed on (sorted default_rng(0) draw).

    Parameters
    ----------
    n_mesh : int, default=76805
        Number of nodes of the full mesh.
    n : int, default=8192
        Size of the subset.

    Returns
    -------
    nodes : numpy.ndarray, shape (n,)
        Sorted node indices.
    """
    return np.sort(np.random.default_rng(0).choice(n_mesh, n, replace=False))


def release_truth(ids, nodes=None, root=None):
    """Ground truth for the given shell ids, read from the release (canonical/<id>.npz): U_peak at `nodes` (default the
    standard 8,192-node subset; "all" for the 76,805 nodes), kappa, r_d2_d1, and the unit node directions P.

    Parameters
    ----------
    ids : sequence of int
        Shell identifiers.
    nodes : numpy.ndarray or 'all', default=None
        Node indices; the standard subset if None.
    root : str, default=None
        Release directory; RELEASE_ROOT if None.

    Returns
    -------
    truth : dict
        U, kappa, r and P of the shells.
    """
    root = root or RELEASE_ROOT
    P = np.load(_os.path.join(root, "shared.npz"))["P"].astype(np.float64)
    nodes = np.arange(len(P)) if (isinstance(nodes, str) and nodes == "all") else (standard_nodes(len(P)) if nodes is None else np.asarray(nodes))
    U = np.zeros((len(ids), len(nodes), 3), np.float32); K = np.zeros(len(ids)); R = np.zeros(len(ids))
    for i, s in enumerate(ids):
        z = np.load(_os.path.join(root, "canonical", "%d.npz" % int(s)))
        U[i] = z["U_peak"][nodes]; K[i] = float(z["kappa"]); R[i] = float(z["r_d2_d1"])
    return dict(true_U=U, true_K=K, r=R, P=(P[nodes] / np.linalg.norm(P[nodes], axis=1, keepdims=True)).astype(np.float32), nodes=nodes)


def score_submission(pred, split, n_boot=2000, alpha=0.05, seed=0, nodes=None, root=None, require_full=True):
    """Score a SUBMISSION: only ids, pred_U (S, N, 3) and pred_K (S,) are read from it; the ground truth is loaded
    from the release by shell id (release_truth), never from the submitted file. N must be the standard 8,192-node
    subset (default) or the full mesh (nodes="all"). With require_full, the ids must be exactly the test set of
    `split` ("B1", "B2", "B3"). Returns score_ci's dict (score() unchanged).

    Parameters
    ----------
    pred : str or mapping
        File with ids, pred_U and pred_K.
    split : str, {'B1', 'B2', 'B3'}
        Protocol.
    n_boot : int, default=2000
        Bootstrap resamples.
    alpha : float, default=0.05
        Two-sided level.
    seed : int, default=0
        Seed of the resampling.
    nodes : numpy.ndarray or 'all', default=None
        Node set of pred_U.
    root : str, default=None
        Release directory.
    require_full : bool, default=True
        Require the ids to be exactly the test set.

    Returns
    -------
    result : dict
        As score_ci.

    Raises
    ------
    ValueError
        If the ids do not match the test set of the split.
    """
    z, has = _load_pred(pred)
    ids = np.asarray(z["ids"]).astype(int)
    if require_full:
        te = set(int(i) for i in _json.load(open(_split_path(split)))["test"])
        missing, extra = te - set(ids.tolist()), set(ids.tolist()) - te
        if missing or extra or len(ids) != len(te):
            raise ValueError("submission ids do not match the %s test set: %d missing, %d extra, %d duplicated"
                             % (split, len(missing), len(extra), len(ids) - len(set(ids.tolist()))))
    t = release_truth(ids, nodes, root)
    pu = np.asarray(z["pred_U"], np.float32); pk = np.asarray(z["pred_K"], np.float64)
    if pu.shape != t["true_U"].shape or pk.shape != (len(ids),):
        raise ValueError("pred_U must be %s and pred_K (%d,), got %s and %s" % (t["true_U"].shape, len(ids), pu.shape, pk.shape))
    if not (np.isfinite(pu).all() and np.isfinite(pk).all()):
        raise ValueError("pred_U / pred_K contain non-finite values (%d / %d entries); a submission must be finite"
                         % (int((~np.isfinite(pu)).sum()), int((~np.isfinite(pk)).sum())))
    return score_ci(dict(ids=ids, true_U=t["true_U"], pred_U=pu, true_K=t["true_K"], pred_K=pk, P=t["P"], r=t["r"]),
                    n_boot=n_boot, alpha=alpha, seed=seed, split=split)


def score_by_regime(pred, n_boot=2000, split=None, seed=0):
    """score_ci on the pooled test set and on its parts: sparse (ids < 4758, beta_min = 25 deg), dense (ids >= 4758,
    beta_min = 10 deg), and the released-q bins q < 0.7, 0.7 <= q <= 0.9, q > 0.9. Each part is scored as its own
    set (its own scalar-mean reference); empty parts are omitted.

    Parameters
    ----------
    pred : str or mapping
        Prediction file.
    n_boot : int, default=2000
        Bootstrap resamples.
    split : str, default=None
        Protocol name.
    seed : int, default=0
        Seed of the resampling.

    Returns
    -------
    result : dict
        score_ci output for the pooled set and for each part.
    """
    z, _ = _load_pred(pred); ids = np.asarray(z["ids"]).astype(int); q = released_q(ids)
    parts = dict(pooled=np.ones(len(ids), bool), sparse=ids < 4758, dense=ids >= 4758,
                 q_below_0p7=q < 0.7, q_0p7_to_0p9=(q >= 0.7) & (q <= 0.9), q_above_0p9=q > 0.9)
    return {k: score_ci(pred, n_boot=n_boot, split=split, seed=seed, subset=m) for k, m in parts.items() if m.sum() >= 2}



# --- twin shells -----------------------------------------------------------------------------------
# The simulated set contained 600 pairs of shells with identical defect tables (metadata/twins.csv). The
# released canonical records keep one member of each pair, so the released splits B1/B2/B3 contain no twin
# pair and nothing is dropped (excluded_test_counts is zero for all three). The helpers below remain for the
# legacy pre-release splits only; score_ci applies them only when asked (drop_leaked=True with a split).
import os as _os, csv as _csv, json as _json
_HERE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))   # repository root
TWINS_CSV = _os.environ.get("BLISS_TWINS", _os.path.join(_HERE, "metadata", "twins.csv"))
SPLITS_DIR = _os.environ.get("BLISS_SPLITS", _os.path.join(_HERE, "splits"))
# The released protocols: BLISS-1.0/splits/B1.json, B2.json, B3.json (identical to splits/v2/random, delta, regime).
V2_SPLITS_DIR = _os.environ.get("BLISS_V2_SPLITS", _os.path.join(RELEASE_ROOT, "splits"))
PROTOCOLS = ("B1", "B2", "B3")


def _split_path(split, splits_dir=None):
    """B1/B2/B3 -> the released split file; any other name -> <splits_dir or SPLITS_DIR>/<name>.json (legacy).

    Parameters
    ----------
    split : str
        Protocol or legacy split name.
    splits_dir : str, default=None
        Directory of legacy split files.

    Returns
    -------
    path : str
        Path of the split file.
    """
    if split in PROTOCOLS and splits_dir is None:
        return _os.path.join(V2_SPLITS_DIR, split + ".json")
    return _os.path.join(splits_dir or SPLITS_DIR, split + ".json")

def twins(path=TWINS_CSV):
    """{shell: twin} for every shell that has one; empty if the file is absent.

    Parameters
    ----------
    path : str, default=TWINS_CSV
        Path of twins.csv.

    Returns
    -------
    twins : dict
        Map from shell id to the id of its twin.
    """
    if not _os.path.exists(path): return {}
    return {int(r["shell"]): int(r["twin"]) for r in _csv.DictReader(open(path))}

def leaked_test_shells(split, splits_dir=None, twins_path=TWINS_CSV):
    """Test shells of `split` whose twin sits in that split's train or val set. These are dropped at scoring.

    Parameters
    ----------
    split : str
        Split name.
    splits_dir : str, default=None
        Directory of legacy split files.
    twins_path : str, default=TWINS_CSV
        Path of twins.csv.

    Returns
    -------
    ids : set of int
        Leaked test shells.
    """
    sp = _json.load(open(_split_path(split, splits_dir))); tw = twins(twins_path)
    seen = set(sp["train"]) | set(sp["val"])
    return {int(s) for s in sp["test"] if int(s) in tw and tw[int(s)] in seen}

def excluded_test_shells(split, splits_dir=None, twins_path=TWINS_CSV):
    """Test shells to leave out of every number of `split`: (a) those whose twin is in train or val (a leak), and
    (b) the higher-id member of every twin pair that has BOTH members in test (an exact duplicate record, which
    would otherwise count twice in every mean and narrow every interval). Returns the set; the two counts are
    available from excluded_test_counts().

    Parameters
    ----------
    split : str
        Split name.
    splits_dir : str, default=None
        Directory of legacy split files.
    twins_path : str, default=TWINS_CSV
        Path of twins.csv.

    Returns
    -------
    ids : set of int
        Test shells left out of every number.
    """
    sp = _json.load(open(_split_path(split, splits_dir))); tw = twins(twins_path)
    seen = set(sp["train"]) | set(sp["val"]); te = set(int(s) for s in sp["test"])
    leak = {s for s in te if s in tw and tw[s] in seen}
    dup = {s for s in te if s in tw and tw[s] in te and s > tw[s]}
    return leak | dup

def excluded_test_counts(split, splits_dir=None, twins_path=TWINS_CSV):
    """Number of test shells left out of a split, by reason.

    Parameters
    ----------
    split : str
        Split name.
    splits_dir : str, default=None
        Directory of legacy split files.
    twins_path : str, default=TWINS_CSV
        Path of twins.csv.

    Returns
    -------
    counts : dict
        leaked, duplicate_in_test and test.
    """
    sp = _json.load(open(_split_path(split, splits_dir))); tw = twins(twins_path)
    seen = set(sp["train"]) | set(sp["val"]); te = set(int(s) for s in sp["test"])
    return dict(leaked=sum(1 for s in te if s in tw and tw[s] in seen),
                duplicate_in_test=sum(1 for s in te if s in tw and tw[s] in te and s > tw[s]), test=len(te))
