"""Tests of the helpers of the full-mesh tables on synthetic arrays and scores (no data download needed)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark", "tables"))
import ladder_native as LN   # noqa: E402
import native_rescore as NR   # noqa: E402
import native_secondary as NS   # noqa: E402


def _metrics(v):
    """Headline metrics with the same value, in the layout the scripts store.

    Parameters
    ----------
    v : float
        Value of every metric.

    Returns
    -------
    m : dict
        Metric -> dict(point, lo, hi).
    """
    return {k: dict(point=v, lo=v - 0.1, hi=v + 0.1) for k, _ in NR.H4}


def test_seed_of():
    """The seed is read from the _s<n> tag of the file name, and is 0 without a tag."""
    assert NR.seed_of("runs/fam_tag_s2_test_pred.npz") == 2 and NR.seed_of("runs/fam_tag_test_pred.npz") == 0
    assert NR.seed_of("runs/fam_s1_tag_test_pred.npz") == 0
    assert NR.seed_of("runs/fam_tag_s12_test_pred.npz") == 12


def test_cut_nodes():
    """The fields and the node directions are cut to the nodes; the per-shell arrays are kept."""
    rng = np.random.default_rng(0); nodes = np.array([1, 4, 5])
    z = dict(ids=np.arange(2), true_U=rng.normal(size=(2, 7, 3)), pred_U=rng.normal(size=(2, 7, 3)), true_K=rng.uniform(size=2),
             pred_K=rng.uniform(size=2), P=rng.normal(size=(7, 3)), r=rng.uniform(size=2))
    c = NR.cut_nodes(z, nodes)
    assert set(c) == set(z) and c["true_U"].shape == (2, 3, 3) and c["P"].shape == (3, 3)
    assert np.array_equal(c["pred_U"], z["pred_U"][:, nodes]) and np.array_equal(c["P"], z["P"][nodes])
    assert all(c[k] is z[k] for k in ("ids", "true_K", "pred_K", "r"))


def test_native_b1_table():
    """One row per family with a final, the mean over the seeds and the range; a missing family is named in the caption."""
    res = {"figconv": dict(name="FigConvNet", seeds={0: dict(metrics=_metrics(1.0)), 1: dict(metrics=_metrics(2.0))}),
           "sfno": "no native final (file missing)", "mgn": dict(name="MeshGraphNet", seeds={})}
    L = NR.table(res)
    rows = [x for x in L if x.endswith("\\\\") and not x.startswith("model")]
    assert len(rows) == 1 and rows[0].startswith("FigConvNet & 1.50\\,{\\tiny(1.00--2.00)} & 1.500")
    assert any("SFNO has no native final." in x for x in L) and L[-1] == "\\end{table}"


def test_pts_and_cell():
    """pts keeps point and interval of every metric; a cell is the mean, with the range for more than one seed."""
    sc = dict(metrics={"a": dict(point=1.0, lo=0.5, hi=1.5, n=7)}, n=7)
    assert LN.pts(sc) == {"a": dict(point=1.0, lo=0.5, hi=1.5)}
    d = [{"a": dict(point=1.0)}, {"a": dict(point=2.0)}]
    assert LN.ms(d, "a", "%.2f") == "1.50\\,{\\tiny(1.00--2.00)}" and LN.ms(d[:1], "a", "%.1f") == "1.0"


def test_npz_shape(tmp_path):
    """The shape of an array is read from the header of the file."""
    f = str(tmp_path / "p.npz"); np.savez(f, pred_U=np.zeros((4, 9, 3), np.float32), ids=np.arange(4))
    assert LN.npz_shape(f) == (4, 9, 3) and LN.npz_shape(f, "ids") == (4,)


def test_ladder_and_native_tables():
    """Three rows per family in the ladder; one row per family and protocol, sorted, in the full-mesh table."""
    seed = dict(n2048=_metrics(1.0), n8192_on_2048=_metrics(2.0), n8192=_metrics(3.0))
    L = LN.ladder_table({"mgn": {0: seed, 1: seed}})
    rows = [x for x in L if x.startswith("MeshGraphNet")]
    assert len(rows) == 3 and "2{,}048 / 2{,}048 & 1.00" in rows[0] and "8{,}192 / 8{,}192 & 3.00" in rows[2]
    T = LN.native_table({"sfno|B3": {0: dict(metrics=_metrics(2.0))}, "figconv|B2": {0: dict(metrics=_metrics(1.0)), 1: dict(metrics=_metrics(3.0))}})
    rows = [x for x in T if x.startswith(("FigConvNet", "SFNO"))]
    assert rows[0].startswith("FigConvNet & B2 & 2 & 2.00\\,{\\tiny(1.00--3.00)}") and rows[1].startswith("SFNO & B3 & 1 & 2.00 &")


def test_sources_and_cache(tmp_path):
    """The files of the two score files are listed with family, protocol and seed; a changed file is not cached."""
    json.dump({"mgn": dict(seeds={"1": dict(file="b1_s1.npz"), "0": dict(file="b1.npz")}), "sfno": "no native final"},
              open(tmp_path / "native_rescore_v2.json", "w"))
    json.dump(dict(ladder={}, native={"mgn|B3": {"0": dict(file="b3.npz")}}), open(tmp_path / "ladder_native_v2.json", "w"))
    assert NS.sources(str(tmp_path)) == [("mgn", "B1", 0, "b1.npz"), ("mgn", "B1", 1, "b1_s1.npz"), ("mgn", "B3", 0, "b3.npz")]
    tag = dict(file="b1.npz", size=10, mtime=1.5)
    assert NS.is_cached(dict(tag, secondary={}), tag) and not NS.is_cached(None, tag)
    assert not NS.is_cached(dict(tag, size=11, secondary={}), tag)
