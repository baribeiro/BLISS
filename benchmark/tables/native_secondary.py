"""Secondary field metrics of the full-mesh (76,805-node) track: RMSE and relative L2 error of the field, on the 8,192
scoring nodes.

    python benchmark/tables/native_secondary.py [--protocols B1,B2,B3] [--out-dir outputs]

No training and no inference: stored test predictions only.

INPUTS
  <out-dir>/native_rescore_v2.json   the B1 files already scored by native_rescore.py
  <out-dir>/ladder_native_v2.json    the B2 and B3 files already scored by ladder_native.py, under the key "native"
  the *_test_pred.npz files named in the two json files, with their fields on all 76,805 nodes
  <out-dir>/canon_nodes_8192.npz     the scoring nodes and their positions (prepare_data.py)
  --out-dir defaults to BLISS_HEAVY or <BLISS_ROOT>/outputs and BLISS_ROOT to the repository.

DEFINITIONS
  Each stored prediction is cut to bliss.metrics.standard_nodes(), the 8,192 scoring nodes, and passed to
  rescore.secondary() with the unit node directions of canon_nodes_8192.npz: the same function and the same node
  positions that give the RMSE and relative L2 columns of the full tables, on the full field and after removing the
  uniform contraction of each shell.
  Cache: a file is scored once. Its result is stored with its path, size and modification time, and a later run
  scores only the files that are new or have changed. The json is written after every file.
Output: <out-dir>/native_secondary_v2.json (read by main_table.py). Reading a file takes most of the time: the files
hold 1 to 4 GB.

Functions
---------
sources
    (fam, protocol, seed, file) of every full-mesh file already scored on the headline metrics.
is_cached
    Whether the stored result of a file was computed from the file as it is now.
main
    Compute the secondary metrics of every file not yet in native_secondary_v2.json.
"""
import argparse, json, os, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs")
sys.path[:0] = [B, os.path.dirname(os.path.abspath(__file__))]
from bliss.metrics import standard_nodes   # noqa: E402
from native_rescore import N_MESH          # noqa: E402
from rescore import secondary              # noqa: E402


def sources(out_dir=H):
    """(fam, protocol, seed, file) of every full-mesh file already scored on the headline metrics.

    Parameters
    ----------
    out_dir : str
        Directory of native_rescore_v2.json and ladder_native_v2.json.

    Returns
    -------
    out : list of tuple
        Sorted (family, protocol, seed, path).
    """
    out = []
    for fam, r in json.load(open(out_dir + "/native_rescore_v2.json")).items():
        if isinstance(r, dict):
            out += [(fam, "B1", int(s), x["file"]) for s, x in r["seeds"].items()]
    for key, rows in json.load(open(out_dir + "/ladder_native_v2.json"))["native"].items():
        fam, prot = key.split("|"); out += [(fam, prot, int(s), x["file"]) for s, x in rows.items()]
    return sorted(out)


def is_cached(old, tag):
    """Whether the stored result of a file was computed from the file as it is now.

    Parameters
    ----------
    old : dict or None
        Stored result of the file (file, size, mtime, secondary), or None if there is none.
    tag : dict
        file, size and mtime of the file now.

    Returns
    -------
    cached : bool
        True if the stored result has the same path, size and modification time.
    """
    return bool(old) and all(old[k] == v for k, v in tag.items())


def main():
    """Compute the secondary metrics of every file not yet in native_secondary_v2.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocols", default="B1,B2,B3")
    ap.add_argument("--out-dir", default=H, help="directory of canon_nodes_8192.npz, of the two json files read and of the json written")
    a = ap.parse_args(); OUT = a.out_dir + "/native_secondary_v2.json"
    for p, what in ((a.out_dir + "/canon_nodes_8192.npz", "the scoring nodes, written by benchmark/data/prepare_data.py"),
                    (a.out_dir + "/native_rescore_v2.json", "written by benchmark/tables/native_rescore.py"),
                    (a.out_dir + "/ladder_native_v2.json", "written by benchmark/tables/ladder_native.py")):
        if not os.path.exists(p): sys.exit("native_secondary.py: missing input %s (%s)" % (p, what))
    NODES = standard_nodes(); cz = np.load(a.out_dir + "/canon_nodes_8192.npz"); assert np.array_equal(NODES, cz["NODES"])
    P = cz["P"].astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for fam, prot, s, f in sources(a.out_dir):
        if prot not in a.protocols.split(","): continue
        st = os.stat(f); tag = dict(file=f, size=st.st_size, mtime=st.st_mtime)
        if is_cached(res.get("%s|%s" % (fam, prot), {}).get(str(s)), tag): continue
        t = time.time(); z = np.load(f); assert z["pred_U"].shape[1] == N_MESH, f
        cut = dict(true_U=z["true_U"][:, NODES], pred_U=z["pred_U"][:, NODES], true_K=z["true_K"], pred_K=z["pred_K"])
        assert np.allclose(z["P"][NODES] / np.linalg.norm(z["P"][NODES], axis=1, keepdims=True), P, atol=1e-5), "node positions differ"
        sec = secondary(cut, P); del z, cut
        res.setdefault("%s|%s" % (fam, prot), {})[str(s)] = dict(tag, secondary=sec)
        json.dump(res, open(OUT, "w"), indent=1, default=float)          # after every file, so an interrupted run keeps its work
        print("%s %s s%d rmse_dev %.4f rel_l2_dev %.3f (%.0f s)" % (fam, prot, s, sec["rmse_mm_deviation"], sec["rel_l2_deviation"], time.time() - t), flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
