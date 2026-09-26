"""One pass over release/core -> two flat arrays the baseline trainers load in seconds.

    rel_nodes_8192.npz  ids, W (n,8192), U (n,8192,3), K, R (r_d2_d1), P (8192,3), NODES
    rel_grid.npz        ids, GW (n,180,720), GU (n,180,720,3), K
Same fixed node subset as heavy/data_all_8192.npz (seed 0), so all point-cloud baselines see the same nodes.
"""
import glob, os, sys, time, numpy as np

# The 180x720 grid is no longer stored per shell. Build it here with the map in shared.npz,
# exactly as the release used to (inverse distance, k=4, float32), so the cached tensor is bit-identical.
_GRID = {}
def _to_grid(a, core_file):
    """Resample a per-node array onto the 180 x 720 grid (inverse distance on the four nearest nodes).

    Parameters
    ----------
    a : numpy.ndarray
        Array of shape (N,) or (N, 3).
    core_file : str
        Record the array comes from (unused).

    Returns
    -------
    g : numpy.ndarray
        Array on the grid, float32.
    """
    if not _GRID:
        sh = np.load(os.path.join(os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "release", "shared.npz"))
        _GRID.update(idx=sh["knn_idx"], w=sh["knn_w"], nb=len(sh["grid_beta"]), nt=len(sh["grid_theta"]))
    idx, w = _GRID["idx"], _GRID["w"]
    out = (a[idx] * (w[..., None] if a.ndim == 2 else w)).sum(1)
    return out.reshape(_GRID["nb"], _GRID["nt"], -1).squeeze().astype(np.float32)
B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); OUT = B + "/outputs"
sh = np.load(B + "/release/shared.npz"); P = sh["P"]
FULL = len(sys.argv) > 1 and sys.argv[1] == "full"
CANON = "canon" in sys.argv
NODES = np.arange(len(P)) if FULL else np.sort(np.random.default_rng(0).choice(len(P), 8192, replace=False))
NN = len(NODES)
SRC = B + ("/BLISS-1.0/canonical" if CANON else "/release/core")
files = sorted(glob.glob(SRC + "/*.npz"), key=lambda f: int(os.path.basename(f)[:-4]))
n = len(files); ids = np.array([int(os.path.basename(f)[:-4]) for f in files])
W = np.zeros((n, NN), np.float32); U = np.zeros((n, NN, 3), np.float32); K = np.zeros(n); R = np.zeros(n)
GW = None if (FULL or CANON) else np.zeros((n, 180, 720), np.float32)
GU = None if (FULL or CANON) else np.zeros((n, 180, 720, 3), np.float32)
t = time.time()
for i, f in enumerate(files):
    z = np.load(f); W[i] = z["w"][NODES]; U[i] = z["U_peak"][NODES]; K[i] = z["kappa"]; R[i] = z["r_d2_d1"]
    if not FULL and not CANON: GW[i] = _to_grid(z["w"], f); GU[i] = _to_grid(z["U_peak"], f)
    if i % 500 == 0: print("  %d/%d  %.0fs" % (i, n, time.time() - t), flush=True)
np.savez(OUT + ("/%s_nodes_%s.npz" % ("canon" if CANON else "rel", "full" if FULL else "8192")),
         ids=ids, W=W, U=U, K=K, R=R, P=P[NODES], NODES=NODES)
if not FULL and not CANON: np.savez(OUT + "/rel_grid.npz", ids=ids, GW=GW, GU=GU, K=K)
if not FULL and CANON:
    g = np.load(OUT + "/rel_grid.npz"); gi = {int(v): j for j, v in enumerate(g["ids"])}
    sel = np.array([gi[int(v)] for v in ids])
    np.savez(OUT + "/canon_grid.npz", ids=ids, GW=g["GW"][sel], GU=g["GU"][sel], K=g["K"][sel])
    print('canon_grid.npz: %d shells selected from the grid file' % len(sel))
print("done: %s, %d shells, %d nos, %.1f GB, %.0fs" % (SRC.split("/")[-1], n, NN, (W.nbytes + U.nbytes) / 1e9, time.time() - t))
