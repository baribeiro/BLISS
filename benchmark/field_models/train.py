"""One trainer for every baseline, on the official splits, with resume.

    python benchmark/field_models/train.py MODEL SPLIT [--epochs 200] [--bs 8] [--select buckling] [hyperparameter flags] [--smoke]
      MODEL  transformer | transolver | mgn | deeponet | mlp | mlp80 | sfno | figconv
      SPLIT  v2/random | v2/delta | v2/regime (= the released B1, B2, B3); random | delta | regime are the pre-release v1

Recipe (the protocol; never searched): AdamW betas 0.9/0.999, eps 1e-8, 5 % linear warm-up then cosine to 1 % of
the peak lr, EMA 0.999, bf16 autocast, grad clip 1, effective batch 8 (micro-batches accumulate), loss = MSE(U) +
0.3 MSE(kappa) on standardised targets, the field term on 8,192 nodes per epoch (SFNO: 8,192 area-weighted grid
cells). lr, weight decay and the architecture knobs come from the flags (search_space.py searches them; confirm.py /
transfer.py pass the frozen winner). Validation every --val-every epochs (5); the checkpoint kept is the EMA at the
best validation --select metric (default buckling = the released buckling_dimple_r2); the test set is scored once
at the end with those weights (never with --no-test). Resumes from heavy/<TAG>_ckpt.pt, written at each validation.
Writes heavy/<TAG>_{val,test}_pred.npz (ids, pred_U, pred_K, true_U, true_K, P, r, epoch, ang) and one line of
heavy/results.jsonl.

Models (plain torch except SFNO, torch-harmonics, and FigConvNet, PhysicsNeMo); sizes are set by the flags:
  transformer  full attention over the nodes, attention pooling for kappa
  transolver   Physics-Attention (Wu et al. 2024)
  mgn          MeshGraphNet-style GN on a kNN graph of the nodes
  deeponet     branch on w (nodes -> 3p), trunk on node position
  mlp / mlp80  MLP on the DEFECT PARAMETERS (top-20 / top-80 by depth + N), not on w
  sfno         SFNO on the released 180 x 720 latitude-longitude grid of the hemisphere (--grid H,W resamples it;
               longitude treated as periodic), read back onto the nodes bilinearly
  figconv      FIGConvUNet on the point cloud, its own scalar head for kappa

Classes
-------
PhysicsAttention
    Physics-Attention of Transolver (Wu et al. 2024).
Block
    Pre-norm transformer block: token mixer plus a 4x feed-forward layer, both residual.
FullAttn
    Full multi-head self-attention over all nodes.
TokenNet
    Transformer and Transolver on the nodes of the shell.
MLP
    MLP on the list of defects (the 20 or 80 deepest defects and their count), with a field head and a
    kappa head.
DeepONet
    DeepONet: a branch network on w and a trunk network on the node position.
MGN
    MeshGraphNet-style graph network on the k-nearest-neighbour graph of the nodes.
SFNONet
    Spherical Fourier Neural Operator (torch-harmonics) on the latitude-longitude grid of the
    hemisphere.
FigConv
    FigConvNet (NVIDIA PhysicsNeMo FIGConvUNet) on the point cloud of the nodes, with its scalar head
    for kappa.

Functions
---------
target_field
    Normalised field target of the given shells, on the model's output grid.
evaluate
    Predict a set of shells with a model and compute the selection metrics.
dump
    Write the predictions of a set of shells to <OUT>/<TAG>_<name>_pred.npz.
query
    Nodes (or grid cells) that carry the field loss in one epoch, drawn with the seed 2027000 + epoch.
"""
import argparse, copy, csv, json, os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("BLISS_HEAVY", B + "/outputs"); dev = "cuda"
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("split")
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--bs", type=int, default=8, help='effective batch size; the protocol fixes 8 and it is not searched'); ap.add_argument("--micro", type=int, default=0)
ap.add_argument("--lr", type=float, default=4e-4); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--params-only", action="store_true", help='print the number of parameters and exit, without training')
ap.add_argument("--size", default="base", choices=["base", "large"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default=""); ap.add_argument("--use-best", action="store_true", help="lr and size from heavy/hp_best.json")
ap.add_argument("--frac", type=float, default=1.0, help="data-scaling: fraction of the training split actually used")
ap.add_argument("--sample-weights", default="", help="splits/<name>.json of per-shell loss weights "
                "(latitude reweighting: same training shells as the control, different weight)")
ap.add_argument("--p", type=int, default=0, help="DeepONet basis size override (rank-p conjecture test); 0 = size default")
ap.add_argument("--grid", default=None, help="SFNO input grid as H,W (default 180,720; 64,128 matches the 8,192 nodes the other models read)")
ap.add_argument("--decision-weight", type=float, default=0.0, help="add a cross-entropy on the two\n                competing sites of each training shell, so the model is supervised on the decision\n                itself and not only through the field. 0 leaves the protocol runs untouched.")
ap.add_argument("--query-nodes", type=int, default=8192,
                help='query nodes for the field loss at each epoch; 0 = all. Evaluation always uses all evaluation nodes, this only sets the supervision.')
ap.add_argument("--val-every", type=int, default=5, help='validate every N epochs')
ap.add_argument("--select", default="buckling", choices=["buckling", "dimple"],
                help='metric that selects the best checkpoint and is returned to the search: buckling = the published one (buckling_dimple_r2, uniform contraction removed, scalar reference); dimple = the earlier proxy')
ap.add_argument("--no-test", action="store_true",
                help='do not touch the test set. The search uses this, because the B1 test set stays closed until the configuration is frozen')
ap.add_argument("--dump-val", action="store_true", help="re-write the VALIDATION predictions from\n                the selected weights, without training. Only needed to rebuild an artefact.")
ap.add_argument("--dump-train", action="store_true", help="also write predictions over the TRAINING\n                shells. The candidate head that reads a field model has no training-split predictions\n                to learn from, so it had to be cross-validated on the test set; this removes that.")
ap.add_argument("--wd", type=float, default=0.05, help='AdamW weight decay')
ap.add_argument("--dim", type=int, default=None, help="transformer/transolver: d_model")
ap.add_argument("--layers", type=int, default=None, help="transformer/transolver/sfno: blocks")
ap.add_argument("--heads", type=int, default=None, help="transformer: attention heads")
ap.add_argument("--slices", type=int, default=None, help="transolver: Physics-Attention slices")
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--hidden", type=int, default=None, help="mgn: hidden dimension")
ap.add_argument("--blocks", type=int, default=None, help="mgn: blocks | figconv: blocks per level")
ap.add_argument("--knn", type=int, default=None, help="mgn: graph neighbours")
ap.add_argument("--embed", type=int, default=None, help="sfno: embedding")
ap.add_argument("--modes", type=int, default=None, help="sfno: spectral modes")
ap.add_argument("--levels", type=int, default=None, help='figconv: grid levels')
ap.add_argument("--width-mult", type=float, default=None, help="figconv: channel multiplier")
ap.add_argument("--width", type=int, default=None, help="deeponet/mlp: width")
ap.add_argument("--depth", type=int, default=None, help="deeponet: depth")
ap.add_argument("--prefix-inc", type=int, default=0, help="feed the displacement field N Riks\n                increments BEFORE the first limit point as extra input channels, read from\n                release/path. App. H measures that the two competing sites only separate in the last\n                few increments; this asks whether a model handed that window reads the decision.")
ap.add_argument("--nodes", type=int, default=8192, help='input resolution: 2048/4096 subsample the 8192 nodes; 16384/32768 read data_all_32768.npz; 76805 reads the full native mesh')
A = ap.parse_args(); torch.backends.cuda.matmul.allow_tf32 = True
if A.use_best:
    hb = json.load(open(OUT + "/hp_best.json"))[A.model]; A.lr, A.size = hb["lr"], hb["size"]
if A.frac < 1 and not A.tag: A.tag = "frac%g" % A.frac
if A.nodes != 8192 and not A.tag: A.tag = "n%d" % A.nodes
if A.p and not A.tag: A.tag = "p%d" % A.p
# The same rule as --sample-weights below, for the two flags that also change the experiment: a smoke
# run truncates the test set to 16 shells, and on 09-19 an untagged one overwrote the transformer|random
# artefact that Table 1 was computed from. Only the checkpoint survived, because --smoke does not write it.
if A.smoke and not A.tag: A.tag = "smk"
if A.prefix_inc and not A.tag: A.tag = "pre%d" % A.prefix_inc
if A.decision_weight and not A.tag: A.tag = "dec%g" % A.decision_weight
# A weighted run is a different experiment from the plain one on the same split, and its control IS
# the plain one, so it must never write over it: default the tag to the weight set.
if A.sample_weights and not A.tag: A.tag = A.sample_weights.replace("regime_", "").replace("_", "")
TAG = "%s_%s" % (A.model, A.split.replace("/", "")) + ("_" + A.tag if A.tag else "") + ("_s%d" % A.seed if A.seed else ""); CK = "%s/%s_ckpt.pt" % (OUT, TAG)
torch.manual_seed(A.seed); LARGE = A.size == "large"
MICRO = A.micro or {"mgn": 4, "transformer": 8, "figconv": 8, "sfno": 8}.get(A.model, A.bs)   # grad-accumulate to A.bs
if not A.micro:
    if A.nodes > 8192: MICRO = min(MICRO, 2)
    if A.nodes > 32768: MICRO = 1

sp = json.load(open("%s/splits/%s.json" % (B, A.split)))
_PFX = "canon" if A.split.startswith("v2/") else "rel"
d = np.load(OUT + "/%s_nodes_%s.npz" % (_PFX, "full" if A.nodes >= 76805 else "8192"))
ids, W, U, K, P = d["ids"], d["W"], d["U"], d["K"], d["P"]
if A.nodes > 8192 and A.nodes < 76805:                     # 32k prep (4975 shells, own node set); r_d2_d1 mapped by id
    d32 = np.load(OUT + "/data_all_32768.npz"); rmap = dict(zip(ids.tolist(), d["R"])); ids, W, U, K, P = d32["ids"], d32["W"], d32["U"], d32["K"], d32["P"]
    d = {"R": np.array([rmap.get(int(i), np.nan) for i in ids])}
if A.nodes != P.shape[0]:
    sel = np.sort(np.random.default_rng(1).choice(P.shape[0], A.nodes, replace=False)); W, U, P = W[:, sel], U[:, sel], P[sel]
pos = {int(i): j for j, i in enumerate(ids)}
tr = np.array([pos[i] for i in sp["train"] if i in pos]); va = np.array([pos[i] for i in sp["val"] if i in pos]); te = np.array([pos[i] for i in sp["test"] if i in pos])
if A.frac < 1: tr = np.sort(np.random.default_rng(100 + A.seed).choice(tr, int(round(A.frac * len(tr))), replace=False))
if A.smoke: tr, va, te, A.epochs = tr[:64], va[:16], te[:16], 2
n_tr, NN = len(tr), P.shape[0]; rng = np.random.default_rng(A.seed)
SW = None
if A.sample_weights:      # importance weights on the training shells; mean 1, so the gradient scale is unchanged
    _w = json.load(open("%s/splits/%s.json" % (B, A.sample_weights)))["weights"]
    _v = np.array([_w.get(str(int(ids[j])), 1.0) for j in tr], dtype=np.float32); _v /= _v.mean()
    print("sample weights %s: min %.2f max %.2f ESS %.0f of %d"
          % (A.sample_weights, _v.min(), _v.max(), _v.sum() ** 2 / (_v ** 2).sum(), len(_v)), flush=True)
    SW = dict(zip(tr.tolist(), _v.tolist()))
wm, ws = W[tr].mean(), W[tr].std(); us = U[tr].std(); km, ks = K[tr].mean(), K[tr].std()

# The two competing sites of every training shell, by the same rule the ambiguity metrics use, so a
# model can be supervised on the decision directly instead of only through the field it is read from.
SITES = None
if A.decision_weight > 0:
    sys.path.insert(0, B)
    from bliss.metrics import _two_peaks, _ring
    Pu_ = P / np.linalg.norm(P, axis=1, keepdims=True)       # the two-peak rule works on unit normals
    nb = _ring(Pu_, 12)
    s1s, s2s, rows = [], [], []
    for j in tr.tolist():
        u_ = U[j]
        D = u_ - (u_ * Pu_).sum(-1).mean() * Pu_             # the uniform contraction out, as score() does
        i1, i2 = _two_peaks(D, Pu_, nb)
        if i2 is None: continue                       # one clear site: nothing to decide
        rows.append(j); s1s.append(i1); s2s.append(i2)
    SITES = dict(zip(rows, zip(s1s, s2s)))
    print("decision loss on %d of %d training shells that carry two competing sites (weight %.2f)"
          % (len(SITES), n_tr, A.decision_weight), flush=True)
T = lambda a: torch.tensor(a, device=dev)
Pt = T(P); Yk = T(((K - km) / ks).astype(np.float32)); Y = T(U / us)
print("data: %s split, %d train / %d val / %d test, %d nodes, model %s" % (A.split, n_tr, len(va), len(te), NN, A.model), flush=True)
to_nodes = lambda f: f
def target_field(idx): 
    """Normalised field target of the given shells, on the model's output grid.

    Parameters
    ----------
    idx : numpy.ndarray
        Indices of the shells in the loaded arrays.

    Returns
    -------
    Y : torch.Tensor
        Target field.
    """
    return Y[idx]
DIM, LAYERS, HEADS, SLICES = (384, 8, 8, 64) if LARGE else (256, 6, 8, 64)     # --size large: ~2.5x params
DIM = A.dim or DIM; LAYERS = A.layers or LAYERS; HEADS = A.heads or HEADS; SLICES = A.slices or SLICES
DROP = A.dropout or 0.0

if A.model in ("transformer", "transolver", "mgn", "deeponet"):
    X = T((W - wm) / ws)[:, :, None]
    if A.prefix_inc:
        Pu_ = P / np.linalg.norm(P, axis=1, keepdims=True)
        pre = np.zeros((len(ids), NN, 3), np.float32); miss = 0
        for i_, s_ in enumerate(ids):
            f_ = "%s/release/path/%d.npz" % (B, int(s_))
            if not os.path.exists(f_): miss += 1; continue
            u_ = np.load(f_)["U"]
            j_ = max(0, len(u_) - 1 - A.prefix_inc)
            d_ = np.asarray(u_[j_], np.float32)
            d_ = d_ - (d_ * Pu_).sum(-1).mean() * Pu_
            if d_.shape[0] >= NN: pre[i_] = d_[:NN]
        ps = float(pre[tr].std()) + 1e-12
        print('prefix-inc %d: %d shells without a load-path file, scale %.3e' % (A.prefix_inc, miss, ps), flush=True)
        X = torch.cat([X, T(pre / ps)], -1)
elif A.model in ("mlp", "mlp80"):
    TOPK = 80 if A.model == "mlp80" else 20; D = {}
    for r in csv.DictReader(open(B + "/metadata/defects.csv")):
        D.setdefault(int(r["id"]), []).append((float(r["delta"]), float(r["theta"]), float(r["phi"])))
    feats = np.zeros((len(ids), 3 * TOPK + 1), np.float32)
    for i, s in enumerate(ids):
        for j, (dl, th, ph) in enumerate(sorted(D[int(s)], key=lambda x: -x[0])[:TOPK]): feats[i, 3 * j:3 * j + 3] = (dl, np.cos(th) * ph, np.sin(th) * ph)
        feats[i, 3 * TOPK] = len(D[int(s)]) / 80.0
    X = T((feats - feats[tr].mean(0)) / (feats[tr].std(0) + 1e-6))
elif A.model == "sfno":
    from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator as SFNO
    g = np.load(OUT + ("/canon_grid.npz" if A.split.startswith("v2/") else "/rel_grid.npz")); assert (g["ids"] == ids).all(); GW, GU = g["GW"], g["GU"]
    # --grid H,W evaluates the same operator at a smaller input budget. The released grid is 180x720 =
    # 129,600 samples while every other model reads 8,192 nodes, so 64x128 = 8,192 is the matched cell and
    # the comparison of the two says how much of SFNO's lead is the operator and how much is resolution.
    GH, GWD = (int(x) for x in (A.grid.replace("x", ",").split(",") if A.grid else ("180", "720")))  # "64x128" too: a comma in --export splits the variable list
    if (GH, GWD) != (180, 720):
        import torch.nn.functional as _F
        # Longitude is PERIODIC: the released grid's columns sit at theta_j = -pi + j 2pi/720 (the seam +pi is not a
        # column), and to_nodes reads column k of a GWD-wide grid at -pi + k 2pi/GWD. Resizing the 720 columns with
        # align_corners=True alone put column k at -pi + k (719/(GWD-1)) 2pi/720 instead, up to 2.3 deg off at the
        # seam for 128 columns (ML2-01; truth round trip on 20 B1 test shells: buckling R2 0.969 -> 0.999). So the
        # seam column is appended, resized to GWD+1 and dropped. Latitude (0..pi/2 inclusive both ends) was right.
        # Only this --grid branch changes; the 180x720 path never enters it.
        def _r(a):
            """Resize grids to (GH, GWD), treating the longitude as periodic.

            Parameters
            ----------
            a : numpy.ndarray, shape (S, C, 180, 720)
                Grid arrays.

            Returns
            -------
            b : numpy.ndarray, shape (S, C, GH, GWD)
                Resized grids.
            """
            t = torch.tensor(a, dtype=torch.float32); t = torch.cat([t, t[..., :1]], -1)
            return _F.interpolate(t, size=(GH, GWD + 1), mode="bilinear", align_corners=True)[..., :GWD].numpy()
        GW = _r(GW[:, None])[:, 0]; GU = np.moveaxis(_r(np.moveaxis(GU, -1, 1)), 1, -1)
    beta = np.linspace(0, np.pi / 2, GH, dtype=np.float32)
    coords = np.stack([np.broadcast_to(np.cos(beta)[:, None], (GH, GWD)), np.broadcast_to(np.sin(beta)[:, None], (GH, GWD))])
    X = torch.tensor(np.concatenate([((GW - wm) / ws)[:, None], np.broadcast_to(coords, (len(ids), 2, GH, GWD))], 1), device=dev)
    Yg = T(np.moveaxis(GU / us, -1, 1).copy()); area = T(np.sin(beta) + 1e-3)[:, None]; area = area / area.sum() / GWD
    from bliss.geometry import frame_of
    e1, e2, ax = frame_of(P.astype(np.float64)); Q = P @ np.stack([e1, e2, ax]).T
    nb = np.arccos(np.clip(Q[:, 2], -1, 1)); nt = np.arctan2(Q[:, 1], Q[:, 0])
    samp = T(np.stack([(nt + np.pi) / np.pi - 1, nb / (np.pi / 2) * 2 - 1], -1).astype(np.float32))[None, :, None]
    def to_nodes(f):
        """Read a grid field back onto the nodes by bilinear interpolation (longitude periodic).

        Parameters
        ----------
        f : torch.Tensor, shape (B, C, GH, GWD)
            Field on the grid.

        Returns
        -------
        g : torch.Tensor, shape (B, N, C)
            Field on the nodes.
        """
        fw = torch.cat([f, f[..., :1]], -1)
        return F.grid_sample(fw, samp.expand(f.shape[0], -1, -1, -1), mode="bilinear", align_corners=True)[..., 0].transpose(1, 2)
    def target_field(idx): 
        """Normalised field target of the given shells, on the model's output grid.

        Parameters
        ----------
        idx : numpy.ndarray
            Indices of the shells in the loaded arrays.

        Returns
        -------
        Y : torch.Tensor
            Target field.
        """
        return Yg[idx]
elif A.model == "figconv":
    from physicsnemo.models.figconvnet.figconvunet import FIGConvUNet
    X = T((W - wm) / ws)[:, :, None]

class PhysicsAttention(nn.Module):
    """Physics-Attention of Transolver (Wu et al. 2024).

    Nodes are softly assigned to slices, attention runs between the slice tokens, and the result is
    projected back onto the nodes.
    """
    def __init__(s, dim, heads, slices):
        """Build the projections of the slice attention.

        Parameters
        ----------
        dim : int
            Width of the token features.
        heads : int
            Number of heads.
        slices : int
            Number of slices per head.
        """
        super().__init__(); s.h, s.dh, s.M = heads, dim // heads, slices
        s.to_x = nn.Linear(dim, dim); s.to_slice = nn.Linear(s.dh, slices); s.tau = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        s.qkv = nn.Linear(s.dh, 3 * s.dh); s.out = nn.Linear(dim, dim)
    def forward(s, x):
        """Mix the node features through the slice tokens.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N, dim)
            Node features.

        Returns
        -------
        y : torch.Tensor, shape (B, N, dim)
            Mixed node features.
        """
        Bn, Nn, _ = x.shape; xh = s.to_x(x).view(Bn, Nn, s.h, s.dh).transpose(1, 2)
        w = F.softmax(s.to_slice(xh) / s.tau.clamp(min=0.01), dim=-1)
        z = torch.einsum("bhnm,bhnd->bhmd", w, xh) / (w.sum(2)[..., None] + 1e-5)
        q, k, v = s.qkv(z).chunk(3, -1); z = F.scaled_dot_product_attention(q, k, v)
        return s.out(torch.einsum("bhnm,bhmd->bhnd", w, z).transpose(1, 2).reshape(Bn, Nn, -1))

class Block(nn.Module):
    """Pre-norm transformer block: token mixer plus a 4x feed-forward layer, both residual."""
    def __init__(s, attn):
        """Build the block around a token mixer.

        Parameters
        ----------
        attn : torch.nn.Module
            The token mixer (full attention or Physics-Attention).
        """
        super().__init__(); s.n1, s.n2, s.attn = nn.LayerNorm(DIM), nn.LayerNorm(DIM), attn
        s.ff = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(), nn.Dropout(DROP), nn.Linear(4 * DIM, DIM))
        s.do = nn.Dropout(DROP)
    def forward(s, x): 
        """Apply the block.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N, DIM)
            Node features.

        Returns
        -------
        x : torch.Tensor, shape (B, N, DIM)
            Updated node features.
        """
        x = x + s.do(s.attn(s.n1(x))); return x + s.ff(s.n2(x))

class FullAttn(nn.Module):
    """Full multi-head self-attention over all nodes."""
    def __init__(s): 
        """Build the attention layer (width DIM, HEADS heads)."""
        super().__init__(); s.a = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
    def forward(s, x): 
        """Self-attention over the nodes.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N, DIM)
            Node features.

        Returns
        -------
        y : torch.Tensor, shape (B, N, DIM)
            Attended features.
        """
        return s.a(x, x, x, need_weights=False)[0]

class TokenNet(nn.Module):                      # transformer and transolver share everything but the mixer
    """Transformer and Transolver on the nodes of the shell.

    Node tokens are built from the position and w, pass LAYERS blocks, and feed a per-node field
    head and an attention-pooled kappa head.
    """
    def __init__(s, kind):
        """Build the token network.

        Parameters
        ----------
        kind : str, {'transformer', 'transolver'}
            Token mixer: full attention or Physics-Attention.
        """
        super().__init__()
        s.inp = nn.Sequential(nn.Linear(4 + (3 if A.prefix_inc else 0), DIM), nn.GELU(), nn.Linear(DIM, DIM))
        mk = (lambda: FullAttn()) if kind == "transformer" else (lambda: PhysicsAttention(DIM, HEADS, SLICES))
        s.blocks = nn.Sequential(*[Block(mk()) for _ in range(LAYERS)])
        s.field = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, 3))
        s.q = nn.Parameter(torch.randn(1, 1, DIM) * 0.02); s.pool = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        s.kappa = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM), nn.GELU(), nn.Linear(DIM, 1))
    def forward(s, w):
        """Predict the field and kappa of a batch of shells.

        Parameters
        ----------
        w : torch.Tensor, shape (B, N, c)
            Normalised input per node (w, and the optional prefix channels).

        Returns
        -------
        field : torch.Tensor, shape (B, N, 3)
            Normalised displacement field.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        h = s.blocks(s.inp(torch.cat([Pt[None].expand(w.shape[0], -1, -1), w], -1)))
        g, _ = s.pool(s.q.expand(w.shape[0], -1, -1), h, h); return s.field(h), s.kappa(g[:, 0]).squeeze(-1)

class MLP(nn.Module):
    """MLP on the list of defects (the 20 or 80 deepest defects and their count), with a field head and a kappa head."""
    def __init__(s, h=None):
        """Build the MLP.

        Parameters
        ----------
        h : int, default=None
            Hidden width (overridden by --hidden or --width).
        """
        h = A.hidden or A.width or h or (1024 if LARGE else 512)
        L = A.blocks or 3
        super().__init__()
        c = []
        for k in range(L): c += [nn.Linear(X.shape[1] if k == 0 else h, h), nn.GELU(), nn.Dropout(DROP)]
        s.body = nn.Sequential(*c)
        s.kappa = nn.Linear(h, 1); s.field = nn.Linear(h, NN * 3)
    def forward(s, x): 
        """Predict the field and kappa from the defect features.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_features)
            Normalised defect features.

        Returns
        -------
        field : torch.Tensor, shape (B, N, 3)
            Normalised displacement field.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        h = s.body(x); return s.field(h).view(-1, NN, 3), s.kappa(h).squeeze(-1)

class DeepONet(nn.Module):
    """DeepONet: a branch network on w and a trunk network on the node position.

    The inner product of the two gives the field; kappa is read from the branch output.
    """
    def __init__(s, p=None, h=None):
        """Build the branch and trunk networks.

        Parameters
        ----------
        p : int, default=None
            Number of basis functions per component.
        h : int, default=None
            Width of the branch network.
        """
        h = A.width or h or (2048 if LARGE else 1024)
        super().__init__(); p = p or A.p or (256 if LARGE else 128)
        L = A.depth or 3
        def _mlp(ent, larg, sai, n):
            """A plain MLP with GELU and dropout.

            Parameters
            ----------
            ent : int
                Input width.
            larg : int
                Hidden width.
            sai : int
                Output width.
            n : int
                Number of linear layers.

            Returns
            -------
            mlp : torch.nn.Sequential
                The network.
            """
            c = [nn.Linear(ent, larg), nn.GELU(), nn.Dropout(DROP)]
            for _ in range(max(0, n - 2)): c += [nn.Linear(larg, larg), nn.GELU(), nn.Dropout(DROP)]
            return nn.Sequential(*(c + [nn.Linear(larg, sai)]))
        s.branch = _mlp(NN, h, 3 * p, L)
        s.trunk = _mlp(3, 256, p, L)
        s.bias = nn.Parameter(torch.zeros(3)); s.kappa = nn.Sequential(nn.Linear(3 * p, 256), nn.GELU(), nn.Linear(256, 1)); s.p = p
    def forward(s, w):
        """Predict the field and kappa of a batch of shells.

        Parameters
        ----------
        w : torch.Tensor, shape (B, N, 1)
            Normalised w per node.

        Returns
        -------
        field : torch.Tensor, shape (B, N, 3)
            Normalised displacement field.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        b = s.branch(w[:, :, 0]); t = s.trunk(Pt)
        return torch.einsum("bcp,np->bnc", b.view(-1, 3, s.p), t) / s.p ** 0.5 + s.bias, s.kappa(b).squeeze(-1)

class MGN(nn.Module):
    """MeshGraphNet-style graph network on the k-nearest-neighbour graph of the nodes.

    Edge and node updates with residual connections, a per-node field decoder, and a kappa head on
    the mean and max of the node features.
    """
    def __init__(s, h=None, L=None, k=None):
        """Build the graph and the network.

        Parameters
        ----------
        h : int, default=None
            Hidden width.
        L : int, default=None
            Number of message-passing blocks.
        k : int, default=None
            Neighbours per node.
        """
        h = A.hidden or h or (192 if LARGE else 128); L = A.blocks or L or (10 if LARGE else 8); k = A.knn or k or 8
        super().__init__(); from scipy.spatial import cKDTree
        _, nb = cKDTree(P).query(P, k=k + 1); src = np.repeat(np.arange(NN), k); dst = nb[:, 1:].ravel()
        s.register_buffer("src", torch.tensor(src)); s.register_buffer("dst", torch.tensor(dst))
        rel = P[dst] - P[src]; s.register_buffer("efeat0", torch.tensor(np.c_[rel, np.linalg.norm(rel, axis=1)], dtype=torch.float32))
        s.enc_n = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h)); s.enc_e = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, h))
        s.eu = nn.ModuleList([nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h)) for _ in range(L)])
        s.nu = nn.ModuleList([nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h)) for _ in range(L)])
        s.dec = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3)); s.kappa = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(s, w):
        """Predict the field and kappa of a batch of shells.

        Parameters
        ----------
        w : torch.Tensor, shape (B, N, c)
            Normalised input per node (w, and the optional prefix channels).

        Returns
        -------
        field : torch.Tensor, shape (B, N, 3)
            Normalised displacement field.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        Bn = w.shape[0]; hn = s.enc_n(torch.cat([Pt[None].expand(Bn, -1, -1), w], -1)); he = s.enc_e(s.efeat0)[None].expand(Bn, -1, -1)
        for eu, nu in zip(s.eu, s.nu):
            he = he + eu(torch.cat([he, hn[:, s.src], hn[:, s.dst]], -1))
            agg = torch.zeros(hn.shape, dtype=he.dtype, device=hn.device).index_add_(1, s.dst, he)   # edge dtype: bf16/f32 mix under autocast
            hn = hn + nu(torch.cat([hn, agg], -1))
        return s.dec(hn), s.kappa(torch.cat([hn.mean(1), hn.amax(1)], -1)).squeeze(-1)

class SFNONet(nn.Module):
    """Spherical Fourier Neural Operator (torch-harmonics) on the latitude-longitude grid of the hemisphere.

    The fourth output channel, integrated over the area, gives kappa.
    """
    def __init__(s):
        """Build the operator for the grid size in use."""
        super().__init__()
        kw = {}
        if A.modes:
            import inspect as _i
            _p = _i.signature(SFNO.__init__).parameters
            for _n in ("num_modes", "modes", "n_modes", "num_freq"):
                if _n in _p: kw[_n] = A.modes; break
        s.op = SFNO(img_size=(GH, GWD), in_chans=3, out_chans=4,
                                        embed_dim=A.embed or (192 if LARGE else 128),
                                        num_layers=A.layers or (8 if LARGE else 6), scale_factor=2, **kw)
    def forward(s, x): 
        """Predict the field on the grid and kappa.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, GH, GWD)
            Normalised w and the two grid coordinates.

        Returns
        -------
        field : torch.Tensor, shape (B, 3, GH, GWD)
            Normalised field on the grid.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        y = s.op(x); return y[:, :3], (y[:, 3] * area).sum((1, 2))

class FigConv(nn.Module):
    """FigConvNet (NVIDIA PhysicsNeMo FIGConvUNet) on the point cloud of the nodes, with its scalar head for kappa."""
    def __init__(s):
        """Build the FIGConvUNet with the channel widths of the configuration."""
        super().__init__()
        base = [96, 192, 192, 192] if LARGE else [64, 128, 128, 128]
        if A.width_mult: base = [max(16, int(round(c * A.width_mult))) for c in base]
        if A.levels: base = (base + [base[-1]] * A.levels)[:max(2, A.levels + 1)]
        s.op = FIGConvUNet(in_channels=1, out_channels=3, kernel_size=3, hidden_channels=base, num_levels=A.levels or 3, mlp_channels=[512, 512] if LARGE else [256, 256],
                           aabb_min=(-1.0, -1.0, -1.0), aabb_max=(1.0, 1.0, 1.0), has_input_features=True, use_scalar_output=True)
    def forward(s, x): 
        """Predict the field and kappa of a batch of shells.

        Parameters
        ----------
        x : torch.Tensor, shape (B, N, 1)
            Normalised w per node.

        Returns
        -------
        field : torch.Tensor, shape (B, N, 3)
            Normalised displacement field.
        kappa : torch.Tensor, shape (B,)
            Normalised knockdown factor.
        """
        f, k = s.op(Pt[None].expand(x.shape[0], -1, -1).contiguous(), x); return f, k.squeeze(-1)

net = {"transformer": lambda: TokenNet("transformer"), "transolver": lambda: TokenNet("transolver"), "mgn": MGN, "deeponet": DeepONet,
       "mlp": MLP, "mlp80": MLP, "sfno": SFNONet, "figconv": FigConv}[A.model]().to(dev)
ema = copy.deepcopy(net).eval()
for p in ema.parameters(): p.requires_grad_(False)
#   betas  0.9/0.95 -> 0.9/0.999
#   warmup min(2000, steps//10) = 10% -> 5%
#   cosseno ate zero -> ate min_lr = 0.01 * lr
opt = torch.optim.AdamW(net.parameters(), A.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=A.wd)
steps = A.epochs * ((n_tr + A.bs - 1) // A.bs); warm = max(1, int(0.05 * steps))
MIN_F = 0.01
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
    min(1.0, s / warm) if s < warm else
    MIN_F + (1 - MIN_F) * 0.5 * (1 + np.cos(np.pi * min(1, (s - warm) / max(1, steps - warm))))))
start, best = 0, -np.inf; best_pair = (float("nan"), float("nan"))
if os.path.exists(CK) and not A.smoke:
    ck = torch.load(CK, map_location=dev, weights_only=False); net.load_state_dict(ck["net"]); ema.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start, best = ck["epoch"], ck["best"]; best_pair = ck.get("best_pair", (0.0, 0.0))
    if ck.get("rng"): rng.bit_generator.state = ck["rng"]; print("resumed at epoch %d (best %.4f)" % (start, best), flush=True)
NPAR = sum(p.numel() for p in net.parameters()) / 1e6
print("%s params: %.2fM  steps: %d  micro %d  lr %g  size %s  seed %d" % (A.model, NPAR, steps, MICRO, A.lr, A.size, A.seed), flush=True)
if A.params_only: sys.exit(0)
DECAY = 0.0 if A.smoke else 0.999

def _rel_r2(a, b):
    """R2 of the published scorer: the reference is the SCALAR mean of a, as in bliss.metrics._r2.

    Parameters
    ----------
    a : numpy.ndarray
        True values.
    b : numpy.ndarray
        Predicted values.

    Returns
    -------
    r2 : float
        R^2 against the scalar mean of a.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())


def evaluate(model, idx):
    """Predict a set of shells with a model and compute the selection metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The network (usually the EMA copy).
    idx : numpy.ndarray
        Indices of the shells in the loaded arrays.

    Returns
    -------
    r2f, r2d, r2k : float
        Field, dimple-region and kappa R^2.
    ang : numpy.ndarray
        Site error per shell (deg).
    pf, pk : numpy.ndarray
        Predicted fields (mm) and knockdown factors.
    r2b, r2bd : float
        Buckling R^2 and its dimple-region version (the selection metric).
    """
    pf, pk = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), 8):
            a, b = model(X[idx[i:i + 8]]); pf.append(to_nodes(a.float())); pk.append(b.float())
    pf = torch.cat(pf).cpu().numpy() * us; pk = torch.cat(pk).cpu().numpy() * ks + km
    y, k = U[idx], K[idx]
    r2f = 1 - ((pf - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
    amp = np.linalg.norm(y, axis=2); m = amp > 0.1 * amp.max(1, keepdims=True)
    r2d = 1 - ((pf - y)[m] ** 2).sum() / ((y - y.mean(0))[m] ** 2).sum()
    r2k = 1 - ((pk - k) ** 2).sum() / ((k - k.mean()) ** 2).sum()
    ur_t = (y * P).sum(-1).mean(1); ur_p = (pf * P).sum(-1).mean(1)
    Dt = y - ur_t[:, None, None] * P; Dp = pf - ur_p[:, None, None] * P
    r2b = _rel_r2(Dt, Dp)
    _md = np.linalg.norm(Dt, axis=-1); _md = _md > 0.1 * _md.max(1, keepdims=True)
    r2bd = _rel_r2(Dt[_md], Dp[_md])
    jt = amp.argmax(1); jp = np.linalg.norm(pf, axis=2).argmax(1)
    ang = np.degrees(np.arccos(np.clip((P[jt] * P[jp]).sum(1), -1, 1)))
    return r2f, r2d, r2k, ang, pf, pk, r2b, r2bd

def dump(name, idx, pf, pk, ang, ep):
    """Write the predictions of a set of shells to <OUT>/<TAG>_<name>_pred.npz.

    Parameters
    ----------
    name : str, {'val', 'test', 'train'}
        Which part.
    idx : numpy.ndarray
        Indices of the shells.
    pf : numpy.ndarray
        Predicted fields (mm).
    pk : numpy.ndarray
        Predicted knockdown factors.
    ang : numpy.ndarray
        Site errors (deg).
    ep : int
        Epoch of the weights (-1 for the final test).
    """
    np.savez("%s/%s_%s_pred.npz" % (OUT, TAG, name), ids=ids[idx], pred_U=pf, pred_K=pk, true_U=U[idx], true_K=K[idx], ang=ang, P=P, epoch=ep, r=d["R"][idx])

#
_QN = 0 if (not A.query_nodes or (A.model != "sfno" and A.query_nodes >= NN)) else A.query_nodes
if _QN:
    _tot = (GH * GWD) if A.model == "sfno" else NN
    _que = "grid cells" if A.model == "sfno" else "nodes"
    print("field loss on %d of %d %s per epoch (query_seed = 2027000 + epoch)"
          % (_QN, _tot, _que), flush=True)
if A.model == "sfno":
    _w = np.repeat(np.sin(beta) + 1e-3, GWD).astype(np.float64)
    _w = _w / _w.sum()
    _NG = len(_w)

def query(ep):
    """Nodes (or grid cells) that carry the field loss in one epoch, drawn with the seed 2027000 + epoch.

    Parameters
    ----------
    ep : int
        Epoch.

    Returns
    -------
    idx : torch.Tensor or None
        Sorted indices, or None to use all of them.
    """
    if not _QN: return None
    r = np.random.default_rng(2027000 + ep)
    if A.model == "sfno":
        if _QN >= _NG: return None
        idx = r.choice(_NG, _QN, replace=False, p=_w)
    else:
        idx = r.choice(NN, _QN, replace=False)
    return torch.tensor(np.sort(idx), device=dev, dtype=torch.long)

t0 = time.time()
for ep in range(start, A.epochs):
    net.train(); p = rng.permutation(tr); tot = 0; Q = query(ep)
    for i in range(0, n_tr, A.bs):
        idx = p[i:i + A.bs]; opt.zero_grad(set_to_none=True)
        for j in range(0, len(idx), MICRO):
            sub = idx[j:j + MICRO]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f, k = net(X[sub])
                if SW is None:
                    _f, _y = f.float(), target_field(sub)
                    if Q is not None:
                        if A.model == "sfno":
                            _f = _f.reshape(len(sub), _f.shape[1], -1)[..., Q]
                            _y = _y.reshape(len(sub), _y.shape[1], -1)[..., Q]
                        else:
                            _f = _f.reshape(len(sub), NN, -1)[:, Q]
                            _y = _y.reshape(len(sub), NN, -1)[:, Q]
                    loss = F.mse_loss(_f, _y) + 0.3 * F.mse_loss(k.float(), Yk[sub])
                    if SITES is not None:
                        # the winner must score above the runner-up: a choice between two nodes, which is
                        # what a squared loss on the whole field never asks the model to make
                        have = [(n, SITES[int(j)]) for n, j in enumerate(sub.tolist()) if int(j) in SITES]
                        if have:
                            n_, pairs = zip(*have)
                            amp = f.float().reshape(len(sub), NN, -1).norm(dim=-1)
                            a1 = amp[list(n_), [p[0] for p in pairs]]
                            a2_ = amp[list(n_), [p[1] for p in pairs]]
                            loss = loss + A.decision_weight * F.softplus(a2_ - a1).mean()
                else:
                    sw = torch.tensor([SW[int(j)] for j in sub], device=dev, dtype=torch.float32)
                    def per(a_, b_):
                        """Per-shell mean squared error on the query nodes, for the weighted loss.

                        Parameters
                        ----------
                        a_, b_ : torch.Tensor
                            Prediction and target.

                        Returns
                        -------
                        mse : torch.Tensor, shape (B,)
                            Mean squared error per shell.
                        """
                        if Q is not None:
                            if A.model == "sfno":
                                a_ = a_.reshape(len(sw), a_.shape[1], -1)[..., Q]
                                b_ = b_.reshape(len(sw), b_.shape[1], -1)[..., Q]
                            else:
                                a_ = a_.reshape(len(sw), NN, -1)[:, Q]
                                b_ = b_.reshape(len(sw), NN, -1)[:, Q]
                        return (a_ - b_).reshape(len(sw), -1).pow(2).mean(1)
                    loss = (per(f.float(), target_field(sub)) * sw).mean() + 0.3 * (per(k.float(), Yk[sub]) * sw).mean()
            (loss * len(sub) / len(idx)).backward(); tot += loss.item() * len(sub)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(DECAY).add_(pn.detach(), alpha=1 - DECAY)
    if ep % A.val_every == A.val_every - 1 or ep == A.epochs - 1:
        r2f, r2d, r2k, ang, pf, pk, r2b, r2bd = evaluate(ema, va)
        print("ep %3d  loss %.4f | EMA val: field R2 %.4f  dimple %.4f  buckling %.4f  kappa %.4f | site median %5.1f deg  <10deg %3.0f%% | %.0fs"
              % (ep + 1, tot / n_tr, r2f, r2d, r2bd, r2k, np.median(ang), 100 * np.mean(ang < 10), time.time() - t0), flush=True)
        open(OUT + "/curves.jsonl", "a").write(json.dumps(dict(
            tag=TAG, model=A.model, split=A.split, nodes=A.nodes, seed=A.seed, epoch=ep + 1,
            field_r2=float(r2f), dimple_r2=float(r2d), buckling_r2=float(r2b),
            buckling_dimple_r2=float(r2bd), kappa_r2=float(r2k),
            loc_median=float(np.median(ang)), loc10=float(np.mean(ang < 10)),
            train_loss=float(tot / n_tr), seconds=int(time.time() - t0))) + "\n")
        SEL = r2bd if A.select == "buckling" else r2d
        if SEL > best:
            best = SEL; best_pair = (r2d, r2bd); torch.save(ema.state_dict(), "%s/%s_best_ema.pt" % (OUT, TAG)); dump("val", va, pf, pk, ang, ep + 1)
        if not A.smoke:
            torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, best=best, best_pair=best_pair, rng=rng.bit_generator.state), CK)
ema.load_state_dict(torch.load("%s/%s_best_ema.pt" % (OUT, TAG), weights_only=False))
if A.dump_val:                                    # inference only, to rebuild a lost val artefact
    _a, _b, _c, vag, vpf, vpk, _d, _e = evaluate(ema, va)
    dump("val", va, vpf, vpk, vag, int(torch.load(CK, map_location="cpu", weights_only=False)["epoch"]))
if A.no_test:
    r2f = r2d = r2k = r2b = r2bd = float("nan"); ang = np.array([float("nan")])
else:
    r2f, r2d, r2k, ang, pf, pk, r2b, r2bd = evaluate(ema, te); dump("test", te, pf, pk, ang, -1)
if A.dump_train:                                  # inference only, the selected weights, no training touched
    _a, _b, _c, tang, tpf, tpk, _d, _e = evaluate(ema, tr); dump("train", tr, tpf, tpk, tang, -1)
    print("train predictions written for %d shells" % len(tr), flush=True)
vz = np.load("%s/%s_val_pred.npz" % (OUT, TAG)); vy, vk, vang = vz["true_U"], vz["true_K"], vz["ang"]
vr2k = 1 - ((vz["pred_K"] - vk) ** 2).sum() / ((vk - vk.mean()) ** 2).sum()
import subprocess as _sp
_prov = dict(torch=torch.__version__, cuda=torch.version.cuda, numpy=np.__version__,
             gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
             precision="bf16", effective_batch=A.bs, micro_batch=MICRO,
             grad_accum=max(1, A.bs // max(1, MICRO)),
             query_seed_rule="2027000 + epoch", val_every=A.val_every,
             peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0)
try:
    _prov["git_commit"] = _sp.check_output(["git", "-C", B, "rev-parse", "--short", "HEAD"],
                                           stderr=_sp.DEVNULL, text=True).strip()
except Exception:
    _prov["git_commit"] = None
rec = dict(tag=TAG, model=A.model, split=A.split, lr=A.lr, wd=A.wd, size=A.size, seed=A.seed,
           provenance=_prov,
           hp={k: v for k, v in vars(A).items() if k in ("dim","layers","heads","slices","dropout","hidden","blocks","knn","embed","modes","levels","width_mult","width","depth","p") and v}, epochs=A.epochs, frac=A.frac, nodes=A.nodes, query_nodes=int(_QN), n_train=int(n_tr), params_M=round(NPAR, 2), best_epoch=int(vz["epoch"]),
           select=A.select, val_dimple_r2=float(best_pair[0]), val_buckling_dimple_r2=float(best_pair[1]),
           val_selected=float(best), val_kappa_r2=float(vr2k), val_loc10=float(np.mean(vang < 10)), test_field_r2=float(r2f), test_dimple_r2=float(r2d),
           test_buckling_r2=float(r2b), test_buckling_dimple_r2=float(r2bd), test_kappa_r2=float(r2k), test_loc_median=float(np.median(ang)), test_loc10=float(np.mean(ang < 10)), train_s=int(time.time() - t0), smoke=A.smoke)
open(OUT + "/results.jsonl", "a").write(json.dumps(rec) + "\n")
print("\nTEST %s: field R2 %.4f  dimple %.4f  buckling %.4f  kappa %.4f | site median %.1f deg  <10deg %.0f%%\nDONE" % (TAG, r2f, r2d, r2bd, r2k, np.median(ang), 100 * np.mean(ang < 10)))
