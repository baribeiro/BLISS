"""Inference of the trained B1 field models on the single-factor sweeps P1-P4 (intervention tests).

    python benchmark/analyses/interventions_predict.py build   [--variant n8192|native] [--nproc 8] [--ids 30000,70000]
    python benchmark/analyses/interventions_predict.py infer   --families mlp,deeponet,... [--seeds 0,1,2] [--variant n8192|native|b3]
                                  [--device cuda|cpu] [--ids ...] [--check-test 8] [--out-dir heavy]
                                  [--sweeps P3,P4 --p4-arm b10 --out-tag _p3p4dense]

No training. The weights are the B1 (split v2/random) finals of confirm.py: the winner configuration of
phase2_<fam>_v2random_n<nodes>_buckling<sfx>.json, three seeds, heavy/<TAG>_best_ema.pt (the EMA state
train.py saved at the best validation epoch).

  n8192   the cells of the paper table (runs/final_runs/<fam>_B1_n8192, suffix _hp2), all 8 families
  b3      the B3 (v2/regime) finals of transfer.py (final_runs/<fam>_B3_n8192, tag pb3buck, frozen B1 config)
  native  76,805-node finals trained on the full mesh (figconv, mgn, sfno, transolver; suffix ""), expected in
          runs/native_final/heavy. A missing checkpoint is SKIPPED and logged.

THE SAME CODE AS TRAINING. The model, the training statistics (wm, ws, us, km, ks over the B1 training
shells) and the canonical input tensors come from executing train.py itself, from the top up to the
line that builds the EMA copy, with the exact command line confirm.py used (search_space.as_args of the winner).
The only edits to the executed source: the device (cpu for the smoke test), and for SFNO the 7 GB grid
targets are not loaded (a forward pass does not need them; same edit as time_inference.py). The sweep inputs
are then built by the same expressions: w -> (w[NODES] - wm) / ws for the node models; the 180x720 grid by
the release map (release/shared.npz knn_idx/knn_w, geometry.to_grid, the same map prepare_data.py used)
for SFNO; the top-K defect table for mlp/mlp80. Forward pass as train.evaluate(): batches of 8, bf16
autocast on cuda (fp32 on cpu), to_nodes, pred * us, kappa * ks + km.

SWEEP RECORDS (built once, cached in heavy/sweep_inputs.npz). For every run of cells/{p1,p2,p3,p5}/runs.csv
(p5 = the new P4): w = |X0| - R on the 76,805-node base mesh (checked against release/shared.npz, the
cell_to_release.py test), the limit-point field U at the first limit point (geometry.first_limit on the
path, the release rule), kappa_true = LPF[first limit] * p_ref / p_classical from reports/<id>_path.npz
(E_catalogue = 1.25, the release convention), cross-checked against the field file's kd * L[k] / max(L)
(the cell_to_release rule). The one P2 run without a field (40102, a db = 0 arm, duplicate of core 2443)
takes its input from the core record and is flagged input_from=core:<src>. The source core records of every
sweep are added as sweep=core rows (arm=source), read from BLISS-1.0/canonical/<src>.npz, so each group has
its unmoved reference even where the sweep has no 0 arm (P1, P3).

SITE. D = U - mean_nodes(U . P) P (the uniform radial contraction out, as tools/truth_dominance.py and
bliss.metrics.score), site = argmax |D|. True site on the full mesh (true_node, 76,805 numbering) and on
the model's node set (true_node_sub); predicted site on the model's node set, reported in the 76,805
numbering. Angles in the hemisphere's own frame (geometry.frame_of): beta = polar angle from the axis,
theta = azimuth, degrees. site_err_deg = angle between predicted and true (full mesh) site.

Output: <out-dir>/sweep_eval_<family>_<variant>_s<seed>.csv, one row per record; and
<out-dir>/sweep_eval_<family>_<variant>_s<seed>_fields.npz (pred_U at the model's nodes, float16; n8192 only).

Functions
---------
log
    Print a message with the time.
build
    Build the cache of sweep inputs and truths (stage `build`).
spec
    Checkpoint, tag and command line of one trained model.
build_head
    Execute train.py up to the EMA copy: args, data, training statistics, inputs, network.
sweep_inputs
    The sweep records in the model's input format, by the same expressions train.py applies.
forward
    train.evaluate(), without the scoring: batches of 8, bf16 autocast on cuda, to_nodes, de-normalise.
infer
    Predict every sweep record with the trained B1 (or B3) models and write one CSV per family and seed
    (stage `infer`).
"""
import argparse, csv, json, os, re, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); ML = B + "/benchmark/field_models"; H = B + "/outputs"
RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, ML)
SWEEPS = [("P1", "p1"), ("P2", "p2"), ("P3", "p3"), ("P4", "p5")]      # cells/p5 is the new P4
FAMILIES = ["deeponet", "figconv", "mgn", "sfno", "transformer", "transolver", "mlp", "mlp80"]
VARIANTS = {
    "n8192":  dict(nodes=8192, sfx="_hp2", fams=FAMILIES,
                   dir=lambda f: "%s/final_runs/%s_B1_n8192" % (RUNS, f)),
    "native": dict(nodes=76805, sfx="", fams=["figconv", "mgn", "sfno", "transolver"],
                   dir=lambda f: "%s/native_final/heavy" % RUNS),
    # B3 (split v2/regime) finals of transfer.py: frozen B1 winner, tag pb3buck, three seeds
    "b3":     dict(nodes=8192, sfx="", fams=FAMILIES, phase="B3",
                   dir=lambda f: "%s/final_runs/%s_B3_n8192" % (RUNS, f)),
}
CACHE = H + "/sweep_inputs.npz"
R_SHELL = 25.4


def log(*a):
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------- records
def _frame():
    """Shared mesh and the frame of the release.

    Returns
    -------
    P : numpy.ndarray
        Unit node directions.
    conn : numpy.ndarray
        Element connectivity.
    E : numpy.ndarray
        Frame (e1, e2, axis).
    """
    from bliss.geometry import frame_of
    sh = np.load(B + "/release/shared.npz"); P = sh["P"].astype(np.float64)
    return P, sh["conn"], np.stack(frame_of(P))


def _site(U, P):
    """argmax |D|, D = U minus the uniform radial contraction (truth_dominance.py).

    Parameters
    ----------
    U : numpy.ndarray
        Displacement field, shape (N, 3).
    P : numpy.ndarray
        Unit node directions.

    Returns
    -------
    node : int
        Index of the failure site.
    """
    U = U.astype(np.float64); D = U - (U * P).sum(-1).mean() * P
    return int(np.linalg.norm(D, axis=-1).argmax())


def _angles(p, E):
    """Polar and azimuthal angle of a point in the frame of the shell.

    Parameters
    ----------
    p : numpy.ndarray
        Unit vector.
    E : numpy.ndarray
        Frame (e1, e2, axis).

    Returns
    -------
    polar, azimuth : float
        Angles in degrees.
    """
    q = E @ p
    return float(np.degrees(np.arccos(np.clip(q[2], -1, 1)))), float(np.degrees(np.arctan2(q[1], q[0])))


def _parse(note):
    """Source shell and arm name of a sweep run from its note.

    Parameters
    ----------
    note : str
        Note of the run in runs.csv.

    Returns
    -------
    src : int
        Source core shell.
    arm : str
        Arm of the sweep.
    """
    t = note.split("_"); src = int(t[0][3:])
    arm = "_".join(x for x in t[1:] if not re.match(r"^(w\d+|l\d+|nfix\d+|lv[0-9.]+)$", x))
    return src, arm


def _plan(only=None):
    """Every sweep record to build, from the runs.csv of each cell.

    Parameters
    ----------
    only : set of int or None
        Restrict to these ids.

    Returns
    -------
    jobs : list of dict
        One job per record.
    srcs : dict
        Source shells per sweep.
    """
    jobs, srcs = [], {}
    for S, cell in SWEEPS:
        for r in csv.DictReader(open("%s/cells/%s/runs.csv" % (B, cell))):
            sid = int(r["shell"]); src, arm = _parse(r["note"]); srcs.setdefault(src, set()).add(S)
            if only is None or sid in only:
                jobs.append(dict(id=sid, sweep=S, cell=cell, source=src, arm=arm, note=r["note"]))
    for src in sorted(srcs):
        if only is None or src in only:
            jobs.append(dict(id=src, sweep="core", cell="", source=src, arm="source", note="+".join(sorted(srcs[src]))))
    return jobs


def _one(job):
    """Read one sweep record: inputs at the model nodes, the true kappa at the first limit point and the true site.

    Parameters
    ----------
    job : dict
        The record to build.

    Returns
    -------
    out : dict
        Inputs, truth and status of the record.
    """
    from bliss.geometry import first_limit
    P, conn, E = _G["P"], _G["conn"], _G["E"]; NODES = _G["NODES"]
    out = dict(job); out.update(input_from="field", status="ok", kappa_true=np.nan, kappa_true_field=np.nan, first_lp_inc=-1)
    if job["sweep"] == "core":
        z = np.load("%s/BLISS-1.0/canonical/%d.npz" % (B, job["id"]))
        w, U = z["w"].astype(np.float32), z["U_peak"]; out.update(kappa_true=float(z["kappa"]), kappa_true_field=float(z["kappa"]),
                                                               first_lp_inc=int(z["peak_frame"]), input_from="canonical")
    else:
        d = "%s/cells/%s" % (B, job["cell"]); f = "%s/fields/%d.npz" % (d, job["id"]); rep = "%s/reports/%d_path.npz" % (d, job["id"])
        if os.path.exists(rep):
            r = np.load(rep); L = np.asarray(r["LPF"], np.float64)
            if len(L) >= 5 and not np.all(np.diff(L) >= 0):
                k = first_limit(L); out["kappa_true"] = float(L[k] * float(r["p_ref"]) / float(r["p_classical"]))
            else:
                out["status"] = "no limit point in report"
        if os.path.exists(f):
            z = np.load(f); X0, U3, L = z["X0"], z["U"], np.asarray(z["lpf"], np.float64)
            Pn = X0 / np.linalg.norm(X0, axis=1)[:, None]
            if X0.shape[0] != len(P) or np.abs(Pn - P).max() > 1e-5 or not np.array_equal(z["conn"].astype(np.int32), conn):
                out["status"] = "mesh differs from release/shared.npz"; return out, None, None
            n = min(len(L), U3.shape[0])
            if U3.shape[0] != len(L):          # frames != increments: locate the increment's frame by arc length (cell_to_release)
                k = first_limit(L); kf = int(np.argmin(np.abs(np.asarray(z["frame_arc"]) - np.asarray(z["arc"])[k])))
            else:
                k = kf = first_limit(L[:n])
            out["first_lp_inc"] = int(k); out["kappa_true_field"] = float(z["kd"]) * float(L[k]) / float(np.max(L))
            w = (np.linalg.norm(X0, axis=1) - float(z["radius"])).astype(np.float32); U = U3[kf]
        else:                                  # P2 40102: db = 0 arm, the core record is its input and its field
            c = np.load("%s/BLISS-1.0/canonical/%d.npz" % (B, job["source"]))
            w, U = c["w"].astype(np.float32), c["U_peak"]; out["input_from"] = "core:%d" % job["source"]
            if not np.isfinite(out["kappa_true"]): out["kappa_true"] = float(c["kappa"])
            out["kappa_true_field"] = float(c["kappa"])
    j = _site(U, P); b, t = _angles(P[j], E)
    js = int(NODES[_site(U[NODES], P[NODES])])
    out.update(true_node=j, true_beta_deg=b, true_theta_deg=t, true_node_sub=js, U_max_mm=float(np.abs(U).max()))
    return out, w, np.asarray(U, np.float32)[NODES]


_G = {}


def build(a):
    """Build the cache of sweep inputs and truths (stage `build`).

    Parameters
    ----------
    a : argparse.Namespace
        Command-line options.
    """
    from multiprocessing import Pool
    P, conn, E = _frame(); NODES = np.load(H + "/canon_nodes_8192.npz")["NODES"]
    _G.update(P=P, conn=conn, E=E, NODES=NODES)
    only = set(int(x) for x in a.ids.split(",")) if a.ids else None
    jobs = _plan(only); log("building %d records (%s)" % (len(jobs), ", ".join("%s %d" % (s, sum(j["sweep"] == s for j in jobs)) for s in ["P1", "P2", "P3", "P4", "core"])))
    with Pool(a.nproc) as pool:
        res = pool.map(_one, jobs, chunksize=4)
    bad = [r[0] for r in res if r[1] is None]
    for r in bad: log("  skipped %d: %s" % (r["id"], r["status"]))
    res = [r for r in res if r[1] is not None]
    meta = [r[0] for r in res]
    rel = [abs(m["kappa_true"] / m["kappa_true_field"] - 1) for m in meta if np.isfinite(m["kappa_true"]) and m["input_from"] == "field"]
    log("kappa report vs field rule: max rel diff %.2e over %d records" % (max(rel) if rel else np.nan, len(rel)))
    cols = {k: np.array([m[k] for m in meta]) for k in meta[0]}
    np.savez(a.cache, W=np.stack([r[1] for r in res]), U8192=np.stack([r[2] for r in res]), NODES8192=NODES, **cols)
    log("wrote %s: %d records" % (a.cache, len(meta)))


# ---------------------------------------------------------------------------------------------- models
def spec(fam, variant, seed):
    """Checkpoint, tag and command line of one trained model.

    Parameters
    ----------
    fam : str
        Model family.
    variant : str
        Set of finals (n8192, native, b3).
    seed : int
        Training seed.

    Returns
    -------
    spec : dict or None
        Paths, tags and train.py arguments.
    reason : str or None
        Why the model is skipped, if it is.
    """
    V = VARIANTS[variant]; d = V["dir"](fam)
    if V.get("phase"):                                  # transfer.py: phase<P>_<fam>_n<nodes>_<select>.json
        js = "%s/phase%s_%s_n%d_buckling.json" % (d, V["phase"], fam, V["nodes"])
        if not os.path.exists(js):
            return None, "selection json missing: " + js
        J = json.load(open(js)); J.setdefault("select", "buckling"); J.setdefault("epochs", 200)
        J["winner_params"] = J["frozen_params"]; rank = -1
        tag = "p%s%s" % (V["phase"].lower(), J["select"][:4])                                    # transfer.py._tag
    else:
        js = "%s/phase2_%s_v2random_n%d_buckling%s.json" % (d, fam, V["nodes"], V["sfx"])
        if not os.path.exists(js):
            return None, "selection json missing: " + js
        J = json.load(open(js)); rank = int(J["winner_rank"])
        tag = "f%s%d_c%d%s" % (J["select"][:4], V["nodes"], rank, V["sfx"])                   # phase2._tag
    TAG = "%s_%s_%s" % (fam, J["split"].replace("/", ""), tag) + ("_s%d" % seed if seed else "")  # train TAG
    ck = "%s/%s_best_ema.pt" % (d, TAG)
    if not os.path.exists(ck):
        return None, "checkpoint missing (arriving?): " + ck
    import search_space as hp_space
    argv = ["train.py", fam, J["split"], "--epochs", str(J["epochs"]), "--select", J["select"], "--tag", tag,
            "--seed", "0"] + hp_space.as_args(fam, J["winner_params"])
    if V["nodes"] != 8192: argv += ["--nodes", str(V["nodes"])]
    return dict(fam=fam, variant=variant, seed=seed, json=js, rank=rank, tag=tag, TAG=TAG, ckpt=ck, argv=argv, split=J["split"],
                test_pred="%s/%s_test_pred.npz" % (d, TAG), params=J["winner_params"]), None


def build_head(argv, fam, device):
    """Execute train.py up to the EMA copy: args, data, training statistics, inputs, network.

    Parameters
    ----------
    argv : list of str
        train.py command line of the model.
    fam : str
        Model family.
    device : str
        Torch device.

    Returns
    -------
    ns : dict
        Namespace of the executed head (data, statistics, network).
    """
    src = open(ML + "/train.py").read(); head = src[:src.index("ema = copy.deepcopy(net).eval()")]
    def patch(s, old, new):
        """Replace one anchor of the source text, which must occur exactly once.

        Parameters
        ----------
        s : str
            Source text.
        old : str
            Anchor.
        new : str
            Replacement.

        Returns
        -------
        s : str
            Patched text.
        """
        assert s.count(old) == 1, "patch anchor not found exactly once: %r" % old[:60]; return s.replace(old, new)
    head = patch(head, 'B + "/outputs"); dev = "cuda"', 'B + "/outputs"); dev = %r' % device)
    if fam == "sfno":
        head = patch(head, 'GW, GU = g["GW"], g["GU"]', 'GW = g["GW"]; GU = None')
        head = patch(head, 'Yg = T(np.moveaxis(GU / us, -1, 1).copy()); area', 'Yg = None; area')
    sys.argv = list(argv); ns = {"__name__": "__main__", "__file__": ML + "/train.py"}
    os.environ.pop("BLISS_HEAVY", None); os.environ.pop("BLISS_ROOT", None)
    exec(compile(head, ML + "/train.py", "exec"), ns)
    return ns


def sweep_inputs(ns, C, fam):
    """The sweep records in the model's input format, by the same expressions train.py applies.

    Parameters
    ----------
    ns : dict
        Namespace from build_head.
    C : mapping
        Sweep cache.
    fam : str
        Model family.

    Returns
    -------
    X : torch.Tensor
        Model inputs of the sweep records.
    """
    import torch
    T, wm, ws = ns["T"], ns["wm"], ns["ws"]; NODES = ns["d"]["NODES"]
    if fam in ("transformer", "transolver", "mgn", "deeponet", "figconv"):
        # contiguous: W[:, NODES] comes out Fortran-ordered and FIGConvUNet .view()s its features
        return T(np.ascontiguousarray((C["W"][:, NODES] - wm) / ws))[:, :, None]
    if fam in ("mlp", "mlp80"):
        TOPK = ns["TOPK"]; D = dict(ns["D"])                       # metadata/defects.csv, as train read it
        extra = [str(C["defects_csv"])] if "defects_csv" in C else []      # rot_inputs.npz carries its own rotated tables
        for f in ["%s/cells/%s/defects.csv" % (B, cell) for S, cell in SWEEPS] + extra:
            Ds = {}
            for r in csv.DictReader(open(f)):
                Ds.setdefault(int(r["id"]), []).append((float(r["delta"]), float(r["theta"]), float(r["phi"])))
            D.update(Ds)
        keys = [int(i) if int(i) in D else int(s) for i, s in zip(C["id"], C["source"])]
        feats = np.zeros((len(keys), 3 * TOPK + 1), np.float32)
        for i, s in enumerate(keys):
            for j, (dl, th, ph) in enumerate(sorted(D[int(s)], key=lambda x: -x[0])[:TOPK]): feats[i, 3 * j:3 * j + 3] = (dl, np.cos(th) * ph, np.sin(th) * ph)
            feats[i, 3 * TOPK] = len(D[int(s)]) / 80.0
        ftr = ns["feats"][ns["tr"]]
        return T((feats - ftr.mean(0)) / (ftr.std(0) + 1e-6))
    if fam == "sfno":
        from bliss.geometry import to_grid
        A = ns["A"]; assert not A.grid, "sweep grid built at 180x720 only"
        sh = np.load(B + "/release/shared.npz"); gi, gw = sh["knn_idx"], sh["knn_w"]
        GW = np.stack([to_grid(w, gi, gw) for w in C["W"]])
        return torch.tensor(np.concatenate([((GW - wm) / ws)[:, None], np.broadcast_to(ns["coords"], (len(GW), 2, 180, 720))], 1).astype(np.float32),
                            device=ns["dev"])
    raise SystemExit("unknown family " + fam)


def forward(ns, X, net):
    """train.evaluate(), without the scoring: batches of 8, bf16 autocast on cuda, to_nodes, de-normalise.

    Parameters
    ----------
    ns : dict
        Namespace from build_head.
    X : torch.Tensor
        Inputs.
    net : torch.nn.Module
        The network with the selected EMA weights.

    Returns
    -------
    pf : numpy.ndarray
        Predicted fields on the nodes (mm).
    pk : numpy.ndarray
        Predicted kappa.
    """
    import torch
    dev = ns["dev"]; pf, pk = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
        for i in range(0, len(X), 8):
            a, b = net(X[i:i + 8]); pf.append(ns["to_nodes"](a.float()).cpu()); pk.append(b.float().cpu())
    return torch.cat(pf).numpy() * ns["us"], torch.cat(pk).numpy() * ns["ks"] + ns["km"]


def infer(a):
    """Predict every sweep record with the trained B1 (or B3) models and write one CSV per family and seed (stage `infer`).

    Parameters
    ----------
    a : argparse.Namespace
        Command-line options.
    """
    import torch
    if a.device == "cpu": torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    if not os.path.exists(a.cache): raise SystemExit("no %s: run `build` first" % a.cache)
    Z = np.load(a.cache); C = {k: Z[k] for k in Z.files}
    keep = np.ones(len(C["id"]), bool)
    if a.ids: keep &= np.isin(C["id"], [int(x) for x in a.ids.split(",")])
    if a.sweeps:                                      # e.g. P3,P4 (+ the core rows of their sources)
        sw = a.sweeps.split(",")
        m = np.isin(C["sweep"], sw)
        if a.p4_arm: m &= (C["sweep"] != "P4") | np.char.startswith(C["arm"].astype(str), a.p4_arm)
        keep &= m | ((C["sweep"] == "core") & np.isin(C["source"], C["source"][m]))
    C = {k: (v[keep] if v.ndim and len(v) == len(keep) else v) for k, v in C.items()}
    Pf, _, E = _frame()
    member = {}
    for nm in ("random", "delta", "regime"):
        sp_ = json.load(open(B + "/splits/v2/%s.json" % nm))
        member[nm] = {int(i): s for s in ("train", "val", "test") for i in sp_[s]}
    os.makedirs(a.out_dir, exist_ok=True)
    for fam in a.families.split(","):
        specs = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            sp, why = spec(fam, a.variant, seed)
            if sp is None: log("SKIP %s %s s%d: %s" % (fam, a.variant, seed, why))
            else: specs.append(sp)
        if not specs: continue
        t0 = time.time(); ns = build_head(specs[0]["argv"], fam, a.device); net = ns["net"].eval()
        NODES = ns["d"]["NODES"]; P = ns["P"]
        assert np.allclose(P, Pf[NODES].astype(np.float32)), "model node set is not release P[NODES]"
        X = sweep_inputs(ns, C, fam)
        log("%s: head built in %.0fs (%s, %d nodes), %d sweep records" % (fam, time.time() - t0, a.device, len(NODES), len(X)))
        for sp in specs:
            net.load_state_dict(torch.load(sp["ckpt"], map_location=a.device, weights_only=False)); net.eval()
            pf, pk = forward(ns, X, net)
            if a.check_test:                                   # the same pipeline reproduces the stored B1 test predictions
                te = ns["te"][:a.check_test]; tf, tk = forward(ns, ns["X"][te], net)
                if os.path.exists(sp["test_pred"]):
                    z = np.load(sp["test_pred"]); pos = {int(i): j for j, i in enumerate(z["ids"])}; jj = [pos[int(i)] for i in ns["ids"][te]]
                    dk = np.abs(tk - z["pred_K"][jj]); du = np.abs(tf - z["pred_U"][jj]).max() / np.abs(z["pred_U"][jj]).max()
                    log("  check vs %s on %d test shells: max|dkappa| %.2e (kappa ~%.3f), max|dU|/max|U| %.2e"
                        % (os.path.basename(sp["test_pred"]), len(te), dk.max(), np.abs(z["pred_K"][jj]).mean(), du))
                else:
                    log("  check skipped: no " + sp["test_pred"])
            rows = []
            for i in range(len(pk)):
                U = pf[i].astype(np.float64); D = U - (U * P).sum(-1).mean() * P; j = int(NODES[np.linalg.norm(D, axis=-1).argmax()])
                b, t = _angles(Pf[j], E); jt = int(C["true_node"][i])
                rows.append(dict(id=int(C["id"][i]), sweep=C["sweep"][i], source=int(C["source"][i]), arm=C["arm"][i], note=C["note"][i],
                                 input_from=C["input_from"][i], model_split=sp["split"],
                                 src_split_model=member[sp["split"].split("/")[1]].get(int(C["source"][i]), "none"),
                                 src_split_B1=member["random"].get(int(C["source"][i]), "none"),
                                 kappa_true=float(C["kappa_true"][i]), kappa_pred=float(pk[i]),
                                 pred_node=j, pred_beta_deg=b, pred_theta_deg=t,
                                 true_node=jt, true_beta_deg=float(C["true_beta_deg"][i]), true_theta_deg=float(C["true_theta_deg"][i]),
                                 true_node_sub=int(C["true_node_sub"][i]) if len(NODES) < len(Pf) else jt,
                                 site_err_deg=float(np.degrees(np.arccos(np.clip(Pf[j] @ Pf[jt], -1, 1)))) if jt >= 0 else float("nan"),
                                 family=fam, variant=a.variant, seed=sp["seed"], checkpoint=sp["ckpt"]))
            stem = "%s/sweep_eval_%s_%s%s_s%d" % (a.out_dir, fam, a.variant, a.out_tag, sp["seed"])
            with open(stem + ".csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            if a.variant in ("n8192", "b3") and not a.no_fields:
                np.savez(stem + "_fields.npz", ids=C["id"], pred_U=pf.astype(np.float16 if np.abs(pf).max() < 6e4 else np.float32), pred_K=pk, NODES=NODES, checkpoint=sp["ckpt"])
            log("  %s s%d -> %s.csv (%d rows) [%.0fs]" % (fam, sp["seed"], stem, len(rows), time.time() - t0))
        del ns, net, X
        if a.device == "cuda": torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "infer"])
    ap.add_argument("--families", default=",".join(FAMILIES)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variant", default="n8192", choices=list(VARIANTS)); ap.add_argument("--device", default="cuda")
    ap.add_argument("--ids", default="", help="restrict to these record ids (smoke test)")
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-dir", default=H); ap.add_argument("--check-test", type=int, default=0)
    ap.add_argument("--no-fields", action="store_true")
    ap.add_argument("--sweeps", default="", help="only these sweeps, e.g. P3,P4 (core rows of their sources kept)")
    ap.add_argument("--p4-arm", default="", help="P4 rows only whose arm starts with this, e.g. b10 (dense)")
    ap.add_argument("--out-tag", default="", help="suffix of the output file names")
    a = ap.parse_args()
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1" if a.stage == "build" else os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    build(a) if a.stage == "build" else infer(a)
