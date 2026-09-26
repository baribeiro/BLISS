"""The B3 few-shot rung on the v2 release: fewshot.py re-pointed to the v2 B3 checkpoints and split.

    python benchmark/analyses/fewshot.py --stage split                                   # CPU, seconds: splits/v2/regime_fewshot.json
    python benchmark/analyses/fewshot.py --stage adapt --models figconv,sfno,transformer # GPU, resumable, one json per cell
    python benchmark/analyses/fewshot.py --stage refs                                    # trivial + solver k-1 on the same test set
    python benchmark/analyses/fewshot.py --stage zeroshot --models ...                  # GPU, minutes: zero-shot predictions for recal
    python benchmark/analyses/fewshot.py --stage adapt --draws 5 --draw-ks 5,10          # r3: >= 5 draws per model seed for k <= 10
    python benchmark/analyses/fewshot.py --stage refs2 --draws 5                         # CPU: dense mean, kNN-3, recalibration
    python benchmark/analyses/fewshot.py --stage aggregate --draws 5                     # heavy/fewshot_v2.json

WHAT CHANGED FROM fewshot.py (v1), and nothing else:
  split     BLISS-1.0/splits/B3.json (= splits/v2/regime.json). The v2 test set has no twin exclusions (the 600
            duplicates are physically out of the canonical records), so the pool is 200 of the 2,200 B3 test
            shells drawn with the same seed (POOL_SEED), test = the other 2,000. New file splits/v2/regime_fewshot.json;
            the v1 splits/regime_fewshot.json is not touched.
  data      the canonical arrays the v2 models were trained on (heavy/canon_nodes_8192.npz; SFNO: canon_grid.npz).
  model     the B3 finals of transfer.py (runs/final_runs/<fam>_B3_n8192/<fam>_v2regime_pb3buck[_s<s>]
            _best_ema.pt, the frozen B1 winner config). Built by executing train.py itself with the exact
            transfer.py command line (interventions_predict.build_head), so architecture, B3 training statistics and inputs
            are the training ones.
  seeds     draw seed s uses model seed s (three model seeds x three pool draws, paired), so the spread over
            the three cells of a k is training + draw variance; k = 0 is scored once per model seed.
  optimiser train.py's v2 recipe (AdamW betas 0.9/0.999, eps 1e-8, weight decay = the run's wd) at one
            tenth of the run's lr; the budget is fewshot.py's (5 warm-up steps then cosine to 0, batch min(k, 8),
            60 passes over the k shells capped at 400 steps, no EMA, also scored at half the budget). Loss =
            train's: MSE(field) + 0.3 MSE(kappa), for SFNO on 8,192 area-weighted grid cells per step
            (the v2 query-node rule), for the node models on all 8,192 nodes (as their training did).
  scorer    bliss.metrics.score_ci on the reduced test set, released q, no twin dropping (none exist in v2).
Artefacts: heavy/fewshot_v2_cells/<cell>.json (resume unit), heavy/fewshot_v2.json.
"""
import argparse, json, math, os, sys, time, zipfile
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"
sys.path.insert(0, B); sys.path.insert(0, B + "/benchmark/analyses")
from bliss.metrics import score, score_ci, released_q, excluded_test_counts   # noqa: E402

SPLIT_FILE = B + "/splits/v2/regime_fewshot.json"
POOL_SEED = 20260918; DRAW_SEED0 = 1000; POOL_N = 200
EPOCHS, MAX_STEPS, WARM, LR_SCALE, BS_MAX = 60, 400, 5, 0.1, 8
METRICS = ("buckling_dimple_r2", "kappa_rel_err", "loc_dominant_within_10", "two_way_winner",
           "kappa_r2", "buckling_r2", "field_r2", "loc_dominant_median", "loc_deg_median", "loc_within_10", "same_winner")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--stage", default="all", choices=["split", "adapt", "refs", "zeroshot", "refs2", "aggregate", "all"])
ap.add_argument("--draws", type=int, default=1, help="draws per model seed for the k in --draw-ks (r3: >= 5 for k <= 10); "
                "draw 0 is the original nested draw, draws d >= 1 use default_rng(DRAW_SEED0 + seed + 100 d)")
ap.add_argument("--draw-ks", default="5,10")
ap.add_argument("--models", default="figconv,sfno,transformer")
ap.add_argument("--ks", default="5,10,25,50"); ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--smoke", action="store_true", help="4 steps and a 64-shell test set, to check the plumbing")
A = ap.parse_args()
KS = [int(x) for x in A.ks.split(",") if x]; DRAW_KS = [int(x) for x in A.draw_ks.split(",") if x]; SEEDS = [int(x) for x in A.seeds.split(",") if x != ""]
MODELS = [m for m in A.models.split(",") if m]
CELLS = H + "/fewshot_v2_cells" + ("_smoke" if A.smoke else "")
os.makedirs(CELLS, exist_ok=True)


def log(*a): 
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build_split():
    """Write the few-shot split: B3, with a pool of dense shells for adaptation taken out of the test set."""
    b3 = json.load(open(B + "/BLISS-1.0/splits/B3.json")); v2 = json.load(open(B + "/splits/v2/regime.json"))
    assert all(sorted(b3[k]) == sorted(v2[k]) for k in ("train", "val", "test")), "B3.json != splits/v2/regime.json"
    ex = excluded_test_counts("v2/regime"); assert ex["leaked"] == 0 and ex["duplicate_in_test"] == 0, ex
    scored = sorted(int(s) for s in b3["test"])
    pool = sorted(np.random.default_rng(POOL_SEED).choice(np.array(scored), POOL_N, replace=False).tolist())
    test = sorted(set(scored) - set(pool))
    out = dict(train=b3["train"], val=b3["val"], test=test, adapt_pool=pool, derived_from="BLISS-1.0/splits/B3.json",
               pool_seed=POOL_SEED, n_scored_b3=len(scored),
               note=("B3 (v2) with an adaptation pool: %d dense shells drawn uniformly with seed %d from the %d B3 test "
                     "shells; test is the other %d; train and val are B3's, unchanged." % (POOL_N, POOL_SEED, len(scored), len(test))))
    if os.path.exists(SPLIT_FILE):
        old = json.load(open(SPLIT_FILE))
        assert old["test"] == out["test"] and old["adapt_pool"] == out["adapt_pool"], "split file on disk differs"
    else:
        json.dump(out, open(SPLIT_FILE, "w"), indent=1)
    log("split: train %d val %d pool %d test %d -> %s" % (len(out["train"]), len(out["val"]), len(pool), len(test), SPLIT_FILE))
    return out


def draw(pool, k, seed, d=0):
    """Draw k adaptation shells from the pool.

    Parameters
    ----------
    pool : list of int
        Adaptation pool.
    k : int
        Number of shells.
    seed : int
        Model seed.
    d : int
        Draw index.

    Returns
    -------
    ids : list of int
        Sorted shell ids.
    """
    return sorted(np.random.default_rng(DRAW_SEED0 + seed + 100 * d).permutation(np.array(pool))[:k].tolist())


def cell_names(model, k, s):
    """(name, draw) of every cell of (model, k, model seed): draw 0 keeps the original name.

    Parameters
    ----------
    model : str
        Model family.
    k : int
        Number of adaptation shells.
    s : int
        Model seed.

    Returns
    -------
    cells : list of tuple
        (name, draw) of every cell.
    """
    nd = A.draws if k in DRAW_KS else 1
    return [("%s_k%d_s%d%s" % (model, k, s, "_d%d" % d if d else ""), d) for d in range(nd)]


def npz_memmap(path, name):
    """Memory-map one uncompressed array of an .npz file.

    Parameters
    ----------
    path : str
        Path of the .npz.
    name : str
        Array name.

    Returns
    -------
    a : numpy.ndarray
        Memory-mapped array.
    """
    info = zipfile.ZipFile(path).getinfo(name + ".npy"); assert info.compress_type == 0
    with open(path, "rb") as f:
        f.seek(info.header_offset); h = f.read(30)
        f.seek(info.header_offset + 30 + int.from_bytes(h[26:28], "little") + int.from_bytes(h[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        shape, _, dt = (np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0)(f)
        off = f.tell()
    return np.memmap(path, dtype=dt, mode="r", offset=off, shape=shape)


def predict(ns, net, idx, bs=8):
    """Predict a set of shells with a network.

    Parameters
    ----------
    ns : dict
        Namespace of the executed train.py head (data, statistics, network).
    net : torch.nn.Module
        The network.
    idx : numpy.ndarray
        Indices of the shells.
    bs : int, default=8
        Batch size.

    Returns
    -------
    pred_U : numpy.ndarray
        Predicted fields (mm).
    pred_K : numpy.ndarray
        Predicted kappa.
    """
    import torch
    net.eval(); pf, pk = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(ns["dev"] == "cuda")):
        for i in range(0, len(idx), bs):
            a, b = net(ns["X"][idx[i:i + bs]]); pf.append(ns["to_nodes"](a.float()).cpu()); pk.append(b.float().cpu())
    return torch.cat(pf).numpy() * ns["us"], torch.cat(pk).numpy() * ns["ks"] + ns["km"]


def finetune(ns, net, idx, target_of, lr, wd, steps, bs, seed):
    """Fine-tune a network on k shells with the recipe of train.py (AdamW, warm-up and cosine schedule).

    Parameters
    ----------
    ns : dict
        Namespace of the executed train.py head (data, statistics, network).
    net : torch.nn.Module
        The network.
    idx : numpy.ndarray
        Indices of the adaptation shells.
    target_of : callable
        Normalised field target of a set of indices.
    lr : float
        Learning rate.
    wd : float
        Weight decay.
    steps : int
        Optimisation steps.
    bs : int
        Batch size.
    seed : int
        Seed.

    Returns
    -------
    net : torch.nn.Module
        The fine-tuned network.
    """
    import torch, torch.nn.functional as F
    opt = torch.optim.AdamW(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1, max(0, s - WARM) / max(1, steps - WARM)))))
    rng = np.random.default_rng(7000 + seed); order = np.array([], int); loss_last = float("nan")
    sfno = ns["A"].model == "sfno"; dev = ns["dev"]
    for s in range(steps):
        if len(order) < bs: order = np.concatenate([order, rng.permutation(len(idx))])
        sub = idx[order[:bs]]; order = order[bs:]
        net.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            f, k = net(ns["X"][sub]); f, y = f.float(), target_of(sub)
            if sfno:                                  # the v2 rule: 8,192 area-weighted grid cells per step
                q = torch.tensor(np.sort(rng.choice(len(ns["_w"]), 8192, replace=False, p=ns["_w"])), device=dev)
                f = f.reshape(len(sub), f.shape[1], -1)[..., q]; y = y.reshape(len(sub), y.shape[1], -1)[..., q]
            loss = F.mse_loss(f, y) + 0.3 * F.mse_loss(k.float(), ns["Yk"][sub])
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        loss_last = float(loss.item())
        if s + 1 in (steps // 2, steps): yield s + 1, loss_last


def metrics_of(ns, pred_U, pred_K, idx, boot):
    """Scorer metrics of a prediction on a set of shells.

    Parameters
    ----------
    ns : dict
        Namespace of the executed train.py head (data, statistics, network).
    pred_U : numpy.ndarray
        Predicted fields.
    pred_K : numpy.ndarray
        Predicted kappa.
    idx : numpy.ndarray
        Indices of the shells.
    boot : int
        Bootstrap resamples.

    Returns
    -------
    metrics : dict
        score_ci metrics.
    """
    ids = ns["ids"][idx]
    z = dict(ids=ids, P=ns["P"], true_U=ns["d"]["U"][idx], pred_U=pred_U.astype(np.float32),
             true_K=ns["d"]["K"][idx], pred_K=pred_K, r=ns["d"]["R"][idx])
    if boot:
        r = score_ci(z, n_boot=boot, drop_leaked=False)
        out = {m: dict(point=v["point"], lo=v["lo"], hi=v["hi"]) for m, v in r["metrics"].items() if m in METRICS}
        out["_n"] = dict(n=r["n_shells"], dominant=r["n_dominant"], ambiguous=r["n_ambiguous"], q_source=r["q_source"])
        return out
    sc = score(z["true_U"], z["pred_U"], z["true_K"], z["pred_K"], np.asarray(z["P"], np.float64), z["r"], released_q(ids))
    return {m: dict(point=float(sc[m])) for m in METRICS if m in sc}


def brief(m):
    """One-line summary of the headline metrics.

    Parameters
    ----------
    m : dict
        Metrics.

    Returns
    -------
    text : str
        Summary.
    """
    g = lambda n: m.get(n, {}).get("point", float("nan"))
    return "buck %.3f  kap %.2f%%  loc10 %.0f%%  win %.0f%%" % (g("buckling_dimple_r2"), g("kappa_rel_err"),
                                                                100 * g("loc_dominant_within_10"), 100 * g("two_way_winner"))


def cp(name): 
    """Path of the result file of one cell.

    Parameters
    ----------
    name : str
        Cell name.

    Returns
    -------
    path : str
        JSON path.
    """
    return "%s/%s.json" % (CELLS, name)


def stage_adapt():
    """Fine-tune every model, k and seed on the adaptation shells and score it on the few-shot test set."""
    import torch
    import interventions_predict as ES
    sp = json.load(open(SPLIT_FILE)); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    for model in MODELS:
        specs = {}
        for s in SEEDS:
            x, why = ES.spec(model, "b3", s)
            if x is None: log("SKIP %s s%d: %s" % (model, s, why))
            else: specs[s] = x
        if not specs: continue
        t0 = time.time(); ns = ES.build_head(specs[min(specs)]["argv"], model, dev)
        assert ns["A"].split == "v2/regime" and not ns["A"].grid
        ns["d"] = dict(U=ns["U"], K=ns["K"], R=np.asarray(ns["d"]["R"]))     # in memory, not re-read from the npz
        pos = {int(i): j for j, i in enumerate(ns["ids"])}
        tr_ok = sorted(ns["ids"][ns["tr"]].tolist()) == sorted(sp["train"]); assert tr_ok, "B3 train set differs"
        te = np.array([pos[i] for i in sp["test"]]); pool = sp["adapt_pool"]
        te_full = np.array([pos[i] for i in sorted(set(sp["test"]) | set(pool))])
        if A.smoke: te_full = te_full[:64]; te = np.array([j for j in te if j in set(te_full.tolist())])
        if model == "sfno":
            GU = npz_memmap(H + "/canon_grid.npz", "GU"); gids = np.asarray(npz_memmap(H + "/canon_grid.npz", "ids"))
            assert np.array_equal(gids, ns["ids"])
            target_of = lambda idx: torch.tensor(np.moveaxis(np.asarray(GU[idx]) / ns["us"], -1, 1).astype(np.float32), device=dev)
            # the area weights of train's query rule (defined after the executed head, so rebuilt here)
            _w = np.repeat(np.sin(ns["beta"]) + 1e-3, ns["GWD"]).astype(np.float64); ns["_w"] = _w / _w.sum()
        else:
            target_of = lambda idx: ns["Y"][idx]
        log("%s: head built in %.0fs; lr %g wd %g" % (model, time.time() - t0, ns["A"].lr, ns["A"].wd))
        for s, x in sorted(specs.items()):
            base = torch.load(x["ckpt"], map_location=dev, weights_only=False)
            prov = dict(model=model, model_seed=s, checkpoint=x["ckpt"], tag=x["TAG"], train_lr=ns["A"].lr, train_wd=ns["A"].wd,
                        config=x["params"], split_file=SPLIT_FILE, device=dev)
            name = "%s_k0_s%d" % (model, s)
            if not os.path.exists(cp(name)):
                net = ns["net"]; net.load_state_dict(base); t1 = time.time()
                pf, pk = predict(ns, net, te_full)
                where = {int(v): i for i, v in enumerate(te_full)}; sub = np.array([where[int(j)] for j in te])
                rec = dict(prov, k=0, seed=s, shells=[], steps=0, lr=0.0,
                           final=metrics_of(ns, pf[sub], pk[sub], te, A.n_boot),
                           full_b3=metrics_of(ns, pf, pk, te_full, A.n_boot), secs=round(time.time() - t1))
                json.dump(rec, open(cp(name), "w"), indent=1, default=float)
                log("%-11s k=0   s%d reduced %s | full-B3 %s" % (model, s, brief(rec["final"]), brief(rec["full_b3"])))
            zs = "%s/zeroshot_%s_s%d.npz" % (CELLS, model, s)          # for the recalibration reference (refs2)
            if not os.path.exists(zs):
                net = ns["net"]; net.load_state_dict(base); pf, pk = predict(ns, net, te_full)
                np.savez(zs, ids=ns["ids"][te_full], pred_U=pf.astype(np.float32), pred_K=pk, checkpoint=x["ckpt"])
            for k, (name, dr) in [(k, nd) for k in KS for nd in cell_names(model, k, s)]:
                if os.path.exists(cp(name)): continue
                shells = draw(pool, k, s, dr); idx = np.array([pos[i] for i in shells])
                bs = min(k, BS_MAX); steps = 4 if A.smoke else min(EPOCHS * math.ceil(k / bs), MAX_STEPS)
                net = ns["net"]; net.load_state_dict(base); t1 = time.time()
                rec = dict(prov, k=k, seed=s, draw=dr, shells=shells, steps=steps, lr=ns["A"].lr * LR_SCALE, bs=bs,
                           epochs_equivalent=round(steps * bs / k, 1))
                for done, loss in finetune(ns, net, idx, target_of, ns["A"].lr * LR_SCALE, ns["A"].wd, steps, bs, s):
                    pf, pk = predict(ns, net, te)
                    key = "final" if done == steps else "mid"
                    rec[key] = metrics_of(ns, pf, pk, te, A.n_boot if key == "final" else 0)
                    rec[key + "_steps"] = done; rec[key + "_loss"] = loss
                rec["secs"] = round(time.time() - t1)
                json.dump(rec, open(cp(name), "w"), indent=1, default=float)
                log("%-11s k=%-3d s%d %3d steps loss %.4f %s [%ds]" % (model, k, s, steps, rec["final_loss"], brief(rec["final"]), rec["secs"]))
        del ns; torch.cuda.empty_cache()


def stage_refs():
    """Reference rows of the few-shot test set (training mean and the dense-shell means)."""
    sp = json.load(open(SPLIT_FILE)); d = np.load(H + "/canon_nodes_8192.npz"); pos = {int(i): j for j, i in enumerate(d["ids"])}
    tr = np.array([pos[i] for i in sp["train"]]); te = np.array([pos[i] for i in sp["test"]])
    if A.smoke: te = te[:64]
    U, K, P, R, NODES = d["U"], d["K"], d["P"], d["R"], d["NODES"]
    ns = dict(ids=d["ids"], P=P, d=dict(U=U, K=K, R=R))
    if not os.path.exists(cp("ref_trivial")):
        rec = dict(model="trivial", note="B3 training-part mean field and mean kappa", split_file=SPLIT_FILE,
                   final=metrics_of(ns, np.broadcast_to(U[tr].mean(0), U[te].shape), np.full(len(te), K[tr].mean()), te, A.n_boot))
        json.dump(rec, open(cp("ref_trivial"), "w"), indent=1, default=float); log("trivial     %s" % brief(rec["final"]))
    if not os.path.exists(cp("ref_solver")):
        Up = np.stack([np.load("%s/BLISS-1.0/canonical/%d.npz" % (B, int(d["ids"][j])))["U_prev"][NODES] for j in te])
        rec = dict(model="solver_k-1", note="the solver's previous Riks increment (canonical U_prev) as the prediction; "
                   "kappa is the truth, so kappa columns are not defined", split_file=SPLIT_FILE,
                   final=metrics_of(ns, Up, K[te], te, A.n_boot))
        json.dump(rec, open(cp("ref_solver"), "w"), indent=1, default=float); log("solver k-1  %s" % brief(rec["final"]))


def stage_refs2():
    """r3 competitors fitted on the SAME k shells as each adaptation cell (every seed and draw), scored on the 2,000:
    densemean  the k shells' mean field and mean kappa;
    knn3       for each test shell, the mean field and kappa of its min(3, k) nearest neighbours AMONG the k shells
               (L2 on w standardised with the B3 training statistics, the models' input scaling);
    recal      a model's zero-shot prediction (zeroshot_<model>_s<s>.npz) plus a constant kappa shift and a constant
               uniform-contraction shift, both the mean residual over the k shells (the matching contraction shift
               u_bar -> u_bar + mean(u_bar_true - u_bar_pred), applied as that scalar times P on every node)."""
    sp = json.load(open(SPLIT_FILE)); d = np.load(H + "/canon_nodes_8192.npz"); pos = {int(i): j for j, i in enumerate(d["ids"])}
    U, K, W, P, R = d["U"], d["K"], d["W"].astype(np.float64), d["P"].astype(np.float64), d["R"]
    tr = np.array([pos[i] for i in sp["train"]]); te = np.array([pos[i] for i in sp["test"]]); pool = sp["adapt_pool"]
    if A.smoke: te = te[:64]
    ns = dict(ids=d["ids"], P=d["P"], d=dict(U=U, K=K, R=R)); wm, ws = W[tr].mean(), W[tr].std(); Zte = (W[te] - wm) / ws
    ub = lambda X: (X * P).sum(-1).mean(-1)
    zs = {}
    for m in MODELS:
        for s in SEEDS:
            f = "%s/zeroshot_%s_s%d.npz" % (CELLS, m, s)
            if os.path.exists(f):
                z = np.load(f); zs[(m, s)] = ({int(i): j for j, i in enumerate(z["ids"])}, z["pred_U"], z["pred_K"])
    for k in KS:
        for s in SEEDS:
            for name, dr in cell_names("_", k, s):
                shells = draw(pool, k, s, dr); idx = np.array([pos[i] for i in shells]); tag = name[2:]      # k<k>_s<s>[_d<d>]
                if not os.path.exists(cp("r2_densemean_" + tag)):
                    rec = dict(kind="densemean", k=k, seed=s, draw=dr, shells=shells,
                               final=metrics_of(ns, np.broadcast_to(U[idx].mean(0), U[te].shape), np.full(len(te), K[idx].mean()), te, A.n_boot))
                    json.dump(rec, open(cp("r2_densemean_" + tag), "w"), indent=1, default=float)
                if not os.path.exists(cp("r2_knn3_" + tag)):
                    Zk = (W[idx] - wm) / ws; d2 = (Zte ** 2).sum(1)[:, None] + (Zk ** 2).sum(1)[None] - 2 * Zte @ Zk.T
                    nn = np.argsort(d2, 1)[:, :min(3, k)]
                    rec = dict(kind="knn3", k=k, seed=s, draw=dr, shells=shells,
                               final=metrics_of(ns, U[idx][nn].mean(1), K[idx][nn].mean(1), te, A.n_boot))
                    json.dump(rec, open(cp("r2_knn3_" + tag), "w"), indent=1, default=float)
                for m in MODELS:
                    if (m, s) not in zs or os.path.exists(cp("r2_recal_%s_%s" % (m, tag))): continue
                    zp, pu, pk = zs[(m, s)]
                    if not all(int(d["ids"][j]) in zp for j in np.r_[idx, te]): continue       # (--smoke zero-shot covers 64 shells only)
                    ik = [zp[int(d["ids"][j])] for j in idx]; it = [zp[int(d["ids"][j])] for j in te]
                    dk = float(np.mean(K[idx] - pk[ik])); du = float(np.mean(ub(U[idx].astype(np.float64)) - ub(pu[ik].astype(np.float64))))
                    rec = dict(kind="recal", model=m, k=k, seed=s, draw=dr, shells=shells, kappa_shift=dk, ubar_shift_mm=du,
                               final=metrics_of(ns, pu[it] + du * P[None].astype(np.float32), pk[it] + dk, te, A.n_boot))
                    json.dump(rec, open(cp("r2_recal_%s_%s" % (m, tag)), "w"), indent=1, default=float)
        log("refs2 k=%d done" % k)


def stage_aggregate():
    """Collect every cell into the few-shot curve and write outputs/fewshot_v2.json."""
    sp = json.load(open(SPLIT_FILE))
    cells = {f[:-5]: json.load(open(CELLS + "/" + f)) for f in sorted(os.listdir(CELLS)) if f.endswith(".json")}
    curve = {}
    for m in MODELS:
        rows = {}
        for k in [0] + KS:
            got = [cells[n] for s in SEEDS for n, _ in (cell_names(m, k, s) if k else [("%s_k0_s%d" % (m, s), 0)]) if n in cells]
            if not got: continue
            v = {me: [g["final"][me]["point"] for g in got if me in g["final"]] for me in METRICS}
            rows[str(k)] = dict(n_cells=len(got), seeds=[g["seed"] for g in got], draws=[g.get("draw", 0) for g in got],
                                mean={me: float(np.mean(x)) for me, x in v.items() if x},
                                range={me: [float(min(x)), float(max(x))] for me, x in v.items() if x},
                                per_seed=v, mid={me: [g["mid"][me]["point"] for g in got if "mid" in g and me in g["mid"]] for me in METRICS},
                                full_b3={me: [g["full_b3"][me]["point"] for g in got if "full_b3" in g and me in g["full_b3"]] for me in METRICS} if k == 0 else None)
        curve[m] = rows
    refs = {nm: {me: cells[key]["final"][me] for me in METRICS if me in cells[key]["final"]}
            for nm, key in (("trivial", "ref_trivial"), ("solver_k-1", "ref_solver")) if key in cells}
    for kind in ("densemean", "knn3", "recal"):                  # the r3 competitors, per k over seeds x draws
        for m in (MODELS if kind == "recal" else ["_"]):
            for k in KS:
                pre = ("r2_%s_k%d_" % (kind, k)) if m == "_" else ("r2_recal_%s_k%d_" % (m, k))
                got = [c for n, c in cells.items() if n.startswith(pre)]
                if not got: continue
                v = {me: [g["final"][me]["point"] for g in got if me in g["final"]] for me in METRICS}
                refs["%s%s|k=%d" % (kind, "" if m == "_" else "|" + m, k)] = dict(n_cells=len(got), mean={me: float(np.mean(x)) for me, x in v.items() if x},
                                                                          range={me: [float(min(x)), float(max(x))] for me, x in v.items() if x})
    out = dict(generated=time.strftime("%Y-%m-%d %H:%M"), script="benchmark/analyses/fewshot.py", split_file=SPLIT_FILE,
               n=dict(train=len(sp["train"]), val=len(sp["val"]), pool=len(sp["adapt_pool"]), test=len(sp["test"])),
               protocol=__doc__.split("WHAT CHANGED")[1], references=refs, curve=curve)
    fo = CELLS + "/fewshot_v2_smoke.json" if A.smoke else H + "/fewshot_v2.json"
    json.dump(out, open(fo, "w"), indent=1, default=float)
    for m, rows in curve.items():
        log("%-11s " % m + " | ".join("k=%s buck %.3f win %.2f" % (k, r["mean"].get("buckling_dimple_r2", np.nan), r["mean"].get("two_way_winner", np.nan)) for k, r in rows.items()))
    log("wrote %s" % fo)


if A.stage in ("split", "all"): build_split()
if A.stage in ("adapt", "all"): stage_adapt()
if A.stage in ("refs", "all"): stage_refs()
if A.stage in ("zeroshot",): A.ks, KS = "", []; stage_adapt()   # zero-shot predictions only (the k loop is empty)
if A.stage in ("refs2", "all"): stage_refs2()
if A.stage in ("aggregate", "all"): stage_aggregate()
