"""Neighbour probe and count-matched cut on the axially compressed cylinder: inference of the sparse-to-dense (B3)
field models on the prepared probe shells, and their scores.

    python benchmark/cylinders/probe_cylinders.py infer MODEL [--seed 0] [--device cpu|cuda] [--protocol B3-cyl]
                                                  [--data outputs/cyl_nodes_8192.npz] [--split-file splits/cylinders/B3.json]
                                                  [--train-args "MODEL cyl ..."] [--trainer benchmark/field_models/train.py]
                                                  [--probe outputs/cyl_probe_cut_8192.npz] [--ckpt-dir outputs] [--out-dir outputs/probe_cylinders]
                                                  [--check 16] [--batch 1] [--limit N] [--topup]
    python benchmark/cylinders/probe_cylinders.py score [--in-dir outputs/probe_cylinders] [--pred-dir outputs] [--ckpt-dir outputs]
                                                  [--split-file splits/cylinders/B3.json]
                                                  [--probe outputs/cyl_probe_cut_8192.npz] [--data outputs/cyl_nodes_8192.npz] [--n-boot 2000]
                                                  [--out outputs/probe_cylinders.json]

No training. The weights are the finals of the B3 runs, <ckpt-dir>/<TAG>_best_ema.pt (the EMA state
benchmark/field_models/train.py saved at the best validation epoch).

PROBE SHELLS (one file on the 8,192 nodes of the training arrays of benchmark/cylinders/prepare_cylinders.py, with
its keys ids, W, K, P, NODES and `source`, the dense test shell each probe shell was built from). Ids carry the role
of the shell:
  neighbour probe  triples built from one dense test shell i: FULL0 + i the dense shell itself, SINGLE0 + i its
                   deepest dimple alone, NB0 + i the deepest dimple with its neighbours only.
  cut              CUT0 + i, a dense test shell (id in `source`) with dimples removed until its number of dimples
                   matches that of the sparse shells: the count is matched, the spacing stays that of a dense shell.

INFER. The same code as training: the network and the training statistics (wm, ws, km, ks over the training shells)
come from executing the trainer itself, from the top up to the line that builds the EMA copy, with the command line of
the run. By default that command line is rebuilt from the configuration of the family (configs/cylinders_<MODEL>.json:
the frozen hyperparameters and the tag of --protocol) with --data, --split-file and --seed; --train-args gives it
verbatim when the run was made with other arguments. The only edit to the executed source is the device. The
probe shells must sit on the nodes of the training arrays (P and NODES are checked); the input is X = (W - wm) / ws
and kappa = head * ks + km, as train.evaluate. cpu runs in fp32, cuda with bf16 autocast.
  --check N   also predicts the last N shells of the run's stored test predictions and prints the largest difference,
              to show the weights were loaded correctly (cpu fp32 against the stored cuda bf16 values).
  --topup     keeps the stored predictions of the shells already in the output file (same checkpoint, labels checked
              unchanged) and predicts only the shells added to the probe file since.
  --limit N   first N probe shells only, a plumbing test; the output is tagged and is not read by `score`.
Output: <out-dir>/<TAG>_probecut[_cuda].npz (ids, true_K, pred_K, source, ckpt, argv, device, check arrays). ckpt is
the file name of the checkpoint and argv the arguments of the trainer with every path reduced to its file name; no
folder is stored. The optional flag RERUN of the probe file (True where the label of a shell comes from a rerun
record, as in prepare_cylinders.py) is carried through; a probe file that stores it under the key S15 is also read.

SCORE. Only final checkpoints are scored: a run counts as final when its test predictions exist (the trainer writes
them once, after the last epoch); predictions older than their checkpoint (looked up by its file name in --ckpt-dir),
or that do not cover the current probe file, are skipped.
  neighbour probe  per triple, r_nb = kappa(NB) / kappa(SINGLE) and r_full = kappa(FULL) / kappa(SINGLE), for the
                   solver and for the model. A ratio below 1 means the neighbours lower the collapse load of the
                   deepest dimple. Reported: median, share below 1, 5th and 95th percentiles of the model's ratio,
                   Spearman correlation between the model's and the solver's ratios over the triples, median
                   |r_model - r_solver|; and the model's kappa error on the single-dimple shells.
  cut              E = median kappa error in percent (100 |kappa_pred / kappa_true - 1|, the kappa_rel_err of
                   bliss.metrics.score). E_cs on the cut shells; E_sp and E_de on the sparse and dense test shells of
                   the run. Spacing share S = (E_cs - E_sp) / (E_de - E_sp): the part of the sparse-to-dense error
                   that remains when the number of dimples is matched. 95 % interval by shell bootstrap, --n-boot
                   (2,000) draws, the cut, sparse and dense shells resampled independently, one generator of seed 0
                   shared by the checkpoints in sorted order. Also E_de on the dense sources of the cut shells
                   (S_sources) and S with E_sp from the validation shells (S_valsp).
  per family       the mean over the scored seeds.
  solver           the same share on the true kappa, (median K_cut - median K_sparse) / (median K_dense - median
                   K_sparse) with the test shells of the split, and the probe ratios of the solver alone.
Output: <out> (default <BLISS_ROOT>/outputs/probe_cylinders.json): checkpoints, families, solver_share, solver_probe.

Functions
---------
log
    Print a message with the time.
kappa_error
    Relative error of the knockdown factor per shell, in percent.
train_argv
    Command line of a run of the trainer, from the configuration of its family.
stored_argv
    Arguments of the trainer as stored with the predictions: every path reduced to its file name.
build_head
    Execute the trainer up to the EMA copy: arguments, data, training statistics, network.
predict_kappa
    Knockdown factor of the given shells, by the forward pass of train.evaluate.
infer
    Predict the probe shells with one checkpoint (stage `infer`).
probe_block
    Statistics of the neighbour probe for one set of predictions.
cut_block
    Statistics of the count-matched cut for one checkpoint.
score
    Score the stored probe predictions (stage `score`).
main
    Parse the command line and run the stage.
"""
import argparse, glob, json, os, re, shlex, sys, time
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs")
sys.path.insert(0, B)
FULL0, SINGLE0, NB0, CUT0 = 1400000, 1500000, 1600000, 1700000   # first id of each role of the probe shells


def log(*a):
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def kappa_error(pred, true):
    """Relative error of the knockdown factor per shell, in percent.

    Parameters
    ----------
    pred, true : array_like, shape (S,)
        Predicted and true knockdown factors.

    Returns
    -------
    err : numpy.ndarray, shape (S,)
        100 |pred / true - 1|.
    """
    return np.abs(np.asarray(pred) / np.asarray(true) - 1) * 100


# ---------------------------------------------------------------------------------------------- inference
def train_argv(fam, protocol, data, split_file, seed, config_dir, trainer):
    """Command line of a run of the trainer, from the configuration of its family.

    Parameters
    ----------
    fam : str
        Model family.
    protocol : str
        Protocol of the run, a key of the configuration's `protocols` (gives the tag).
    data : str
        Training arrays of the cylinder shells.
    split_file : str
        Split file of the protocol.
    seed : int
        Training seed.
    config_dir : str
        Folder of the configurations (cylinders_<family>.json: task, epochs, select, winner_params, protocols).
    trainer : str
        Path of the trainer; search_space.py is read from its folder.

    Returns
    -------
    argv : list of str
        Arguments of the trainer, the script name first.
    """
    J = json.load(open("%s/cylinders_%s.json" % (config_dir, fam))); sys.path.insert(0, os.path.dirname(os.path.abspath(trainer)))
    import search_space as hp_space
    return ([os.path.basename(trainer), fam, J.get("task", "cyl"), "--data", data, "--split-file", split_file, "--epochs", str(J.get("epochs", 200)),
             "--select", J.get("select", "buckling"), "--tag", J["protocols"][protocol]["tag"]] + hp_space.as_args(fam, J["winner_params"]) + ["--seed", str(seed)])


def stored_argv(argv):
    """Arguments of the trainer as stored with the predictions: every path reduced to its file name.

    Parameters
    ----------
    argv : list of str
        Command line of the run, the script name first.

    Returns
    -------
    args : str
        The arguments after the script name, joined by spaces; an argument that holds a path separator is replaced
        by its last component.
    """
    return " ".join(os.path.basename(x.rstrip("/\\")) if ("/" in x or os.sep in x) else x for x in argv[1:])


def build_head(argv, trainer, device):
    """Execute the trainer up to the EMA copy: arguments, data, training statistics, network.

    Parameters
    ----------
    argv : list of str
        Command line of the run, the script name first.
    trainer : str
        Path of the trainer.
    device : str
        Torch device.

    Returns
    -------
    ns : dict
        Namespace of the executed head (TAG, ids, W, P, d, T, wm, ws, km, ks, net).
    """
    src = open(trainer).read(); head = src[:src.index("ema = copy.deepcopy(net).eval()")]
    rx = re.compile(r"\bdev = [^;\n]+")
    assert len(rx.findall(head)) == 1, "the device assignment was not found exactly once in " + trainer
    head = rx.sub(lambda m: "dev = %r" % device, head)
    sys.path.insert(0, os.path.dirname(os.path.abspath(trainer)))
    old = sys.argv; sys.argv = list(argv); ns = {"__name__": "__main__", "__file__": os.path.abspath(trainer)}
    try: exec(compile(head, trainer, "exec"), ns)
    finally: sys.argv = old
    return ns


def predict_kappa(ns, W, device, batch=1):
    """Knockdown factor of the given shells, by the forward pass of train.evaluate.

    Parameters
    ----------
    ns : dict
        Namespace of build_head, with the weights loaded.
    W : numpy.ndarray, shape (S, N)
        Surface deviation at the nodes of the training arrays.
    device : str
        Torch device (bf16 autocast on cuda, fp32 on cpu).
    batch : int, default=1
        Shells per forward pass.

    Returns
    -------
    kappa : numpy.ndarray, shape (S,)
        Predicted knockdown factors.
    """
    import torch
    T, net, out = ns["T"], ns["net"], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        for i in range(0, len(W), batch):
            x = T(np.ascontiguousarray((W[i:i + batch] - ns["wm"]) / ns["ws"]).astype(np.float32))[:, :, None]
            out.append(net(x)[1].float().cpu().numpy())
    return np.concatenate(out) * ns["ks"] + ns["km"]


def infer(a):
    """Predict the probe shells with one checkpoint (stage `infer`).

    Parameters
    ----------
    a : argparse.Namespace
        Command-line options.
    """
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    argv = ([os.path.basename(a.trainer)] + shlex.split(a.train_args) if a.train_args
            else train_argv(a.model, a.protocol, a.data, a.split_file, a.seed, a.config_dir, a.trainer))
    ns = build_head(argv, a.trainer, a.device); TAG = ns["TAG"]; net = ns["net"]
    ckpt = "%s/%s_best_ema.pt" % (a.ckpt_dir, TAG); tp_f = "%s/%s_test_pred.npz" % (a.ckpt_dir, TAG)
    assert os.path.exists(ckpt), "checkpoint missing: " + ckpt
    net.load_state_dict(torch.load(ckpt, map_location=a.device, weights_only=False)); net.eval()
    log("%s: argv %s" % (TAG, " ".join(argv[1:])))
    log("  stats wm %.5g ws %.5g km %.5g ks %.5g" % (ns["wm"], ns["ws"], ns["km"], ns["ks"]))
    z = np.load(a.probe)
    assert np.allclose(z["P"], ns["P"]) and (z["NODES"] == ns["d"]["NODES"]).all(), "the probe shells are not on the nodes of the training arrays"
    n = len(z["ids"]) if not a.limit else min(a.limit, len(z["ids"]))
    save = dict(ids=z["ids"][:n], true_K=z["K"][:n], source=z["source"][:n], ckpt=os.path.basename(ckpt), argv=stored_argv(argv), device=a.device)
    for k in ("RERUN", "S15"):                           # optional flag of the probe file, carried through as RERUN
        if k in z.files: save["RERUN"] = z[k][:n]; break
    if a.check and os.path.exists(tp_f):
        tp = np.load(tp_f); pos = {int(i): j for j, i in enumerate(ns["ids"])}
        cid = tp["ids"][-a.check:]; pc = predict_kappa(ns, ns["W"][[pos[int(i)] for i in cid]], a.device, a.batch)
        save.update(chk_ids=cid, chk_pred=pc, chk_gpu=tp["pred_K"][-a.check:])
        log("  check vs stored test predictions (%d shells): max |dK| %.2e, median rel %.2e"
            % (len(cid), np.abs(pc - tp["pred_K"][-a.check:]).max(), np.median(np.abs(pc / tp["pred_K"][-a.check:] - 1))))
    os.makedirs(a.out_dir, exist_ok=True)
    out_f = "%s/%s_probecut%s%s.npz" % (a.out_dir, TAG, "" if a.device == "cpu" else "_cuda", "_limit%d" % a.limit if a.limit else "")
    if a.topup and os.path.exists(out_f):
        o = np.load(out_f); om = {int(i): j for j, i in enumerate(o["ids"])}
        assert os.path.basename(str(o["ckpt"])) == os.path.basename(ckpt) and os.path.getmtime(ckpt) < os.path.getmtime(out_f), "the stored predictions are not of this checkpoint"
        old_j = [j for j in range(n) if int(z["ids"][j]) in om]; new_j = [j for j in range(n) if int(z["ids"][j]) not in om]
        assert np.array_equal(o["true_K"][[om[int(z["ids"][j])] for j in old_j]], z["K"][old_j]), "labels of the stored shells changed"
        pk = np.zeros(n); pk[old_j] = o["pred_K"][[om[int(z["ids"][j])] for j in old_j]]
        if new_j: pk[new_j] = predict_kappa(ns, z["W"][new_j], a.device, a.batch)
        save["topup_ids"] = z["ids"][new_j]
        log("  topup: kept %d, predicted %d new (%s)" % (len(old_j), len(new_j), z["ids"][new_j].tolist()))
    else:
        pk = predict_kappa(ns, z["W"][:n], a.device, a.batch)
    save["pred_K"] = pk
    log("  probe and cut: %d shells, median rel err %.2f %%" % (len(pk), np.median(kappa_error(pk, z["K"][:n]))))
    np.savez(out_f, **save); log("wrote", out_f)


# ---------------------------------------------------------------------------------------------- scores
def probe_block(ids, pk, tk):
    """Statistics of the neighbour probe for one set of predictions.

    Parameters
    ----------
    ids : array_like, shape (S,)
        Ids of the probe shells.
    pk, tk : array_like, shape (S,)
        Predicted and true knockdown factors.

    Returns
    -------
    out : dict
        nb and full (n, solver and model median and share below 1, model p5 and p95, spearman, median_abs_err),
        single_err_median_pct, single_pred_median.
    """
    from scipy.stats import spearmanr
    m = {int(i): j for j, i in enumerate(ids)}; out = {}
    I = [i - SINGLE0 for i in sorted(m) if SINGLE0 <= i < NB0 and all(o + i - SINGLE0 in m for o in (FULL0, NB0))]
    for nm, o in (("nb", NB0), ("full", FULL0)):
        rs = np.array([tk[m[o + i]] / tk[m[SINGLE0 + i]] for i in I]); rm = np.array([pk[m[o + i]] / pk[m[SINGLE0 + i]] for i in I])
        out[nm] = dict(n=len(I), solver_median=float(np.median(rs)), solver_frac_below1=float(np.mean(rs < 1)),
                       model_median=float(np.median(rm)), model_frac_below1=float(np.mean(rm < 1)),
                       model_p5_p95=[float(np.percentile(rm, 5)), float(np.percentile(rm, 95))],
                       spearman=float(spearmanr(rm, rs)[0]), median_abs_err=float(np.median(np.abs(rm - rs))))
    out["single_err_median_pct"] = float(np.median([kappa_error(pk[m[SINGLE0 + i]], tk[m[SINGLE0 + i]]) for i in I]))
    out["single_pred_median"] = float(np.median([pk[m[SINGLE0 + i]] for i in I]))
    return out


def cut_block(z, tp, vp, regime, rng, n_boot=2000):
    """Statistics of the count-matched cut for one checkpoint.

    Parameters
    ----------
    z : mapping
        Probe predictions (ids, pred_K, true_K, source).
    tp, vp : mapping
        Test and validation predictions of the run (ids, pred_K, true_K).
    regime : dict
        {id: 'sparse' | 'dense'} of the test shells.
    rng : numpy.random.Generator
        Generator of the bootstrap, shared by the checkpoints.
    n_boot : int, default=2000
        Bootstrap draws.

    Returns
    -------
    cut : dict
        Numbers of shells, E_cs, E_sp, E_de, S and its 95 % interval, E_de_sources, S_sources, E_sp_val, S_valsp,
        medians of the predicted and true kappa of the cut shells.
    """
    ids, pk, tk = z["ids"], z["pred_K"], z["true_K"]
    rg = np.array([regime[int(i)] for i in tp["ids"]]); et = kappa_error(tp["pred_K"], tp["true_K"])
    esp_a, ede_a = et[rg == "sparse"], et[rg == "dense"]
    cm = ids >= CUT0; ecs_a = kappa_error(pk[cm], tk[cm]); src = z["source"][cm]
    tpos = {int(i): j for j, i in enumerate(tp["ids"])}; ede_src = et[[tpos[int(s)] for s in src]]
    Ecs, Esp, Ede = (float(np.median(v)) for v in (ecs_a, esp_a, ede_a)); Esv = float(np.median(kappa_error(vp["pred_K"], vp["true_K"])))
    bs = []
    for _ in range(n_boot):
        c, s, d = (np.median(rng.choice(x, len(x))) for x in (ecs_a, esp_a, ede_a)); bs.append((c - s) / (d - s))
    return dict(n_cut=int(cm.sum()), n_sparse_test=int(len(esp_a)), n_dense_test=int(len(ede_a)), E_cs=Ecs, E_sp=Esp, E_de=Ede,
                S=(Ecs - Esp) / (Ede - Esp), S_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                E_de_sources=float(np.median(ede_src)), S_sources=(Ecs - Esp) / (float(np.median(ede_src)) - Esp),
                E_sp_val=Esv, S_valsp=(Ecs - Esv) / (Ede - Esv),
                cut_pred_median=float(np.median(pk[cm])), cut_true_median=float(np.median(tk[cm])))


def score(a):
    """Score the stored probe predictions (stage `score`).

    Parameters
    ----------
    a : argparse.Namespace
        Command-line options.
    """
    S = json.load(open(a.split_file))
    if "test_regime" in S: reg = {int(k): v for k, v in S["test_regime"].items()}
    else: reg = {**{int(i): "sparse" for i in S["test_sparse"]}, **{int(i): "dense" for i in S["test_dense"]}}
    rng = np.random.default_rng(0)
    ZC = np.load(a.probe); zcp = {int(i): j for j, i in enumerate(ZC["ids"])}
    res = {"checkpoints": {}, "families": {}, "probe_file": dict(n=int(len(ZC["ids"])))}
    for f in sorted(glob.glob(a.in_dir + "/*_probecut.npz")):
        z = np.load(f); tag = os.path.basename(f).replace("_probecut.npz", "")
        tp_f = "%s/%s_test_pred.npz" % (a.pred_dir, tag)
        if not os.path.exists(tp_f): log("skip (not final, no test predictions):", tag); continue
        ck = "%s/%s" % (a.ckpt_dir, os.path.basename(str(z["ckpt"])))
        if os.path.exists(ck) and os.path.getmtime(ck) > os.path.getmtime(f): log("skip (checkpoint newer than predictions):", tag); continue
        ids, pk, tk = z["ids"], z["pred_K"], z["true_K"]
        if not np.array_equal(np.sort(ids), np.sort(ZC["ids"])) or not np.array_equal(tk, ZC["K"][[zcp[int(i)] for i in ids]]):
            log("skip (predictions do not cover the current probe file):", tag); continue
        tp = np.load(tp_f); vp = np.load("%s/%s_val_pred.npz" % (a.pred_dir, tag))
        r = {"ckpt": os.path.basename(str(z["ckpt"])), "device": str(z["device"]), "probe": probe_block(ids, pk, tk)}
        if "chk_pred" in z.files: r["load_check_max_abs_dK"] = float(np.abs(z["chk_pred"] - z["chk_gpu"]).max())
        r["cut"] = cut_block(z, tp, vp, reg, rng, a.n_boot); res["checkpoints"][tag] = r
    fams = {}
    for tag in res["checkpoints"]: fams.setdefault(tag.split("_")[0], []).append(tag)
    def mean_of(tags, path):
        """Mean over checkpoints of one entry of their results.

        Parameters
        ----------
        tags : list of str
            Checkpoints.
        path : tuple of str
            Keys leading to the entry.

        Returns
        -------
        mean : float
            Mean over the checkpoints.
        """
        v = []
        for t in tags:
            x = res["checkpoints"][t]
            for k in path: x = x[k]
            v.append(x)
        return float(np.mean(v))
    for fam, tags in fams.items():
        F = {"seeds": tags}
        for p in ("nb", "full"):
            F[p] = {k: mean_of(tags, ("probe", p, k)) for k in ("model_median", "model_frac_below1", "spearman", "median_abs_err")}
        F["cut"] = {k: mean_of(tags, ("cut", k)) for k in ("E_cs", "E_sp", "E_de", "S", "S_sources", "S_valsp")}
        res["families"][fam] = F
    if os.path.exists(a.data):                           # the same share on the true kappa
        zm = np.load(a.data); mpos = {int(i): j for j, i in enumerate(zm["ids"])}; K = zm["K"]; cm = ZC["ids"] >= CUT0
        ksp = np.median([K[mpos[i]] for i, v in reg.items() if v == "sparse"]); kde = np.median([K[mpos[i]] for i, v in reg.items() if v == "dense"])
        kcut = np.median(ZC["K"][cm]); ksrc = np.median([K[mpos[int(i)]] for i in ZC["source"][cm]])
        res["solver_share"] = dict(K_cut_median=float(kcut), K_sparse_test_median=float(ksp), K_dense_test_median=float(kde),
                                   share=float((kcut - ksp) / (kde - ksp)), K_cut_sources_median=float(ksrc),
                                   share_vs_sources=float((kcut - ksp) / (ksrc - ksp)))
    else: log("no training arrays at %s: the solver share is skipped" % a.data)
    res["solver_probe"] = probe_block(ZC["ids"], ZC["K"], ZC["K"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
    for t, r in res["checkpoints"].items():
        p, c = r["probe"], r["cut"]
        print("%-26s nb: med %.3f <1 %.2f rho %+.2f MAE %.3f | full: med %.3f <1 %.2f rho %+.2f MAE %.3f | E_sp %.2f E_cs %.2f E_de %.2f S %.2f [%.2f, %.2f]"
              % (t, p["nb"]["model_median"], p["nb"]["model_frac_below1"], p["nb"]["spearman"], p["nb"]["median_abs_err"],
                 p["full"]["model_median"], p["full"]["model_frac_below1"], p["full"]["spearman"], p["full"]["median_abs_err"],
                 c["E_sp"], c["E_cs"], c["E_de"], c["S"], *c["S_ci95"]))
    print("solver:", res.get("solver_share"), {k: res["solver_probe"][k]["solver_median"] for k in ("nb", "full")})
    log("wrote", a.out)


def main():
    """Parse the command line and run the stage."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("infer", help="predict the probe shells with one checkpoint")
    p.add_argument("model"); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--protocol", default="B3-cyl", help="protocol of the run in configs/cylinders_<model>.json")
    p.add_argument("--data", default=H + "/cyl_nodes_8192.npz", help="training arrays of the cylinder shells")
    p.add_argument("--split-file", default=B + "/splits/cylinders/B3.json")
    p.add_argument("--train-args", default="", help="the trainer arguments of the run, verbatim (replaces --protocol, --data, --split-file, --seed and the configuration)")
    p.add_argument("--trainer", default=B + "/benchmark/field_models/train.py"); p.add_argument("--config-dir", default=B + "/configs")
    p.add_argument("--probe", default=H + "/cyl_probe_cut_8192.npz", help="probe shells on the nodes of the training arrays")
    p.add_argument("--ckpt-dir", default=H, help="folder of <TAG>_best_ema.pt and <TAG>_test_pred.npz")
    p.add_argument("--out-dir", default=B + "/outputs/probe_cylinders")
    p.add_argument("--check", type=int, default=16, help="stored test shells predicted again as a loading check; 0 = none")
    p.add_argument("--batch", type=int, default=1, help="shells per forward pass"); p.add_argument("--limit", type=int, default=0, help="first N probe shells only (plumbing test)")
    p.add_argument("--topup", action="store_true", help="predict only the shells missing from the stored output")
    p = sub.add_parser("score", help="score the stored probe predictions")
    p.add_argument("--in-dir", default=B + "/outputs/probe_cylinders", help="folder of the <TAG>_probecut.npz files")
    p.add_argument("--pred-dir", default=H, help="folder of <TAG>_test_pred.npz and <TAG>_val_pred.npz")
    p.add_argument("--ckpt-dir", default=H, help="folder of <TAG>_best_ema.pt, for the check that the predictions are newer than their checkpoint")
    p.add_argument("--split-file", default=B + "/splits/cylinders/B3.json"); p.add_argument("--probe", default=H + "/cyl_probe_cut_8192.npz")
    p.add_argument("--data", default=H + "/cyl_nodes_8192.npz", help="training arrays of the cylinder shells (ids, K), for the solver share")
    p.add_argument("--n-boot", type=int, default=2000); p.add_argument("--out", default=B + "/outputs/probe_cylinders.json")
    a = ap.parse_args()
    infer(a) if a.stage == "infer" else score(a)


if __name__ == "__main__":
    main()
