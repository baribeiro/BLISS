"""Edited shells of the controlled tests: the prepared arrays, the inference of the trained field models on them, and
the reference errors of the B3 models. Shared by count_matched.py, paired_layouts.py and neighbour_probe.py.

    python benchmark/analyses/edited_shells.py --sets cut_dense,layout_10deg,layout_25deg --variant b3
                                  [--families figconv,mgn,transformer,transolver,deeponet] [--seeds 0,1,2]
                                  [--device cuda|cpu] [--batch 8] [--edited-dir outputs/edited_shells] [--out-dir outputs]
                                  [--save-fields] [--threads 4]

No training and no solver. An edited shell is a core shell whose list of defects was changed (defects removed or
moved) and which was simulated again with the solver of the release. This script starts from the result of that
simulation, as arrays.

INPUT CONTRACT (one file per set of edited shells, <edited-dir>/<set>.npz; S shells, N = 8,192 model nodes):
  ids            (S,)       id of every edited shell
  source         (S,)       id of the core shell it was edited from; shells of different sets with the same source are
                            the paired versions of one shell
  X              (N, 3)     node positions on the sphere of radius R (mm), the node set of the field models
                            (outputs/canon_nodes_8192.npz: P[NODES]); the unit directions u = X / |X|
  W              (S, N)     imperfection w at those nodes (mm): the radial deviation of the edited list of defects,
                            w = sum_i -delta_i t exp(-(beta_i / (lambda_i / pref))^2) (interventions_inputs.rebuild)
  kappa          (S,)       knockdown factor of the solver; NaN where the path has no limit point
  U              (S, N, 3)  optional: displacement of the solver at the limit point, read at the nearest mesh node of
                            every model node (mm); rows of NaN where the field is missing
  defect_shell, defect_delta, defect_lambda, defect_theta, defect_phi   (M,)  optional: the edited list of defects,
                            one row per defect in the order of the generator (the first defect of a shell is its seed
                            defect); angles in the frame of bliss.geometry.frame, radians
The finite-element models that produce these arrays are not part of this repository.

INFERENCE. The same path as the intervention tests (interventions_predict.py: spec, build_head, sweep_inputs,
forward): the network, the training statistics and the input expression come from executing train.py with the command
line of the trained model, and the EMA weights of the best validation epoch are loaded. w enters at the model nodes
exactly as a core shell does. The node models only (figconv, mgn, transformer, transolver, deeponet): SFNO and the
MLPs read inputs these arrays do not carry (the full mesh, the table of defects of the release). --variant n8192
= the B1 finals, b3 = the B3 finals (trained on sparse shells only). On cpu the forward pass is fp32, on cuda bf16
autocast; --batch 1 keeps the memory of the transformer low on cpu and gives the same outputs per shell.

SCORES PER SHELL (stored with the prediction):
  D = U - mean_nodes(U . u) u, the field without the uniform radial contraction, for the prediction and the solver;
  field_r2 = 1 - sum (D_pred - D_true)^2 / sum (D_true - mean(D_true))^2 over the nodes and components of the shell;
  site10   = 1 if argmax |D_pred| is within 10 deg of argmax |D_true|, else 0 (bliss.metrics.loc_error_deg);
  both NaN where the solver field is missing.

REFERENCE ERRORS of a B3 model (published predictions of transfer.py, <runs>/final_runs/<fam>_B3_n8192/
<TAG>_{val,test}_pred.npz): e_sp = relative kappa error (percent) per shell of the B3 validation set (sparse shells),
e_de = the same on the dense reference set. Its default is the full B3 test set of splits/B3.json
(--dense-ids all). The scoring scripts also take --dense-ids with a json list of ids, such as
splits/analysis/controlled_tests_dense.json, read relative to BLISS_ROOT: the B3 test shells outside the few-shot
adaptation pool and outside the sources of the neighbour probe. --exclude-sets is optional: it names sets
of edited shells whose source shells are also left out, and is empty by default.

THREADS. OMP_NUM_THREADS, MKL_NUM_THREADS and OPENBLAS_NUM_THREADS are set when this module is loaded, before numpy
is imported, to SLURM_CPUS_PER_TASK if it is defined and to 4 otherwise; values already in the environment are kept.
--threads sets the number of torch threads on cpu (default: the same value).

Output: <out-dir>/edited_pred_<family>_<variant>_<set>_s<seed>.npz with ids, source, kappa_true, kappa_pred,
field_r2, site10 (and pred_U, float16, with --save-fields).

Functions
---------
log
    Print a message with the time.
relerr
    Relative error in percent.
load_set
    The prepared arrays of one set of edited shells.
deviation
    Field without the uniform radial contraction.
field_scores
    Field R^2 and site within 10 deg, per shell.
pred_file
    Path of the prediction of one family, variant, set and seed.
load_pred
    Stored predictions of one family on one set, per seed.
published_predictions
    Validation and test prediction files of the B3 finals of one family, per seed.
dense_test_ids
    Ids of the dense reference set.
reference_errors
    Relative kappa error per shell of a B3 model on its sparse validation set and on the dense test set.
seed_mean_median
    Mean over seeds of the median over shells.
seed_mean_scores
    Field R^2 and site rate of a set, averaged over seeds.
predict
    Predict every edited shell with the trained models and write one file per family, set and seed.
main
    Parse the command line and predict the edited shells.
"""
import argparse, glob, json, os, re, sys, time

THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):      # before numpy is imported
    os.environ.setdefault(_v, THREADS)
import numpy as np   # noqa: E402

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
H = os.environ.get("BLISS_HEAVY", B + "/outputs"); RUNS = os.environ.get("BLISS_RUNS", B + "/runs")
sys.path.insert(0, B); sys.path.insert(0, B + "/benchmark/analyses")
from bliss.metrics import _r2, loc_error_deg   # noqa: E402

EDITED = H + "/edited_shells"
DENSE_IDS = B + "/splits/analysis/controlled_tests_dense.json"      # an alternative dense reference set (--dense-ids)
B3_SPLIT = B + "/splits/B3.json"                                    # --dense-ids all: its "test" list
FAMILIES = ["figconv", "mgn", "transformer", "transolver", "deeponet"]      # the node models; this order fixes the bootstrap stream
SITE_DEG = 10.0


def log(*a):
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def relerr(p, t):
    """Relative error in percent.

    Parameters
    ----------
    p, t : array_like
        Prediction and truth.

    Returns
    -------
    e : numpy.ndarray
        |p / t - 1| * 100.
    """
    return np.abs(np.asarray(p, float) / np.asarray(t, float) - 1) * 100


def load_set(name, edited_dir=EDITED):
    """The prepared arrays of one set of edited shells (the input contract of the module docstring).

    Parameters
    ----------
    name : str
        Name of the set.
    edited_dir : str
        Directory with <name>.npz.

    Returns
    -------
    S : dict
        ids, source, X, u (unit directions), W, kappa, D (list of the solver's deviation fields, None where
        missing) and, if present, defects ({shell id: unit centres of its defects, in the order of the file}).
    """
    f = "%s/%s.npz" % (edited_dir, name)
    if not os.path.exists(f): raise SystemExit("no %s: the prepared arrays of the edited shells are an input (see the module docstring)" % f)
    z = np.load(f); n = len(z["ids"])
    X = z["X"].astype(float); u = X / np.linalg.norm(X, axis=1)[:, None]
    S = dict(name=name, ids=z["ids"].astype(int), source=z["source"].astype(int), X=X, u=u, W=z["W"].astype(np.float32), kappa=z["kappa"].astype(float))
    assert S["W"].shape == (n, len(u)) and len(S["kappa"]) == n and len(S["source"]) == n, "arrays of %s do not match the contract" % f
    S["D"] = [None] * n
    if "U" in z.files:
        U = z["U"]; assert U.shape == (n, len(u), 3)
        S["D"] = [deviation(U[i].astype(float), u) if np.isfinite(U[i]).all() else None for i in range(n)]
    if "defect_shell" in z.files:
        th, ph = z["defect_theta"].astype(float), z["defect_phi"].astype(float); D = {}
        for i, c in zip(z["defect_shell"].astype(int), np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th), np.cos(ph)], 1)):
            D.setdefault(int(i), []).append(c)
        S["defects"] = {k: np.array(v) for k, v in D.items()}
    return S


def deviation(U, u):
    """Field without the uniform radial contraction.

    Parameters
    ----------
    U : numpy.ndarray, shape (N, 3)
        Displacement field.
    u : numpy.ndarray, shape (N, 3)
        Unit node directions.

    Returns
    -------
    D : numpy.ndarray, shape (N, 3)
        U - mean_nodes(U . u) u.
    """
    return U - (U * u).sum(1).mean() * u


def field_scores(Dp, Ds, u):
    """Field R^2 (scalar-mean reference, per shell) and site within 10 deg, per shell.

    Parameters
    ----------
    Dp : sequence of numpy.ndarray
        Predicted deviation fields.
    Ds : sequence of numpy.ndarray or None
        Deviation fields of the solver; None where missing.
    u : numpy.ndarray, shape (N, 3)
        Unit node directions.

    Returns
    -------
    r2 : numpy.ndarray, shape (S,)
        Field R^2 per shell; NaN where the solver field is missing.
    site : numpy.ndarray, shape (S,)
        1.0 where the predicted site is within 10 deg of the solver's, 0.0 where not, NaN where missing.
    """
    r2, site = [], []
    for a, b in zip(Dp, Ds):
        if b is None: r2.append(np.nan); site.append(np.nan); continue
        r2.append(_r2(b, a))
        site.append(float(loc_error_deg(b[None], a[None], u)[0] < SITE_DEG))
    return np.array(r2), np.array(site)


def pred_file(out_dir, fam, variant, name, seed):
    """Path of the prediction of one family, variant, set and seed.

    Parameters
    ----------
    out_dir : str
        Directory.
    fam : str
        Model family.
    variant : str
        Set of finals (n8192, b3).
    name : str
        Set of edited shells.
    seed : int
        Training seed.

    Returns
    -------
    path : str
        The npz file.
    """
    return "%s/edited_pred_%s_%s_%s_s%d.npz" % (out_dir, fam, variant, name, seed)


def load_pred(in_dir, fam, variant, name, seeds=None):
    """Stored predictions of one family on one set, per seed.

    Parameters
    ----------
    in_dir : str
        Directory of the prediction files.
    fam : str
        Model family.
    variant : str
        Set of finals (n8192, b3).
    name : str
        Set of edited shells.
    seeds : iterable of int or None
        Keep these seeds; all seeds on disk if None.

    Returns
    -------
    D : dict
        Seed -> dict(ids, source, kp, KT, field, site, src).
    """
    out = {}
    for f in sorted(glob.glob(pred_file(in_dir, fam, variant, name, 0)[:-len("0.npz")] + "[0-9]*.npz")):
        s = int(re.search(r"_s(\d+)\.npz$", f).group(1))
        if seeds is not None and s not in seeds: continue
        z = np.load(f)
        out[s] = dict(ids=z["ids"], source=z["source"], kp=z["kappa_pred"], KT=z["kappa_true"], field=z["field_r2"], site=z["site10"], src=os.path.basename(f))
    return out


def published_predictions(fam, runs=RUNS):
    """Validation and test prediction files of the B3 finals of one family, per seed (written by train.py).

    Parameters
    ----------
    fam : str
        Model family.
    runs : str
        Directory of the trained models (holds final_runs/<fam>_B3_n8192).

    Returns
    -------
    files : dict
        Seed -> (validation file, test file); seeds without both files are left out.
    """
    out = {}
    for ft in sorted(glob.glob("%s/final_runs/%s_B3_n8192/*_test_pred.npz" % (runs, fam))):
        fv = ft.replace("_test_pred", "_val_pred")
        if not os.path.exists(fv): continue
        m = re.search(r"_s(\d+)_test_pred\.npz$", ft); s = int(m.group(1)) if m else 0
        assert s not in out, "more than one B3 final of %s for seed %d under %s" % (fam, s, runs)
        out[s] = (fv, ft)
    return out


def dense_test_ids(dense_ids="all", exclude_sets="", edited_dir=EDITED):
    """Ids of the dense reference set, minus the sources of the excluded sets if any are named.

    Parameters
    ----------
    dense_ids : str
        Json file with the ids, a list or a dictionary with the list under "ids" (or "test"); the word "all" takes
        the full B3 test set of splits/B3.json. A relative path that does not exist is read relative to BLISS_ROOT.
    exclude_sets : str
        Comma-separated sets of edited shells whose source shells are left out; empty for none.
    edited_dir : str
        Directory of the prepared arrays (read only when exclude_sets is not empty).

    Returns
    -------
    te : set of int
        Shell ids.
    """
    f = B3_SPLIT if dense_ids == "all" else dense_ids
    if not os.path.isabs(f) and not os.path.exists(f): f = os.path.join(B, f)
    if not os.path.exists(f): raise SystemExit("no %s: the id list of the dense reference set (--dense-ids FILE, or --dense-ids all)" % f)
    J = json.load(open(f))
    if isinstance(J, dict): J = J["test"] if dense_ids == "all" or "ids" not in J else J["ids"]
    te = set(int(i) for i in J)
    for name in [x for x in (exclude_sets or "").split(",") if x]:
        f = "%s/%s.npz" % (edited_dir, name)
        if not os.path.exists(f): raise SystemExit("no %s: named in --exclude-sets (leave the option out to exclude no shell)" % f)
        te -= set(int(i) for i in np.load(f)["source"])
    return te


def reference_errors(fam, seed, te, runs=RUNS):
    """Relative kappa error per shell of a B3 model on its sparse validation set and on the dense test set.

    Parameters
    ----------
    fam : str
        Model family.
    seed : int
        Training seed.
    te : set of int
        Ids of the dense reference set (dense_test_ids).
    runs : str
        Directory of the trained models.

    Returns
    -------
    e_sp : numpy.ndarray
        Error on the validation shells (percent).
    e_de : numpy.ndarray
        Error on the dense test shells (percent).
    """
    files = published_predictions(fam, runs)
    if seed not in files: raise SystemExit("no published B3 predictions of %s seed %d under %s/final_runs" % (fam, seed, runs))
    zv, zt = np.load(files[seed][0]), np.load(files[seed][1])
    mt = np.array([int(i) in te for i in zt["ids"]])
    return relerr(zv["pred_K"], zv["true_K"]), relerr(zt["pred_K"][mt], zt["true_K"][mt])


def seed_mean_median(rows):
    """Mean over seeds of the median over shells.

    Parameters
    ----------
    rows : sequence of numpy.ndarray
        Per-shell values, one array per seed.

    Returns
    -------
    m : float
        The statistic.
    """
    return float(np.mean([np.median(r) for r in rows]))


def seed_mean_scores(fields, sites):
    """Field R^2 and site rate of a set: mean over seeds of the median field R^2 and of the share of sites within 10 deg.

    Parameters
    ----------
    fields : sequence of numpy.ndarray
        Field R^2 per shell, one array per seed (NaN where the solver field is missing).
    sites : sequence of numpy.ndarray
        Site hits per shell, one array per seed.

    Returns
    -------
    field_r2, site10 : float
        The two statistics; NaN when no solver field was given.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows when the solver fields are not given
        return float(np.mean([np.nanmedian(x) for x in fields])), float(np.mean([np.nanmean(x) for x in sites]))


def predict(a):
    """Predict every edited shell with the trained models and write one file per family, set and seed.

    Parameters
    ----------
    a : argparse.Namespace
        Command-line options.
    """
    import torch
    import interventions_predict as IP
    if a.device == "cpu": torch.set_num_threads(a.threads)
    sets = {n: load_set(n, a.edited_dir) for n in a.sets.split(",") if n}
    os.makedirs(a.out_dir, exist_ok=True)
    for fam in a.families.split(","):
        if fam not in FAMILIES: raise SystemExit("%s does not read w at the model nodes; families: %s" % (fam, ",".join(FAMILIES)))
        specs = []
        for seed in [int(s) for s in a.seeds.split(",")]:
            sp, why = IP.spec(fam, a.variant, seed)
            if sp is None: log("SKIP %s %s s%d: %s" % (fam, a.variant, seed, why))
            else: specs.append(sp)
        if not specs: continue
        t0 = time.time(); ns = IP.build_head(specs[0]["argv"], fam, a.device); net = ns["net"].eval()
        NODES = ns["d"]["NODES"]; nfull = int(np.max(NODES)) + 1
        for name, S in sets.items():
            assert S["u"].shape == ns["P"].shape and np.abs(S["u"] - ns["P"]).max() < 1e-5, "X of %s is not the node set of the model" % name
        log("%s %s: head built in %.0fs (%s, %d nodes)" % (fam, a.variant, time.time() - t0, a.device, len(NODES)))
        for sp in specs:
            net.load_state_dict(torch.load(sp["ckpt"], map_location=a.device, weights_only=False)); net.eval()
            for name, S in sets.items():
                Wb = np.zeros((len(S["ids"]), nfull), np.float32); Wb[:, NODES] = S["W"]
                fp, kp = IP.forward(ns, IP.sweep_inputs(ns, {"W": Wb}, fam), net, bs=a.batch); fp = np.asarray(fp, float)
                if fp.ndim == 3 and fp.shape[1] != len(NODES) and fp.shape[1] == nfull: fp = fp[:, NODES]
                kp = np.asarray(kp, float).ravel()
                r2, site = field_scores([deviation(x, S["u"]) for x in fp], S["D"], S["u"])
                out = dict(ids=S["ids"], source=S["source"], kappa_true=S["kappa"], kappa_pred=kp, field_r2=r2, site10=site)
                if a.save_fields: out["pred_U"] = fp.astype(np.float16 if np.abs(fp).max() < 6e4 else np.float32)
                f = pred_file(a.out_dir, fam, a.variant, name, sp["seed"]); np.savez(f, **out)
                log("  %s s%d %s: median kappa error %.2f %% -> %s [%.0fs]" % (fam, sp["seed"], name, np.nanmedian(relerr(kp, S["kappa"])), f, time.time() - t0))
        del ns, net
        if a.device == "cuda": torch.cuda.empty_cache()


def main():
    """Parse the command line and predict the edited shells."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="comma-separated sets of edited shells (<edited-dir>/<set>.npz)")
    ap.add_argument("--families", default=",".join(FAMILIES)); ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--variant", default="b3", choices=["n8192", "b3"]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8, help="shells per forward pass (8 = the training evaluation)")
    ap.add_argument("--edited-dir", default=EDITED); ap.add_argument("--out-dir", default=H)
    ap.add_argument("--save-fields", action="store_true", help="also store the predicted fields")
    ap.add_argument("--threads", type=int, default=int(THREADS), help="torch threads on cpu")
    predict(ap.parse_args())


if __name__ == "__main__":
    main()
