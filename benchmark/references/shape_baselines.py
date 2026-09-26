"""Reference rows that read only w (or nothing), on the released protocols B1/B2/B3, v2 canonical records.

    python benchmark/references/shape_baselines.py [--nproc 8] [--smoke 60]

Same footing as the learned models: the 8,192-node subset and arrays they were trained on (heavy/canon_nodes_8192.npz
= BLISS-1.0/canonical w, U_peak at NODES), the released split files, and the SAME model-selection protocol:
configurations chosen once on the B1 VALIDATION set with the selection metric of confirm.py/search.py
(val buckling_dimple_r2, the released definition; ties broken by val median kappa error, as search_space.py states),
then held FIXED for B2 and B3. Every method below is deterministic given its configuration, so the three "seeds"
of the protocol are identical runs: they are not repeated, and the json says so.

  D-N1 learners (r3: 25 Optuna trials each on B1 val, see SPACE in main(); the grids below were r2's 6-config version):
    knn         k nearest TRAINING shells in w (L2 on the standardised w over the 8,192 nodes); prediction = mean
                of their fields and kappas. k in {1, 2, 5, 10, 20, 50}.
    pca_ridge   PCA of w fitted on train, ridge from the standardised leading-m scores to every node/component of U
                and to kappa. (m, alpha) in {16, 64, 256} x {1, 100} (alpha on standardised scores).
    gb_w        HistGradientBoosting on w features (the 32 leading PC scores + min, mean, std of w, the number of
                nodes deeper than 0.5 t), predicting kappa; field = training-mean field. (learning_rate,
                max_leaf_nodes) in {0.03, 0.1, 0.3} x {15, 63}, 500 iterations. Its field metrics equal the
                mean-field row for every configuration, so the selection is decided by the tie-break (val kappa).
    regime_mean per-regime training mean field and kappa (sparse: ids < 4758, beta_min 25 deg; dense: ids >= 4758);
                no hyperparameters. For B3 (train sparse only) the dense test shells get the sparse mean: stated.
  D-N5 near-duplicate audit: distance of every test shell to its nearest training shell in w, over the median
                nearest-neighbour distance among training shells; histogram per protocol.
  D-M4 weakest link from w, clamped: kappa_WL = kappa_1(clip(max(-w)/t, curve range)), t = 0.23 mm (the deck wall, the unit of delta),
                on the full 76,805-node w of the canonical record and on the 8,192 subset; the oracle version
                (delta_max from metadata/defects.csv, clamped) beside it. All test shells, counts from split files.
  D-M5 deeper-of-two near-tie reference: on the ambiguous test shells (released q > 0.9) with two competing sites
                (evaluate._two_peaks on the true D, 8,192 nodes, as the scorer), predict that the site with the
                deeper w (min of w within 5 deg of the site) wins; its two-way accuracy, beside chance (0.5) and the
                mean-field two-way floor.
  D-N8 localisation resolution: angular nearest-neighbour spacing of the 8,192 scored nodes and of the full mesh.

Writes heavy/wref_v2/<model>_<B>_test_pred.npz (train format, scored by rescore.py like every model),
heavy/references_w_v2.json.

Functions
---------
log
    Print a message with the time.
sel_score
    Selection metrics of a prediction on validation.
pick
    Selection: max val buckling_dimple_r2, ties (to 1e-9) broken by min val kappa error.
save
    Write the test predictions of a reference in the train.py format.
main
    Fit, select and score every reference that reads only w, on B1, B2 and B3.
"""
import argparse, csv, json, os, sys, time
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, os.environ.get("SLURM_CPUS_PER_TASK", "8"))
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); H = B + "/outputs"; REL = B + "/BLISS-1.0"; OUT = H + "/wref_v2"
sys.path.insert(0, B)
from bliss.metrics import score, released_q, _two_peaks, _ring   # noqa: E402
PROT = ("B1", "B2", "B3"); T_WALL = 0.23   # the deck wall: delta is in units of 0.23 mm (FACTS_dataset 4h; rotation_inputs check)
KD1 = os.environ.get("BLISS_KD1", os.path.join(B, "metadata", "single_defect_kd.csv"))   # kappa of the single-defect shells by depth and width


def log(*a): 
    """Print a message with the time.

    Parameters
    ----------
    *a : object
        Items to print.
    """
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sel_score(Ut, Up, Kt, Kp, P, ids):
    """Selection metrics of a prediction on validation.

    Parameters
    ----------
    Ut, Up : numpy.ndarray
        True and predicted fields.
    Kt, Kp : numpy.ndarray
        True and predicted kappa.
    P : numpy.ndarray
        Unit node directions.
    ids : numpy.ndarray
        Shell identifiers.

    Returns
    -------
    b : float
        buckling_dimple_r2.
    k : float
        Median relative error of kappa (%).
    """
    sc = score(Ut, Up, Kt, Kp, P, None, released_q(ids))
    return sc["buckling_dimple_r2"], sc["kappa_rel_err"]


def pick(rows):
    """Selection: max val buckling_dimple_r2, ties (to 1e-9) broken by min val kappa error.

    Parameters
    ----------
    rows : list of dict
        Configurations with their validation scores.

    Returns
    -------
    row : dict
        The selected configuration.
    """
    best = max(r["val_buckling_dimple_r2"] for r in rows)
    tied = [r for r in rows if r["val_buckling_dimple_r2"] >= best - 1e-9]
    return min(tied, key=lambda r: r["val_kappa_rel_err"])


def save(name, prot, ids, Up, Kp, D, idx, extra=None):
    """Write the test predictions of a reference in the train.py format.

    Parameters
    ----------
    name : str
        Name of the reference.
    prot : str
        Protocol.
    ids : numpy.ndarray
        Test shell ids.
    Up : numpy.ndarray
        Predicted fields.
    Kp : numpy.ndarray
        Predicted kappa.
    D : dict
        Loaded data.
    idx : numpy.ndarray
        Indices of the test shells.
    extra : dict, default=None
        Extra arrays to store.
    """
    os.makedirs(OUT, exist_ok=True)
    np.savez("%s/%s_%s_test_pred.npz" % (OUT, name, prot), ids=ids, pred_U=Up.astype(np.float32), pred_K=np.asarray(Kp, np.float64),
             true_U=D["U"][idx], true_K=D["K"][idx], P=D["P"], r=D["R"][idx], epoch=-1, **(extra or {}))


def main():
    """Fit, select and score every reference that reads only w, on B1, B2 and B3."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nproc", type=int, default=8); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--trials", type=int, default=25)
    a = ap.parse_args(); t0 = time.time()
    z = np.load(H + "/canon_nodes_8192.npz"); D = {k: z[k] for k in z.files}
    P = D["P"].astype(np.float64); P /= np.linalg.norm(P, axis=1, keepdims=True); D["P"] = P.astype(np.float32)
    pos = {int(i): j for j, i in enumerate(D["ids"])}
    SP = {p: json.load(open("%s/splits/%s.json" % (REL, p))) for p in PROT}
    I = {p: {s: np.array([pos[i] for i in SP[p][s]]) for s in ("train", "val", "test")} for p in PROT}
    if a.smoke:
        for p in PROT:
            for s in ("val", "test"): I[p][s] = I[p][s][:a.smoke]
            I[p]["train"] = I[p]["train"][:4 * a.smoke]
    res = dict(generated=time.strftime("%Y-%m-%dT%H:%M:%S"), smoke=a.smoke, selection=("B1 validation, val buckling_dimple_r2 "
               "(phase2/run_optuna objective), ties -> val median kappa error (search_space.py); fixed for B2/B3"),
               seeds="deterministic methods: the three protocol seeds are identical runs", counts={}, models={})
    for p in PROT: res["counts"][p] = {s: int(len(I[p][s])) for s in ("train", "val", "test")}
    W = D["W"].astype(np.float64); SVD = {}
    global OUT
    if a.smoke: OUT = os.path.abspath("wref_v2_smoke")

    # ---- the four learners -------------------------------------------------------------------------------
    def fit_predict(model, cfg, tr, targets):
        """Returns {name: (pred_U, pred_K)} for each index array in targets.

        Parameters
        ----------
        model : str
            Reference name.
        cfg : dict
            Configuration.
        tr : numpy.ndarray
            Training indices.
        targets : dict
            Name -> index array to predict.

        Returns
        -------
        pred : dict
            Name -> (pred_U, pred_K).
        """
        wm, ws = W[tr].mean(), W[tr].std(); Z = (W - wm) / ws
        Um, Km = D["U"][tr].mean(0), D["K"][tr].mean(); out = {}
        if model == "knn":
            Ztr = Z[tr]; ntr = (Ztr ** 2).sum(1)
            for nm, idx in targets.items():
                d2 = (Z[idx] ** 2).sum(1)[:, None] + ntr[None] - 2 * Z[idx] @ Ztr.T
                nn = np.argsort(d2, 1)[:, :cfg["k"]]
                if cfg.get("weights", "uniform") == "distance":           # 1/distance weights
                    wt = 1 / np.maximum(np.sqrt(np.maximum(np.take_along_axis(d2, nn, 1), 0)), 1e-9); wt /= wt.sum(1, keepdims=True)
                    out[nm] = ((D["U"][tr][nn] * wt[..., None, None]).sum(1), (D["K"][tr][nn] * wt).sum(1))
                else:
                    out[nm] = (D["U"][tr][nn].mean(1), D["K"][tr][nn].mean(1))
        elif model in ("pca_ridge", "gb_w"):
            m = cfg.get("m", 32); key = hash(tr.tobytes())
            if key not in SVD:                                 # one PCA per training set, reused by every config
                mu_ = Z[tr].mean(0); SVD[key] = (mu_, np.linalg.svd(Z[tr] - mu_, full_matrices=False)[2])
            mu, Vt = SVD[key]; m = min(m, Vt.shape[0]); V = Vt[:m].T      # m <= n_train (only binds in --smoke)
            sc = lambda idx: (Z[idx] - mu) @ V
            s_tr = sc(tr); s_m, s_s = s_tr.mean(0), s_tr.std(0) + 1e-12; std = lambda x: (x - s_m) / s_s
            if model == "pca_ridge":
                X = std(s_tr); Y = np.c_[D["U"][tr].reshape(len(tr), -1) - Um.reshape(-1), D["K"][tr] - Km]
                Wr = np.linalg.solve(X.T @ X + cfg["alpha"] * np.eye(m), X.T @ Y)
                for nm, idx in targets.items():
                    y = std(sc(idx)) @ Wr
                    out[nm] = (y[:, :-1].reshape(len(idx), -1, 3) + Um, y[:, -1] + Km)
            else:
                from sklearn.ensemble import HistGradientBoostingRegressor
                feat = lambda idx: np.c_[std(sc(idx)), W[idx].min(1), W[idx].mean(1), W[idx].std(1), (W[idx] < -0.5 * T_WALL).sum(1)]
                g = HistGradientBoostingRegressor(max_iter=500, learning_rate=cfg["lr"], max_leaf_nodes=cfg["leaves"],
                                                  l2_regularization=cfg.get("l2", 0.0), early_stopping=False,
                                                  random_state=0).fit(feat(tr), D["K"][tr])
                for nm, idx in targets.items():
                    out[nm] = (np.broadcast_to(Um, (len(idx),) + Um.shape), g.predict(feat(idx)))
        elif model == "regime_mean":
            dense_tr = D["ids"][tr] >= 4758; mean = {}
            for flag in (False, True):
                same = tr[dense_tr == flag]
                mean[flag] = (D["U"][same].mean(0), D["K"][same].mean()) if len(same) else (Um, Km)
            for nm, idx in targets.items():
                dn = D["ids"][idx] >= 4758
                out[nm] = (np.stack([mean[bool(x)][0] for x in dn]), np.array([mean[bool(x)][1] for x in dn]))
                out[nm + "_fallback"] = int(sum(1 for x in dn if not (dense_tr == bool(x)).any()))
        return out

    # r3 (ML2-05/06): the learned models' protocol. 25 Optuna trials per method (TPE, 7 random start-up trials, as
    # search_space.N_STARTUP; sampler seed 0), objective = B1 val buckling_dimple_r2 with the search_space tie-break folded
    # in (minus 1e-6 x val median kappa error in %, which only orders exact ties such as gb_w, whose field is the
    # training mean for every configuration), then FIXED for B2/B3. The fits are deterministic given the
    # configuration, so the three protocol seeds would be identical runs and are not repeated.
    SPACE = {"knn": lambda t: dict(k=t.suggest_int("k", 1, 200, log=True), weights=t.suggest_categorical("weights", ["uniform", "distance"])),
             "pca_ridge": lambda t: dict(m=t.suggest_int("m", 4, 1024, log=True), alpha=t.suggest_float("alpha", 1e-3, 1e4, log=True)),
             "gb_w": lambda t: dict(m=t.suggest_int("m", 8, 128, log=True), lr=t.suggest_float("lr", 0.01, 0.5, log=True),
                                    leaves=t.suggest_int("leaves", 7, 255, log=True), l2=t.suggest_float("l2", 1e-4, 10.0, log=True)),
             "regime_mean": None}
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    for model, space in SPACE.items():
        rows = []; tr, va = I["B1"]["train"], I["B1"]["val"]

        def trial_fn(cfg):
            """Score one configuration on validation and record it.

            Parameters
            ----------
            cfg : dict
                Configuration.

            Returns
            -------
            value : float
                Validation buckling_dimple_r2.
            """
            Up, Kp = fit_predict(model, cfg, tr, {"val": va})["val"]
            b, k = sel_score(D["U"][va], Up, D["K"][va], Kp, D["P"], D["ids"][va])
            rows.append(dict(config=cfg, val_buckling_dimple_r2=float(b), val_kappa_rel_err=float(k)))
            log("%-11s %-60s val buckling_dimple_r2 %.4f  kappa err %.3f %%" % (model, json.dumps(cfg), b, k))
            return float(b) - 1e-6 * float(k)
        if space is None:
            trial_fn({})
        else:
            st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=7, seed=0))
            st.optimize(lambda t: trial_fn(space(t)), n_trials=a.trials)
        win = pick(rows); cfg = win["config"]; entry = dict(trials=len(rows), search=rows, chosen=cfg, test={})
        for p in PROT:                                       # frozen config, each protocol's own train set
            tr, te = I[p]["train"], I[p]["test"]
            o = fit_predict(model, cfg, tr, {"test": te}); Up, Kp = o["test"]
            save(model, p, D["ids"][te], np.asarray(Up), Kp, D, te, dict(config=json.dumps(cfg)))
            sc = score(D["U"][te], np.asarray(Up, np.float32), D["K"][te], Kp, D["P"], D["R"][te], released_q(D["ids"][te]))
            entry["test"][p] = {k: v for k, v in sc.items() if not isinstance(v, dict)}
            if "test_fallback" in o: entry["test"][p]["regime_fallback_shells"] = int(o["test_fallback"])
            log("%-11s %s test: buckling_r2 %.3f  bdr2 %.3f  kappa err %.2f %%  loc<10 dom %.2f  two-way %.2f"
                % (model, p, sc["buckling_r2"], sc["buckling_dimple_r2"], sc["kappa_rel_err"], sc.get("loc_dominant_within_10", np.nan), sc.get("two_way_winner", np.nan)))
        res["models"][model] = entry
        if model == "knn":                                   # the k = 1 memorisation row, reported beside the tuned kNN
            e1 = dict(trials=0, chosen=dict(k=1, weights="uniform"), note="fixed k = 1, not tuned", test={})
            for p in PROT:
                tr, te = I[p]["train"], I[p]["test"]; Up, Kp = fit_predict("knn", e1["chosen"], tr, {"test": te})["test"]
                save("knn1", p, D["ids"][te], np.asarray(Up), Kp, D, te, dict(config=json.dumps(e1["chosen"])))
                sc = score(D["U"][te], np.asarray(Up, np.float32), D["K"][te], Kp, D["P"], D["R"][te], released_q(D["ids"][te]))
                e1["test"][p] = {k: v for k, v in sc.items() if not isinstance(v, dict)}
                log("knn1        %s test: buckling_r2 %.3f  kappa err %.2f %%" % (p, sc["buckling_r2"], sc["kappa_rel_err"]))
            res["models"]["knn1"] = e1

    # ---- D-N5 near-duplicate audit ------------------------------------------------------------------------
    nd = {}
    for p in PROT:
        tr, te = I[p]["train"], I[p]["test"]; Z = (W - W[tr].mean()) / W[tr].std(); Zt = Z[tr]; nt = (Zt ** 2).sum(1)
        d2 = lambda idx: np.maximum((Z[idx] ** 2).sum(1)[:, None] + nt[None] - 2 * Z[idx] @ Zt.T, 0)
        dte = np.sqrt(d2(te).min(1))
        dtt = d2(tr); np.fill_diagonal(dtt, np.inf); ref = float(np.median(np.sqrt(dtt.min(1))))
        rel = dte / ref; edges = [0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 5, 1e9]
        nd[p] = dict(median_train_nn_distance=ref, test_nn_over_ref_quantiles={q: float(np.quantile(rel, q)) for q in (0, 0.01, 0.05, 0.5, 0.95)},
                     histogram=dict(edges=edges, counts=np.histogram(rel, edges)[0].tolist()), n_below_0p1=int((rel < 0.1).sum()))
    res["near_duplicates"] = nd; log("near-duplicates:", {p: nd[p]["n_below_0p1"] for p in PROT})

    # ---- D-M4 weakest link from w -------------------------------------------------------------------------
    T = list(csv.reader(open(KD1, encoding="utf-8-sig")))
    col = [j for j, h in enumerate(T[0]) if "=1)" in h.replace(" ", "")][0]
    dl = np.array([float(r[0]) for r in T[1:]]); k1 = np.array([float(r[col]) for r in T[1:]]); o = np.argsort(dl); dl, k1 = dl[o], k1[o]
    kap1 = lambda d: np.interp(np.clip(d, dl.min(), dl.max()), dl, k1)
    Dd = {}
    for r in csv.DictReader(open(B + "/metadata/defects.csv")): Dd.setdefault(int(r["id"]), []).append(float(r["delta"]))
    allte = sorted(set(np.concatenate([I[p]["test"] for p in PROT]).tolist()))
    from multiprocessing import Pool
    with Pool(a.nproc) as pool:
        wmin_full = dict(zip(allte, pool.map(_wmin, [int(D["ids"][j]) for j in allte], chunksize=16)))
    wl = {}
    for p in PROT:
        te = I[p]["test"]; K = D["K"][te]; ids = D["ids"][te]
        dw_full = np.array([-wmin_full[j] for j in te]) / T_WALL; dw_sub = -W[te].min(1) / T_WALL
        dor = np.array([max(Dd[int(i)]) for i in ids])
        e = {}
        for nm, dd in (("from_w_full_mesh", dw_full), ("from_w_8192", dw_sub), ("oracle_delta_max", dor)):
            kp = kap1(dd); rel = np.abs(kp - K) / K * 100; clamp = (dd < dl.min()) | (dd > dl.max())
            e[nm] = {reg: dict(n=int(m.sum()), n_clamped=int((clamp & m).sum()), median_rel_err_pct=float(np.median(rel[m])),
                               median_signed_err_pct=float(np.median(((kp - K) / K * 100)[m])),
                               kappa_r2=float(1 - ((kp[m] - K[m]) ** 2).sum() / ((K[m] - K[m].mean()) ** 2).sum()) if m.sum() > 1 else None)
                     for reg, m in (("all", np.ones(len(te), bool)), ("sparse", ids < 4758), ("dense", ids >= 4758)) if m.any()}
            if nm == "from_w_full_mesh":
                save("weakest_link_w", p, ids, np.broadcast_to(D["U"][I[p]["train"]].mean(0), D["U"][te].shape), kp, D, te)
        e["depth_ratio_w_over_listed_median"] = float(np.median(dw_full / dor))
        wl[p] = e
        log("weakest link %s: from w %.2f %% (n %d, clamped %d) | oracle %.2f %%" % (p, e["from_w_full_mesh"]["all"]["median_rel_err_pct"],
            e["from_w_full_mesh"]["all"]["n"], e["from_w_full_mesh"]["all"]["n_clamped"], e["oracle_delta_max"]["all"]["median_rel_err_pct"]))
    res["weakest_link"] = dict(rule="kappa_1(clip(max(-w)/t)), t = 0.23 mm, lambda=1 column of %s" % KD1, splits=wl)

    # ---- D-M5 deeper-of-two ---------------------------------------------------------------------------------
    nb = _ring(P.astype(np.float64)); cosr = np.cos(np.radians(5.0)); dt = {}
    for p in PROT:
        te = I[p]["test"]; ids = D["ids"][te]; q = released_q(ids); amb = np.where(q > 0.9)[0]; hit, hit_node, n = [], [], 0
        for j in amb:
            U = D["U"][te[j]].astype(np.float64); Dv = U - (U * P).sum(-1).mean() * P
            t1, t2 = _two_peaks(Dv, P, nb)
            if t2 is None: continue
            w = W[te[j]]; n += 1
            loc = lambda t: w[(P @ P[t]) >= cosr].min()
            hit.append(loc(t1) < loc(t2)); hit_node.append(w[t1] < w[t2])
        Um = D["U"][I[p]["train"]].mean(0)
        mf = score(D["U"][te], np.broadcast_to(Um, D["U"][te].shape), D["K"][te], np.full(len(te), D["K"][I[p]["train"]].mean()), D["P"], None, q)
        dt[p] = dict(n_ambiguous=int(len(amb)), n_two_way=n, deeper_of_two_5deg=float(np.mean(hit)) if hit else None,
                     deeper_of_two_node=float(np.mean(hit_node)) if hit_node else None, chance=0.5,
                     meanfield_two_way=mf.get("two_way_winner"))
        log("deeper-of-two %s: %s" % (p, dt[p]))
    res["deeper_of_two"] = dt

    # ---- D-N8 localisation resolution ---------------------------------------------------------------------
    from scipy.spatial import cKDTree
    Pf = np.load(REL + "/shared.npz")["P"].astype(np.float64); Pf /= np.linalg.norm(Pf, axis=1, keepdims=True)
    res["node_spacing_deg"] = {}
    for nm, X in (("nodes_8192", P), ("full_mesh", Pf)):
        dd, _ = cKDTree(X).query(X, k=2); ang = np.degrees(2 * np.arcsin(np.clip(dd[:, 1] / 2, 0, 1)))
        res["node_spacing_deg"][nm] = dict(median=float(np.median(ang)), p95=float(np.quantile(ang, 0.95)), max=float(ang.max()))
    log("node spacing:", res["node_spacing_deg"])
    f = "references_w_v2_smoke.json" if a.smoke else H + "/references_w_v2.json"
    json.dump(res, open(f, "w"), indent=1, default=float); log("wrote %s [%.0fs]" % (f, time.time() - t0))


def _wmin(i):
    """Deepest point of w of one shell.

    Parameters
    ----------
    i : int
        Shell identifier.

    Returns
    -------
    wmin : float
        Minimum of w (mm).
    """
    return float(np.load("%s/canonical/%d.npz" % (REL, i))["w"].min())


if __name__ == "__main__":
    main()
