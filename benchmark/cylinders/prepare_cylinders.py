"""Second structural system (axially compressed cylinders): per-shell records -> the training arrays and the
three protocols B1-cyl, B2-cyl, B3-cyl, in the format benchmark/field_models/train.py reads with --data and
--split-file.

    python benchmark/cylinders/prepare_cylinders.py [--stage all] [--records DIR] [--design FILE [FILE ...]]
                                                    [--base-design FILE] [--out FILE] [--splits-dir DIR]
                                                    [--rerun-suffix SUFFIX]
      --stage arrays   records -> <BLISS_HEAVY>/cyl_nodes_8192.npz and cyl_nodes_8192_report.json
      --stage splits   cyl_nodes_8192.npz -> splits/cylinders/B1.json, B2.json, B3.json
      --stage all      both (default)

INPUTS (the input contract; nothing here reads raw solver output):
  design file(s)   JSON {"shells": [{"id": <name>, "set": <set>, "delta_max": float, "delta_mean": float,
                   "n_def": int}, ...]}. <name> ends in "_<i>", the index of the shell within its set; <set> is one
                   of train_sparse, val_sparse, test_sparse, test_dense, train_dense, val_dense (shells of any
                   other set are ignored). delta_max is the depth of the deepest dimple of the design, in shell
                   thicknesses; delta_mean the mean depth the design was drawn around; n_def the number of dimples.
                   Several files are merged (a later file overrides an earlier one on the same name).
  records          one folder per shell, <records>/<name>/, holding
                     DONE          empty marker: the record is complete (folders without it are left out)
                     meta.json     {"nth", "nz", "R", "H", "t"}: nodes around the circumference, elements along
                                   the axis, radius, height and thickness (mm). The mesh is structured, nth x
                                   (nz + 1) nodes, node j * nth + i at theta = 2 pi i / nth, z = H j / nz, and is
                                   the same for every shell.
                     result.json   {"peak_passed": bool, "kappa_first": float, "kappa_pre_negeig": float | null}:
                                   whether the load path passed a limit point, the load at the first limit point
                                   over the classical load, the last load before the first negative eigenvalue.
                     fields.npz    w0 (nth * (nz + 1),) radial imperfection at the nodes (mm, outward positive),
                                   U_peak (nth * (nz + 1), 3) Cartesian displacement at the first limit point (mm).
                   A shell whose load path stopped before a limit point may have a longer rerun stored in
                   <records>/<name><rerun-suffix>/ (same four files); it is used when it passed the limit point.
                   --rerun-suffix gives the suffix of those folders. Its default is the empty string: no rerun
                   folders, every shell is read from <records>/<name>/ only.
                   Only complete records with peak_passed enter; the others are listed in the report.

DEFINITIONS
  node set     8,192 of the nth * (nz + 1) nodes, drawn once with numpy default_rng(0) and sorted; the same nodes
               for every shell (NODES = 0-based index j * nth + i).
  ids          train_sparse i -> 1000000 + i, val_sparse 1100000 + i, test_sparse 1200000 + i, test_dense
               1300000 + i, train_dense 1800000 + i, val_dense 1900000 + i. Rows are sorted by shell name.
  regime       dense = set in {test_dense, train_dense, val_dense}; sparse otherwise.
  B1-cyl       within distribution: random 75 / 10 / 15 over the whole pool, numpy default_rng(20261124)
               permutation of the rows.
  B2-cyl       deeper dimples: test = shells whose deepest dimple (delta_max, design depth) is above the 0.72
               quantile of the pool (28 % of the pool); validation = 12 % of the rest, drawn with
               default_rng(20261125); train = the remainder.
  B3-cyl       sparse to dense: train = train_sparse and val = val_sparse shells of the base design; test =
               test_sparse + test_dense, in this order. The base design is the design file the protocol was drawn on
               (--base-design, default the first --design file); shells added to the pool by the later design
               files enter B1-cyl and B2-cyl only.
  Every split file lists ids only: train, val, test, test_sparse, test_dense, test_regime {id: "sparse" |
  "dense"}, and the usual header (split, description, n_shells; B2-cyl also threshold_t).

OUTPUTS
  <BLISS_HEAVY>/cyl_nodes_8192.npz (default BLISS_HEAVY = <BLISS_ROOT>/outputs), the keys of the hemisphere file
  (ids, P, W, U, K, R, NODES) plus the cylinder ones:
    ids      (n,) int64
    P        (8192, 3) perfect-geometry node coordinates (R cos th, R sin th, z - H/2) / (H/2): centred on the
             axis at mid-height and divided by the largest half-extent, so the shell fills [-1, 1] along its axis
             and stays inside the box (-1, 1)^3 that the point-cloud models assume
    W        (n, 8192) radial imperfection, mm, outward positive
    U        (n, 8192, 3) Cartesian displacement at the first limit point, mm
    K        (n,) kappa, first limit point over the classical load
    R        (n,) NaN (depth ratio of the two deepest defects: defined on the hemisphere only; kept so that the
             trainer writes its prediction files unchanged)
    NRM      (8192, 3) unit outward normals (cos th, sin th, 0); THZ (8192, 2) theta (rad), z (mm); RADIUS
    KPN      (n,) last load before the first negative eigenvalue (NaN if none)
    RERUN    (n,) True where the label comes from the rerun record (all False without --rerun-suffix)
    names, set (0 train_sparse, 1 val_sparse, 2 test_sparse, 3 test_dense, 8 train_dense, 9 val_dense),
    base (True for the shells of the base design), delta_max, delta_mean, n_def
  <BLISS_HEAVY>/cyl_nodes_8192_report.json: counts per set, skipped shells, ranges.
  <BLISS_ROOT>/splits/cylinders/B1.json, B2.json, B3.json.

Functions
---------
shell_index
    Index of a shell within its set, read from its name.
read_design
    Merge the design files into {name: row}, keeping the sets of the pool.
select_records
    Complete records that passed a limit point, with the rerun rule.
node_subset
    The fixed 8,192-node subset and its geometry (P, normals, theta, z).
build_arrays
    Stack the records on the node subset and write the training file and its report.
split_dict
    One split file as a dictionary, from boolean masks over the rows.
make_b1
    B1-cyl masks: random 75 / 10 / 15.
make_b2
    B2-cyl masks: the deepest-dimple quantile.
make_b3
    B3-cyl masks: sparse to dense on the base design.
write_splits
    Write the three split files from the training file.
main
    Command line.
"""
import argparse, json, math, os
import numpy as np

B = os.environ.get("BLISS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.environ.get("BLISS_HEAVY", B + "/outputs")
N_NODES = 8192
SETS = {"train_sparse": (0, 1000000), "val_sparse": (1, 1100000), "test_sparse": (2, 1200000),
        "test_dense": (3, 1300000), "train_dense": (8, 1800000), "val_dense": (9, 1900000)}   # set -> (code, id base)
DENSE_CODES = [3, 8, 9]
B1_SEED, B2_SEED = 20261124, 20261125
B1_FRACTIONS = (0.75, 0.85)          # train up to 75 %, validation up to 85 %, test the rest
B2_QUANTILE, B2_VAL = 0.72, 0.12


def shell_index(name):
    """Index of a shell within its set, read from its name.

    Parameters
    ----------
    name : str
        Shell name, ending in "_<i>".

    Returns
    -------
    i : int
        The index.
    """
    return int(name.rsplit("_", 1)[1])


def read_design(files):
    """Merge the design files into {name: row}, keeping the sets of the pool.

    Parameters
    ----------
    files : list of str
        Design files, JSON {"shells": [{"id", "set", "delta_max", "delta_mean", "n_def"}, ...]}.

    Returns
    -------
    design : dict
        {name: row} of the shells whose set is one of SETS; a later file overrides an earlier one.
    """
    design = {}
    for f in files:
        design.update({s["id"]: s for s in json.load(open(f))["shells"] if s["set"] in SETS})
    return design


def select_records(design, records, rerun_suffix=""):
    """Complete records that passed a limit point, with the rerun rule.

    Parameters
    ----------
    design : dict
        {name: row} from read_design.
    records : str
        Folder with one record folder per shell.
    rerun_suffix : str, default=""
        Suffix of the folder of the longer rerun of a shell that stopped before a limit point; empty for no rerun
        folders.

    Returns
    -------
    rows : list of (str, dict, str)
        (name, result, record folder) of the shells that enter, sorted by name.
    skipped : list of (str, str)
        (name, reason) of the complete or failed records that do not enter.
    """
    rows, skipped = [], []
    for name in sorted(design):
        rd = records + "/" + name
        if not os.path.exists(rd + "/DONE"):
            if os.path.exists(rd + "/FAILED"): skipped.append((name, "FAILED"))
            continue
        r = json.load(open(rd + "/result.json"))
        if rerun_suffix and not r.get("peak_passed") and os.path.exists(rd + rerun_suffix + "/DONE"):   # no limit point
            r2 = json.load(open(rd + rerun_suffix + "/result.json"))                   # before the stop: use the longer rerun
            if r2.get("peak_passed"):
                rd, r = rd + rerun_suffix, r2
        if not r.get("peak_passed"):
            skipped.append((name, "peak not passed")); continue
        rows.append((name, r, rd))
    return rows, skipped


def node_subset(nth, nz, radius, height, n_nodes=N_NODES, seed=0):
    """The fixed node subset and its geometry.

    Parameters
    ----------
    nth, nz : int
        Nodes around the circumference and elements along the axis (the mesh has nth * (nz + 1) nodes).
    radius, height : float
        Radius and height of the cylinder (mm).
    n_nodes : int
        Size of the subset.
    seed : int
        Seed of the draw.

    Returns
    -------
    sel : numpy.ndarray, shape (n_nodes,)
        Sorted 0-based node indices j * nth + i.
    P : numpy.ndarray, shape (n_nodes, 3)
        Node coordinates centred at mid-height and divided by height / 2, float32.
    NRM : numpy.ndarray, shape (n_nodes, 3)
        Unit outward normals, float32.
    THZ : numpy.ndarray, shape (n_nodes, 2)
        theta (rad) and z (mm).
    """
    sel = np.sort(np.random.default_rng(seed).choice(nth * (nz + 1), n_nodes, replace=False))
    jj, ii = sel // nth, sel % nth
    th = 2 * math.pi * ii / nth; z = height * jj / nz
    P = (np.stack([radius * np.cos(th), radius * np.sin(th), z - height / 2], 1) / (height / 2)).astype(np.float32)
    NRM = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1).astype(np.float32)
    return sel, P, NRM, np.stack([th, z], 1)


def build_arrays(design, base_names, records, out, rerun_suffix=""):
    """Stack the records on the node subset and write the training file and its report.

    Parameters
    ----------
    design : dict
        {name: row} from read_design (the pool).
    base_names : set of str
        Names of the shells of the base design (the B3-cyl protocol uses these only).
    records : str
        Folder with one record folder per shell.
    out : str
        Path of the training file (.npz); the report is written next to it as <out without .npz>_report.json.
    rerun_suffix : str, default=""
        Suffix of the rerun folders (see select_records); empty for none.

    Returns
    -------
    report : dict
        Counts per set, skipped shells and ranges.
    """
    rows, skipped = select_records(design, records, rerun_suffix)
    m0 = json.load(open(rows[0][2] + "/meta.json"))
    nth, nz, R, H = m0["nth"], m0["nz"], m0["R"], m0["H"]
    sel, P, NRM, THZ = node_subset(nth, nz, R, H)
    n = len(rows)
    ids = np.zeros(n, np.int64); W = np.zeros((n, N_NODES), np.float32); U = np.zeros((n, N_NODES, 3), np.float32)
    K, KPN = np.zeros(n), np.full(n, np.nan)
    st, dx, dm, nd = np.zeros(n, np.int8), np.zeros(n), np.zeros(n), np.zeros(n, np.int32)
    RERUN = np.array([bool(rerun_suffix) and x[2].endswith(rerun_suffix) for x in rows])
    for k, (name, r, rd) in enumerate(rows):
        code, base = SETS[design[name]["set"]]
        m = json.load(open(rd + "/meta.json")); assert (m["nth"], m["nz"]) == (nth, nz)
        f = np.load(rd + "/fields.npz")
        ids[k] = base + shell_index(name); W[k] = f["w0"].reshape(-1)[sel]; U[k] = f["U_peak"][sel]
        K[k] = r["kappa_first"]; KPN[k] = r["kappa_pre_negeig"] if r.get("kappa_pre_negeig") is not None else np.nan
        st[k] = code; dx[k] = design[name]["delta_max"]; dm[k] = design[name]["delta_mean"]; nd[k] = design[name]["n_def"]
        if k % 500 == 0: print("  %d/%d" % (k, n), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez(out, ids=ids, P=P, W=W, U=U, K=K, R=np.full(n, np.nan), NODES=sel.astype(np.int64), NRM=NRM, THZ=THZ,
             RADIUS=np.float64(R), KPN=KPN, names=np.array([x[0] for x in rows]), RERUN=RERUN, set=st,
             base=np.array([x[0] in base_names for x in rows]), delta_max=dx, delta_mean=dm, n_def=nd)
    report = dict(n_rerun=int(RERUN.sum()), n=n, per_set={s: int((st == c).sum()) for s, (c, _) in SETS.items()},
                  skipped=skipped, K_range=[float(K.min()), float(K.max())],
                  W_absmax_over_t=float(np.abs(W).max()) / m0["t"], P_min=P.min(0).tolist(), P_max=P.max(0).tolist(),
                  mesh=[nth, nz])
    json.dump(report, open(out[:-4] + "_report.json", "w"), indent=1)
    return report


def split_dict(name, description, ids, dense, tr, va, te, **extra):
    """One split file as a dictionary, from boolean masks over the rows.

    Parameters
    ----------
    name : str
        Name of the protocol.
    description : str
        One sentence on what changes between train and test.
    ids : numpy.ndarray, shape (n,)
        Shell ids of the rows.
    dense : numpy.ndarray of bool, shape (n,)
        True for the shells of the dense regime.
    tr, va, te : numpy.ndarray of bool, shape (n,)
        Train, validation and test rows.
    **extra
        Further header fields.

    Returns
    -------
    sp : dict
        train, val, test, test_sparse, test_dense (lists of ids), test_regime {id: regime} and the header.
    """
    return dict(train=ids[tr].tolist(), val=ids[va].tolist(), test=ids[te].tolist(),
                test_sparse=ids[te & ~dense].tolist(), test_dense=ids[te & dense].tolist(),
                test_regime={str(i): ("dense" if d else "sparse") for i, d in zip(ids[te], dense[te])},
                split_version="BLISS-cyl-splits-1.0", dataset_version="BLISS-cyl-1.0", split=name,
                description=description, n_shells=int(len(ids)), **extra)


def make_b1(n, seed=B1_SEED, fractions=B1_FRACTIONS):
    """B1-cyl masks: a random permutation of the rows cut at the two fractions.

    Parameters
    ----------
    n : int
        Number of shells in the pool.
    seed : int
        Seed of the permutation.
    fractions : (float, float)
        Cumulative fractions where train and validation end.

    Returns
    -------
    tr, va, te : numpy.ndarray of bool, shape (n,)
        Train, validation and test rows.
    """
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(round(fractions[0] * n)), int(round(fractions[1] * n))
    mask = lambda idx: np.isin(np.arange(n), idx)
    return mask(perm[:a]), mask(perm[a:b]), mask(perm[b:])


def make_b2(delta_max, seed=B2_SEED, quantile=B2_QUANTILE, val=B2_VAL):
    """B2-cyl masks: test above a quantile of the deepest-dimple depth, validation drawn from the rest.

    Parameters
    ----------
    delta_max : numpy.ndarray, shape (n,)
        Depth of the deepest dimple of every shell (design depth, in thicknesses).
    seed : int
        Seed of the validation draw.
    quantile : float
        Quantile of delta_max above which a shell is a test shell.
    val : float
        Share of the remaining shells drawn for validation.

    Returns
    -------
    tr, va, te : numpy.ndarray of bool, shape (n,)
        Train, validation and test rows.
    threshold : float
        The depth threshold (thicknesses).
    """
    n = len(delta_max); thr = float(np.quantile(delta_max, quantile)); te = delta_max > thr; rest = np.where(~te)[0]
    pv = np.random.default_rng(seed).permutation(rest); nv = int(round(val * len(rest)))
    mask = lambda idx: np.isin(np.arange(n), idx)
    return mask(pv[nv:]), mask(pv[:nv]), te, thr


def make_b3(st, base):
    """B3-cyl masks: train and validate on the sparse shells of the base design, test on its sparse and dense
    test shells.

    Parameters
    ----------
    st : numpy.ndarray, shape (n,)
        Set code of every shell (SETS).
    base : numpy.ndarray of bool, shape (n,)
        True for the shells of the base design.

    Returns
    -------
    tr, va, te : numpy.ndarray of bool, shape (n,)
        Train, validation and test rows.
    """
    return base & (st == 0), base & (st == 1), base & np.isin(st, [2, 3])


def write_splits(data, splits_dir):
    """Write the three split files from the training file.

    Parameters
    ----------
    data : str
        Path of the training file written by build_arrays.
    splits_dir : str
        Folder that receives B1.json, B2.json and B3.json.

    Returns
    -------
    splits : dict
        {"B1" | "B2" | "B3": split dictionary}.
    """
    z = np.load(data)
    ids = z["ids"].astype(int); st = z["set"]; base = z["base"]; dmax = z["delta_max"]
    dense = np.isin(st, DENSE_CODES); n = len(ids)
    out = {}
    tr, va, te = make_b1(n)
    out["B1"] = split_dict("B1-cyl", "within distribution on the axially compressed cylinder: random 75/10/15 over "
                           "the whole pool (sparse and dense layouts)", ids, dense, tr, va, te, split_seed=B1_SEED)
    tr, va, te, thr = make_b2(dmax)
    out["B2"] = split_dict("B2-cyl", "deeper dimples on the axially compressed cylinder: test shells have their "
                           "deepest dimple above the threshold (28 % of the pool)", ids, dense, tr, va, te,
                           split_seed=B2_SEED, threshold_t=thr)
    tr, va, te = make_b3(st, base)
    out["B3"] = split_dict("B3-cyl", "regime transfer on the axially compressed cylinder: train and validate on "
                           "sparse layouts (min spacing 1.26 l_c), test on sparse (1.26 l_c) and dense (0.51 l_c) "
                           "layouts", ids, dense, tr, va, te)
    b3 = out["B3"]; b3["n_shells"] = int((tr | va | te).sum())
    b3["test"] = b3["test_sparse"] + b3["test_dense"]            # sparse test shells first, then the dense ones
    b3["test_regime"] = {str(i): b3["test_regime"][str(i)] for i in b3["test"]}
    os.makedirs(splits_dir, exist_ok=True)
    for k, sp in out.items():
        json.dump(sp, open("%s/%s.json" % (splits_dir, k), "w"))
        m = lambda key: np.isin(ids, sp[key])
        print("%s  train %d (dense %d)  val %d (dense %d)  test %d (dense %d)" % (
            sp["split"], m("train").sum(), (m("train") & dense).sum(), m("val").sum(), (m("val") & dense).sum(),
            m("test").sum(), (m("test") & dense).sum()) + ("  threshold %.4f t" % sp["threshold_t"] if "threshold_t" in sp else ""))
    return out


def main():
    """Command line: build the training file, the split files, or both."""
    ap = argparse.ArgumentParser(description="Cylinder records -> training arrays and the B1-cyl, B2-cyl, B3-cyl splits.")
    ap.add_argument("--stage", default="all", choices=["arrays", "splits", "all"])
    ap.add_argument("--records", default=B + "/cylinders/records", help="folder with one record folder per shell")
    ap.add_argument("--design", nargs="+", default=[B + "/cylinders/design.json"],
                    help="design file(s) of the pool; later files override earlier ones on the same shell name")
    ap.add_argument("--base-design", default="", help="design file the B3-cyl protocol was drawn on "
                    "(default: the first --design file)")
    ap.add_argument("--rerun-suffix", default="", help="suffix of the record folder of the longer rerun of a "
                    "shell that stopped before a limit point (default: empty, no rerun folders)")
    ap.add_argument("--out", default=OUT + "/cyl_nodes_%d.npz" % N_NODES, help="training file (.npz)")
    ap.add_argument("--splits-dir", default=B + "/splits/cylinders", help="folder for B1.json, B2.json, B3.json")
    A = ap.parse_args()
    if A.stage in ("arrays", "all"):
        design = read_design(A.design)
        base_names = set(read_design([A.base_design or A.design[0]]))
        print(json.dumps(build_arrays(design, base_names, A.records, A.out, A.rerun_suffix), indent=1))
    if A.stage in ("splits", "all"):
        write_splits(A.out, A.splits_dir)


if __name__ == "__main__":
    main()
