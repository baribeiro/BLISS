"""BLISS: loader of the dataset of buckling of imperfect spherical shells.

A release directory holds one shared file with the mesh, one record per shell, the parametric tables and the
split files. This module opens it and returns shells as objects whose arrays are read on demand.

    import bliss
    d = bliss.load("BLISS-1.0")          # or bliss.load() if the cwd is the release
    tr, va, te = d.split("B1")           # arrays of shell ids
    s = d[tr[0]]                         # one shell
    s.w        (N,)       imperfection field, mm, radial deviation from the sphere   <- input
    s.U        (N, 3)     displacement at the first limit point                      <- field target
    s.kappa    scalar     collapse load / classical (Zoelly) load                    <- scalar target
    s.X        (N, 3)     node coordinates, mm, computed as (R + w) * P
    s.q        scalar     ratio of the two competing collapse-site amplitudes

    from bliss import score
    m = score(U_true, U_pred, k_true, k_pred, d.P)

The mesh is identical across shells, so it is stored once in shared.npz; `d.P` are unit directions and a node
sits at (R + w) * P. Every shell also carries `U_prev` and `U_next`, the field one solver increment on either
side of the limit point, which is the resolution of the label.

Classes
-------
Shell
    One simulated shell, with its arrays read lazily from its record.
Dataset
    A release directory: shells, splits and parametric tables.

Functions
---------
load
    Open a release directory.
score
    The benchmark metrics of one split (from bliss.metrics).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import csv
import json
import os

# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = "The BLISS authors (withheld for double-blind review)"
__credits__ = []
__status__ = "Stable"

__all__ = ["load", "Dataset", "Shell", "score"]
R_DEFAULT = 25.4
T_SHELL = 0.23


# =============================================================================
#
# =============================================================================
class Shell:
    """One simulated shell. Arrays are read lazily from its record and cached.

    Attributes
    ----------
    id : int
        Shell identifier.
    w : numpy.ndarray, shape (N,)
        Radial deviation of the surface from the sphere (mm). The model input.
    U : numpy.ndarray, shape (N, 3)
        Displacement at the first limit point (mm). The field target.
    U_prev, U_next : numpy.ndarray, shape (N, 3)
        The same field one solver increment before and after the limit point (mm).
    mode : numpy.ndarray, shape (N, 3)
        Displacement from the limit point to three increments past it (mm).
    lpf : numpy.ndarray, shape (n_increments,)
        Load proportionality factor at every increment of the load path.
    kappa : float
        Knockdown factor, the collapse pressure over the classical pressure. The scalar target.
    q : float
        Ratio of the amplitudes of the two competing collapse sites (near-tie when q > 0.9).
    peak_frame : int
        Index of the limit point along `lpf`.
    r_d2_d1 : float
        Depth ratio of the two deepest defects.
    """

    __slots__ = ("id", "_z", "_ds", "_c")

    def __init__(self, sid, path, ds):
        """Open the record of one shell.

        Parameters
        ----------
        sid : int
            Shell identifier.
        path : str
            Path of the shell record (.npz).
        ds : Dataset
            The dataset the shell belongs to (for the shared mesh and the tables).
        """
        self.id, self._z, self._ds, self._c = int(sid), np.load(path), ds, {}

    def _get(self, k):
        """Read one array of the record, once.

        Parameters
        ----------
        k : str
            Key of the array in the record.

        Returns
        -------
        value : numpy.ndarray
            The array stored under `k`.
        """
        if k not in self._c: self._c[k] = self._z[k]
        return self._c[k]

    w = property(lambda s: s._get("w"))                      # (N,)   mm
    U = property(lambda s: s._get("U_peak"))                 # (N,3)  mm
    U_prev = property(lambda s: s._get("U_prev"))
    U_next = property(lambda s: s._get("U_next"))
    mode = property(lambda s: s._get("mode"))
    lpf = property(lambda s: s._get("lpf"))                  # load factor per increment
    kappa = property(lambda s: float(s._get("kappa")))
    q = property(lambda s: float(s._get("q")))
    peak_frame = property(lambda s: int(s._get("peak_frame")))
    r_d2_d1 = property(lambda s: float(s._get("r_d2_d1")))

    @property
    def X(self):
        """Node coordinates of the imperfect shell.

        Returns
        -------
        X : numpy.ndarray, shape (N, 3)
            Coordinates in mm, (R + w) * P. Built from the shared mesh, not stored.
        """
        return (self._ds.R + self.w)[:, None] * self._ds.P

    @property
    def X_deformed(self):
        """Node coordinates at the first limit point.

        Returns
        -------
        X_deformed : numpy.ndarray, shape (N, 3)
            Coordinates in mm, X + U. Contains the field target.
        """
        return self.X + self.U

    @property
    def defects(self):
        """The parameters that generated `w`.

        `w` is reproducible from them:
            w = sum_i -delta_i * t * exp(-(beta_i / beta0_i)**2)
            beta0_i = lambda_i * (12(1-nu**2))**-0.25 * (R/t)**-0.5

        Returns
        -------
        defects : numpy.ndarray, shape (n_defects, 4)
            (delta, lambda, theta, phi) per defect.
        """
        return self._ds.defects(self.id)

    def __repr__(self):
        """Short description of the shell."""
        return "<BLISS shell %d: kappa=%.4f q=%.3f %d defects>" % (self.id, self.kappa, self.q, len(self.defects))


# =============================================================================
class Dataset:
    """A release directory: shells, splits and parametric tables.

    Attributes
    ----------
    root : str
        Absolute path of the release directory.
    P : numpy.ndarray, shape (N, 3)
        Unit directions of the nodes of the shared mesh.
    conn : numpy.ndarray, shape (E, 4), or None
        Connectivity of the S4R elements (None when the directory carries no mesh, as the review sample).
    R : float
        Outer radius (mm).
    t : float
        Wall thickness (mm).
    ids : numpy.ndarray
        Identifiers of the shells present in the directory.
    partial : bool
        True when some listed shells are missing from the directory.
    """

    def __init__(self, root):
        """Open a release directory.

        Parameters
        ----------
        root : str
            Path of the release directory (holding shared.npz, canonical/, metadata/ and splits/).
        """
        self.root = os.path.abspath(root)
        sh = np.load(os.path.join(self.root, "shared.npz"))
        self.P = sh["P"].astype(np.float64)                  # (N,3) unit directions
        self.conn = sh["conn"] if "conn" in sh.files else None   # (E,4) S4R elements; absent in the review sample
        self.R = float(sh["R"]); self.t = T_SHELL
        here = {int(f[:-4]) for f in os.listdir(os.path.join(self.root, "canonical")) if f.endswith(".npz")}
        if os.path.exists(os.path.join(self.root, "metadata", "sample_ids.csv")):
            listed = {int(r["sample_id"]) for r in self._csv("sample_ids.csv")}
        else:                                                # review sample: the shells present are the list
            listed = here
        self.ids = np.array(sorted(listed & here))
        self.partial = len(self.ids) < len(listed)
        self._def = None
        self._meta = {}

    def _csv(self, name):
        """Read one parametric table.

        Parameters
        ----------
        name : str
            File name in metadata/.

        Returns
        -------
        rows : list[dict]
            One dictionary per row.
        """
        return list(csv.DictReader(open(os.path.join(self.root, "metadata", name))))

    def split(self, name):
        """Shell identifiers of one protocol.

        Parameters
        ----------
        name : str, {'B1', 'B2', 'B3'}
            'B1' within distribution, 'B2' deeper defects, 'B3' sparse to dense.

        Returns
        -------
        train, val, test : numpy.ndarray
            Shell identifiers of the three parts.
        """
        d = json.load(open(os.path.join(self.root, "splits", "%s.json" % name)))
        return np.array(d["train"]), np.array(d["val"]), np.array(d["test"])

    def split_info(self, name):
        """Metadata of one protocol (version, seed, description).

        Parameters
        ----------
        name : str, {'B1', 'B2', 'B3'}
            The protocol.

        Returns
        -------
        info : dict
            Every field of the split file except the three lists of identifiers.
        """
        d = json.load(open(os.path.join(self.root, "splits", "%s.json" % name)))
        return {k: v for k, v in d.items() if k not in ("train", "val", "test")}

    def defects(self, sid):
        """Defect parameters of one shell.

        Parameters
        ----------
        sid : int
            Shell identifier.

        Returns
        -------
        defects : numpy.ndarray, shape (n_defects, 4)
            (delta, lambda, theta, phi) per defect; empty if the shell is not in the table.
        """
        if self._def is None:
            self._def = {}
            for r in self._csv("defect_parameters.csv"):
                self._def.setdefault(int(r["id"]), []).append(
                    (float(r["delta"]), float(r["lambda"]), float(r["theta"]), float(r["phi"])))
        return np.array(self._def.get(int(sid), []))

    def meta(self, name):
        """Read a parametric table once and cache it.

        Parameters
        ----------
        name : str
            File name in metadata/.

        Returns
        -------
        rows : list[dict]
            One dictionary per row.
        """
        if name not in self._meta: self._meta[name] = self._csv(name)
        return self._meta[name]

    def __getitem__(self, sid):
        """The shell with identifier `sid`.

        Parameters
        ----------
        sid : int
            Shell identifier.

        Returns
        -------
        shell : Shell
            The shell, with arrays read on demand.
        """
        return Shell(sid, os.path.join(self.root, "canonical", "%d.npz" % int(sid)), self)

    def __len__(self):
        """Number of shells present."""
        return len(self.ids)

    def __iter__(self):
        """Iterate over the shells present, in identifier order."""
        for i in self.ids: yield self[i]

    def batch(self, ids, keys=("w", "U", "kappa")):
        """Stack a few shells into arrays.

        The full core at 76,805 nodes in float32 is about 4 GB, so shells are read on demand.

        Parameters
        ----------
        ids : sequence of int
            Shell identifiers.
        keys : tuple of str, default=('w', 'U', 'kappa')
            Attributes of Shell to stack.

        Returns
        -------
        arrays : dict
            One stacked array per key, with the shells along the first axis.
        """
        out = {k: [] for k in keys}
        for i in ids:
            s = self[i]
            for k in keys: out[k].append(getattr(s, k))
        return {k: (np.stack(v) if np.ndim(v[0]) else np.array(v)) for k, v in out.items()}

    def __repr__(self):
        """Short description of the dataset."""
        return "<BLISS %s: %d shells%s, %d nodes, %s elements>" % (
            os.path.basename(self.root), len(self.ids), " (partial)" if self.partial else "",
            len(self.P), len(self.conn) if self.conn is not None else "no")


# =============================================================================
def load(root=None):
    """Open a release directory.

    Parameters
    ----------
    root : str, default=None
        Path of the release. If None, the current directory is tried, then ./BLISS-1.0.

    Returns
    -------
    dataset : Dataset
        The opened release.

    Raises
    ------
    FileNotFoundError
        If no shared.npz is found.
    """
    for c in ([root] if root else [".", "BLISS-1.0"]):
        if os.path.exists(os.path.join(c, "shared.npz")): return Dataset(c)
    raise FileNotFoundError("no shared.npz found in %s" % (root or "'.' or './BLISS-1.0'"))


from .metrics import score  # noqa: E402,F401  (the released scorer)
