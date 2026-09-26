"""Geometry helpers of the shell mesh, shared by the benchmark scripts.

The mesh spans a hemisphere whose pole lies on one Cartesian axis. These functions find that axis, build the
frame in which the defect angles are defined, resample per-node arrays onto the 180 x 720 grid, and find the
first limit point of a load path.

Functions
---------
first_limit
    Index of the first limit point of a load path.
frame_of
    Orthonormal frame (e1, e2, axis) of the hemisphere from its unit node directions.
to_grid
    Resample a per-node array onto the 180 x 720 grid.
pole_axis
    Name and direction of the pole axis of the mesh.
frame
    Frame (e1, e2, pole) in which the defect angles (theta, phi) are defined.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = "The BLISS authors (withheld for double-blind review)"
__credits__ = []
__status__ = "Stable"

R, NB, NT, DROP = 25.4, 180, 720, 1e-3


# =============================================================================
def first_limit(l):
    """Index of the first limit point of a load path.

    The first increment that is a local maximum of the load and is followed by a drop of at least DROP
    (relative) is the limit point; if there is none, the global maximum is returned.

    Parameters
    ----------
    l : numpy.ndarray
        Load proportionality factor at every increment.

    Returns
    -------
    i : int
        Index of the first limit point.
    """
    for i in range(1, len(l) - 1):
        if l[i] >= l[i - 1] and l[i] > l[i + 1] and (l[i] - l[i + 1:].min()) / l[i] >= DROP:
            return i
    return int(np.argmax(l))


def frame_of(p):
    """Orthonormal frame of the hemisphere from its unit node directions.

    Parameters
    ----------
    p : numpy.ndarray, shape (N, 3)
        Unit node directions.

    Returns
    -------
    e1, e2, ax : numpy.ndarray
        Two in-plane unit vectors and the pole axis.
    """
    for v in (np.array([0, 0, 1.]), np.array([0, 1., 0]), np.array([1., 0, 0])):
        if np.degrees(np.arccos(np.clip(p @ v, -1, 1))).max() < 91:
            ax = v
    e1 = np.array([1., 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1., 0])
    e1 = e1 - (e1 @ ax) * ax; e1 /= np.linalg.norm(e1)
    return e1, np.cross(ax, e1), ax


def to_grid(a, idx, w):
    """Resample a per-node array onto the 180 x 720 grid (inverse distance on the four nearest nodes).

    Parameters
    ----------
    a : numpy.ndarray, shape (N,) or (N, 3)
        Per-node array.
    idx : numpy.ndarray, shape (NB * NT, 4)
        Indices of the four nearest nodes of every grid cell (knn_idx of shared.npz).
    w : numpy.ndarray, shape (NB * NT, 4)
        Their weights (knn_w of shared.npz).

    Returns
    -------
    g : numpy.ndarray, shape (NB, NT) or (NB, NT, 3)
        The array on the grid, float32.
    """
    out = (a[idx] * (w[..., None] if a.ndim == 2 else w)).sum(1)
    return out.reshape(NB, NT, -1).squeeze().astype(np.float32)


def pole_axis(X):
    """The mesh spans a hemisphere; the pole is the axis for which beta never exceeds 90 deg.

    Parameters
    ----------
    X : numpy.ndarray, shape (N, 3)
        Node positions.

    Returns
    -------
    name : str or None
        'Z', 'Y' or 'X'.
    axis : numpy.ndarray or None
        Unit vector of the pole.
    """
    r = np.linalg.norm(X, axis=1)
    for name, v in (("Z", (0, 0, 1.0)), ("Y", (0, 1.0, 0)), ("X", (1.0, 0, 0))):
        b = np.degrees(np.arccos(np.clip(X @ np.array(v) / r, -1, 1)))
        if b.max() < 90.5:
            return name, np.array(v)
    return None, None


def frame(axis_name):
    """Orthonormal frame (e1, e2, pole) so that (theta, phi) map the same way as the generator.

    Parameters
    ----------
    axis_name : str, {'Z', 'Y', 'X'}
        Pole axis of the mesh.

    Returns
    -------
    e1, e2, pole : numpy.ndarray
        The frame.
    """
    if axis_name == "Z":
        return np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    if axis_name == "Y":
        return np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
    return np.array([0, 1.0, 0]), np.array([0, 0, 1.0]), np.array([1.0, 0, 0])
