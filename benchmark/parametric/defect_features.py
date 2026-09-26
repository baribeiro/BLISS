"""Descriptors of the list of defects, shared by the MLP and the tabular models.

They live in one module so that both families see exactly the same input, and a comparison between them
measures the model and not the encoding.

CONTENT. Everything that defines the shell and its defects, nothing else:

  per shell    R, t, nu, eta = R/t, p_classical, N
  per defect   delta, lambda, beta0, and the unit vector (ux, uy, uz) of its position on the sphere

beta0 is the angular half-width of the defect, in degrees:

    beta0_i = lambda_i * (12(1-nu^2))^(-1/4) * (R/t)^(-1/2)

It folds lambda, nu and the slenderness into one number with a physical meaning. Young's modulus does not enter,
because kappa is normalised by p_classical, which scales with E.

The position is given as the unit vector (sin(phi)cos(theta), sin(phi)sin(theta), cos(phi)) rather than as two
angles, which are discontinuous at the pole and at the 2*pi cut.

PADDING. Slots are left over whenever N < K (N ranges from 11 to 81, K = 20). Three ways of filling them are
implemented and the choice is made on validation like any other hyperparameter:

  zero   zero padding.
  mask   zero padding plus one bit per slot saying whether it is real.
  nan    an absent slot is NaN. Trees handle NaN natively; the MLP imputes the training mean, which is the
         neutral value after normalisation.

Functions
---------
usar_catalogo
    Switch the source from the release (4,399 core shells) to the full parametric tables in metadata/.
beta0
    Angular half-width of the defect, in degrees.
sep_min_deg
    Minimum angular separation between defects (NaN for a single defect).
topk
    One slot per defect, deepest first, plus the shell descriptors.
nomes_topk
    Names of the features of `topk`.
phys
    Engineering summary of the shell, without the list of defects.
nomes_phys
    Names of the features of `phys`.
interac
    `phys` plus descriptors of the INTERACTION between defects.
nomes_interac
    Names of the features of `interac`.
adim
    DIMENSIONLESS descriptors, with nothing that identifies the shell geometry.
nomes_adim
    Names of the features of `adim`.
matriz
    Feature matrix of a set of shells in one representation.
nomes
    Feature names of one representation.
alvo
    Knockdown factors of a set of shells.
"""
import csv
import math
import os
from functools import lru_cache

import numpy as np

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
K_DEFEITO = 6   # delta, lambda, beta0, ux, uy, uz
N_CASCA = 6     # R, t, nu, eta, p_classical, N
PAD = ("zero", "mask", "nan")
KS = (10, 20, 40, 81)
K_MAX = 81


def _carrega(raiz=None):
    """Read the defect list, the shell parameters, the knockdown factors and the classical pressures.

    Parameters
    ----------
    raiz : str, default=None
        Directory of the parametric tables; the release metadata if None.

    Returns
    -------
    DEF : dict
        Shell id -> list of (delta, lambda, theta, phi).
    CASCA : dict
        Shell id -> R, nu, eta and t.
    KD : dict
        Shell id -> knockdown factor.
    PC : dict
        Shell id -> classical buckling pressure.
    """
    raiz = raiz or (B + "/BLISS-1.0/metadata")
    import os.path
    nome_def = "/defect_parameters.csv" if os.path.exists(raiz + "/defect_parameters.csv") else "/defects.csv"
    dfs = {}
    for r in csv.DictReader(open(raiz + nome_def)):
        dfs.setdefault(int(r["id"]), []).append(
            (float(r["delta"]), float(r["lambda"]), float(r["theta"]), float(r["phi"])))
    casca, kd = {}, {}
    for r in csv.DictReader(open(raiz + "/shells.csv")):
        if not r["knock_down"]:
            continue
        i = int(r["id"]); R = float(r["radius"]); nu = float(r["nu"]); eta = float(r["eta"])
        casca[i] = dict(R=R, nu=nu, eta=eta, t=R / eta)
        kd[i] = float(r["knock_down"])
    pc = {}
    rep = B + "/metadata/reproduction.csv"
    if os.path.exists(rep):
        for r in csv.DictReader(open(rep)):
            if r.get("p_classical"):
                pc[int(r["shell"])] = float(r["p_classical"])
    E_MPA = 1.25                               # neo-Hookean, ver appendix/datasheet.tex
    for i, c in casca.items():
        if i not in pc:
            pc[i] = 2.0 * E_MPA / np.sqrt(3 * (1 - c["nu"] ** 2)) * (c["t"] / c["R"]) ** 2
    return dfs, casca, kd, pc


DEF, CASCA, KD, PC = _carrega()
# BLISS_KAPPA_NOSSO=1: the multi-defect kappa comes from OUR solves (first limit point, as in the
# released records) instead of the tabulated value; singles and doubles keep the tabulated kappa.
# Without the variable nothing changes. Table: outputs/kappa_nosso_multi.csv.
if os.environ.get("BLISS_KAPPA_NOSSO"):
    for _r in csv.DictReader(open(B + "/outputs/kappa_nosso_multi.csv")):
        if int(_r["id"]) in KD:
            KD[int(_r["id"])] = float(_r["kappa_nosso"])


def usar_catalogo():
    """Switch the source from the release (4,399 core shells) to the full parametric tables in metadata/.

    Needed to read the single- and two-defect shells. Their eta and radius differ from the core, and
    eta is collinear with the number of defects, so any difference measured between them and the
    core cannot be attributed to the count alone.

    Returns
    -------
    n : int
        Number of shells with a knockdown factor after the switch.
    """
    global DEF, CASCA, KD, PC
    DEF, CASCA, KD, PC = _carrega(B + "/metadata")
    _ordenado.cache_clear()
    return len(KD)


def beta0(lam, nu, eta):
    """Angular half-width of the defect, in degrees.

    Parameters
    ----------
    lam : float
        Width parameter lambda of the defect.
    nu : float
        Poisson's ratio.
    eta : float
        Slenderness R/t.

    Returns
    -------
    beta0 : float
        Half-width in degrees.
    """
    return lam * (12 * (1 - nu ** 2)) ** -0.25 * eta ** -0.5 * 180.0 / math.pi


def _versor(th, ph):
    """Unit vector of a position on the sphere.

    Parameters
    ----------
    th : float
        Azimuth theta (rad).
    ph : float
        Polar angle phi (rad).

    Returns
    -------
    u : tuple of float
        (ux, uy, uz).
    """
    sp = math.sin(ph)
    return (sp * math.cos(th), sp * math.sin(th), math.cos(ph))


def sep_min_deg(ds):
    """Minimum angular separation between defects (NaN for a single defect).

    Parameters
    ----------
    ds : list
        Defects as (delta, lambda, theta, phi).

    Returns
    -------
    sep : float
        Minimum centre-to-centre angle (deg).
    """
    if len(ds) < 2:
        return np.nan
    u = np.array([_versor(t, p) for _, _, t, p in ds])
    c = np.clip(u @ u.T, -1, 1)
    np.fill_diagonal(c, -1)
    return float(np.degrees(np.arccos(c.max())))


def _casca_vec(sid):
    """Shell-level descriptors.

    Parameters
    ----------
    sid : int
        Shell identifier.

    Returns
    -------
    v : list of float
        R, t, nu, eta, p_classical and the number of defects.
    """
    c = CASCA[sid]
    return [c["R"], c["t"], c["nu"], c["eta"], PC[sid], float(len(DEF[sid]))]


NOMES_CASCA = ["R", "t", "nu", "eta", "p_classical", "N"]


@lru_cache(maxsize=None)
def _ordenado(sid):
    """Defects sorted by decreasing depth, with beta0 and the unit vector precomputed. Cached,
    because the same shell is requested once per (K, padding) combination.

    Parameters
    ----------
    sid : int
        Shell identifier.

    Returns
    -------
    defects : list of tuple
        (delta, lambda, beta0, ux, uy, uz) per defect, deepest first.
    """
    c = CASCA[sid]
    saida = []
    for dl, lam, th, ph in sorted(DEF[sid], key=lambda x: -x[0]):
        ux, uy, uz = _versor(th, ph)
        saida.append((dl, lam, beta0(lam, c["nu"], c["eta"]), ux, uy, uz))
    return saida


def topk(sid, k=20, pad="mask"):
    """One slot per defect, deepest first, plus the shell descriptors.

    Parameters
    ----------
    sid : int
        Shell identifier.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    x : numpy.ndarray
        Feature vector of the shell.
    """
    ds = _ordenado(sid)[:k]
    vazio = np.nan if pad == "nan" else 0.0
    v = np.full(k * K_DEFEITO, vazio)
    m = np.zeros(k)
    for j, linha in enumerate(ds):
        v[j * K_DEFEITO:(j + 1) * K_DEFEITO] = linha
        m[j] = 1.0
    partes = [v, m, _casca_vec(sid)] if pad == "mask" else [v, _casca_vec(sid)]
    return np.concatenate(partes)


def nomes_topk(k=20, pad="mask"):
    """Names of the features of `topk`.

    Parameters
    ----------
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    names : list of str
        Feature names.
    """
    n = [f"{campo}_{j}" for j in range(k)
         for campo in ("delta", "lambda", "beta0", "ux", "uy", "uz")]
    if pad == "mask":
        n += [f"presente_{j}" for j in range(k)]
    return n + NOMES_CASCA


def phys(sid, k=20, pad="mask"):
    """Engineering summary of the shell, without the list of defects.

    Defined for any count, including a shell with a single defect.

    Parameters
    ----------
    sid : int
        Shell identifier.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    x : numpy.ndarray
        Feature vector of the shell.
    """
    ds = sorted(DEF[sid], key=lambda x: -x[0])
    c = CASCA[sid]
    d = [x[0] for x in ds]
    lam = [x[1] for x in ds]
    b = [beta0(x, c["nu"], c["eta"]) for x in lam]
    vazio = 0.0 if pad in ("zero", "mask") else np.nan
    g = lambda v, j: v[j] if len(v) > j else vazio
    sep = sep_min_deg(ds)
    if pad in ("zero", "mask") and not np.isfinite(sep):
        sep = 0.0
    r = [d[0], g(d, 1), g(d, 2), (g(d, 1) / d[0]) if len(d) > 1 else vazio,
         sep, float(np.mean(d)), float(np.std(d)), float(np.sum(d)),
         lam[0], g(lam, 1), b[0], g(b, 1), float(np.mean(b))]
    if pad == "mask":
        r += [1.0, float(len(d) > 1), float(len(d) > 2)]
    return np.array(r + _casca_vec(sid), np.float64)


def nomes_phys(k=20, pad="mask"):
    """Names of the features of `phys`.

    Parameters
    ----------
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    names : list of str
        Feature names.
    """
    n = ["delta_1", "delta_2", "delta_3", "d2_sobre_d1", "sep_min_deg", "delta_mean",
         "delta_std", "delta_sum", "lambda_1", "lambda_2", "beta0_1", "beta0_2", "beta0_mean"]
    if pad == "mask":
        n += ["presente_1", "presente_2", "presente_3"]
    return n + NOMES_CASCA


def interac(sid, k=20, pad="mask"):
    """`phys` plus descriptors of the INTERACTION between defects.

    `phys` gives one number about the spatial structure, the minimum separation. These descriptors
    add how many neighbours each deep defect has within the interaction length.

    Parameters
    ----------
    sid : int
        Shell identifier.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    x : numpy.ndarray
        Feature vector of the shell.
    """
    base = phys(sid, k, pad)
    ds = sorted(DEF[sid], key=lambda x: -x[0])
    c = CASCA[sid]
    b0 = beta0(ds[0][1], c["nu"], c["eta"])
    import math as _m
    u = [_versor(d[2], d[3]) for d in ds]
    def sep(i, j):
        """Angle between two defects (deg).

        Parameters
        ----------
        i, j : int
            Indices of the two defects.

        Returns
        -------
        angle : float
            Centre-to-centre angle.
        """
        d = sum(a * b for a, b in zip(u[i], u[j]))
        return _m.degrees(_m.acos(max(-1.0, min(1.0, d))))
    # contadores davam ZERO nos dois regimes e os descritores eram inuteis.
    lc = _m.pi * b0
    seps1 = [sep(0, j) for j in range(1, len(ds))]
    top = ds[:k]
    sept = [sep(i, j) for i in range(len(top)) for j in range(i + 1, len(top))]
    infl = sum(ds[j + 1][0] * _m.exp(-(seps1[j] / lc) ** 2) for j in range(len(seps1)))
    extra = [float(sum(1 for x in seps1 if x < m * lc)) for m in (1, 2, 3)]
    extra += [float(np.mean(sept)) if sept else 0.0, float(min(sept)) if sept else 0.0,
              float(infl), float(infl / ds[0][0]) if ds[0][0] else 0.0]
    return np.concatenate([base, np.array(extra, np.float64)])


def nomes_interac(k=20, pad="mask"):
    """Names of the features of `interac`.

    Parameters
    ----------
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    names : list of str
        Feature names.
    """
    return nomes_phys(k, pad) + ["n_a_1lc", "n_a_2lc", "n_a_3lc", "sep_media_topk",
                                 "sep_min_topk", "influencia", "influencia_norm"]

def adim(sid, k=20, pad="mask"):
    """DIMENSIONLESS descriptors, with nothing that identifies the shell geometry.

    Used to compare shells with one, two and many defects, which differ in radius and slenderness.

    Parameters
    ----------
    sid : int
        Shell identifier.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    x : numpy.ndarray
        Feature vector of the shell.
    """
    import math as _m
    ds = sorted(DEF[sid], key=lambda x: -x[0])
    c = CASCA[sid]
    b0 = beta0(ds[0][1], c["nu"], c["eta"])
    lc = _m.pi * b0
    u = [_versor(d[2], d[3]) for d in ds]
    def sep(i, j):
        """Angle between two defects in units of the interaction length.

        Parameters
        ----------
        i, j : int
            Indices of the two defects.

        Returns
        -------
        angle : float
            Angle over the interaction length.
        """
        d = sum(a * b for a, b in zip(u[i], u[j]))
        return _m.degrees(_m.acos(max(-1.0, min(1.0, d)))) / lc          # em unidades de lc
    d = [x[0] for x in ds]
    b = [beta0(x[1], c["nu"], c["eta"]) / b0 for x in ds]                # beta0 relativo ao 1o
    s1 = sorted(sep(0, j) for j in range(1, len(ds)))
    vazio = np.nan if pad == "nan" else 0.0
    g = lambda v, j: v[j] if len(v) > j else vazio
    r = [d[0], g(d, 1) / d[0] if len(d) > 1 else vazio, g(d, 2) / d[0] if len(d) > 2 else vazio,
         float(np.mean(d) / d[0]), float(np.std(d) / d[0]),
         g(s1, 0), g(s1, 1), g(s1, 2),
         float(np.mean(s1)) if s1 else vazio,
         g(b, 1), float(np.mean(b)),
         float(sum(1 for x in s1 if x < 1.0)), float(sum(1 for x in s1 if x < 2.0)),
         float(sum(ds[j + 1][0] / d[0] * _m.exp(-(s1[j]) ** 2) for j in range(len(s1))))]
    return np.array(r, np.float64)


def nomes_adim(k=20, pad="mask"):
    """Names of the features of `adim`.

    Parameters
    ----------
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    names : list of str
        Feature names.
    """
    return ["delta_1", "d2_sobre_d1", "d3_sobre_d1", "d_media_norm", "d_std_norm",
            "sep1_lc", "sep2_lc", "sep3_lc", "sep_media_lc", "b2_sobre_b1", "b_media_norm",
            "n_dentro_1lc", "n_dentro_2lc", "influencia_norm"]

def matriz(ids, rep="topk", k=20, pad="mask"):
    """Feature matrix of a set of shells in one representation.

    Parameters
    ----------
    ids : sequence of int
        Shell identifiers.
    rep : str, {'topk', 'phys', 'interac', 'adim'}
        Representation.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    X : numpy.ndarray, shape (n_shells, n_features)
        Features.
    """
    f = {"topk": topk, "phys": phys, "interac": interac, "adim": adim}[rep]
    return np.stack([f(i, k, pad) for i in ids])


def nomes(rep="topk", k=20, pad="mask"):
    """Feature names of one representation.

    Parameters
    ----------
    rep : str, {'topk', 'phys', 'interac', 'adim'}
        Representation.
    k : int
        Number of defect slots.
    pad : str, {'zero', 'mask', 'nan'}
        How the unused slots are filled.

    Returns
    -------
    names : list of str
        Feature names.
    """
    return {"topk": nomes_topk, "phys": nomes_phys, "interac": nomes_interac, "adim": nomes_adim}[rep](k, pad)


def alvo(ids):
    """Knockdown factors of a set of shells.

    Parameters
    ----------
    ids : sequence of int
        Shell identifiers.

    Returns
    -------
    kappa : numpy.ndarray
        Knockdown factors.
    """
    return np.array([KD[i] for i in ids])
