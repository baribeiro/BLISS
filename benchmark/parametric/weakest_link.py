"""The weakest link: kappa_WL = kappa_1(delta_max, beta0), fitted on the single-defect shells.

A shell with many defects cannot be stronger than the same shell with only its worst defect, so kappa_1 of the
deepest defect is the zero-order prediction, and what a model has to learn is the remainder:

    Delta_kappa = kappa_FE - kappa_WL          and the prediction is   kappa_WL + f(defects, ...)

On single-defect shells this residual is zero by construction; on two-defect shells it is the interaction of a
pair; on the multi-defect core the question is whether that interaction composes.

FIT. The 399 single-defect shells give kappa against (delta, beta0) directly. A smooth surface is fitted in these
two variables and evaluated at the deepest defect of any shell.

ASSUMPTION. The single-defect shells have eta = 108 and R = 24.85 mm, the core has eta = 110 and R = 25.4 mm, so
kappa_1 is transported across a 2% difference in slenderness. In beta0 there is no transport: the single-defect
shells cover 0.80 to 9.55 deg and the core sits at 3.154 deg. Outside that range the function returns NaN.

Classes
-------
EloFraco
    The weakest-link rule kappa_1(delta_max, beta0), fitted on the single-defect shells.
"""
import os

import numpy as np
import pandas as pd

B = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MESTRE = B + "/outputs/parametric_master.parquet"


class EloFraco:
    """The weakest-link rule kappa_1(delta_max, beta0), fitted on the single-defect shells.

    Evaluated at the deepest defect of any shell; outside the fitted domain it returns NaN (or
    clips).
    """
    def __init__(s, mestre=None, origem="single", grau=3):
        """Fit the rule on the shells of one origin in the parametric master table.

        Parameters
        ----------
        mestre : str, default=None
            Path of parametric_master.parquet.
        origem : str, default='single'
            Origin of the shells used for the fit.
        grau : int, default=3
            Degree of the polynomial.

        Raises
        ------
        ValueError
            If there are no shells of that origin.
        """
        df = pd.read_parquet(mestre or MESTRE)
        u = df[df.origin == origem]
        if not len(u):
            raise ValueError('no shells of origin %s' % origem)
        s.d = np.array([r[0] for r in u["delta"]], float)
        s.b = np.array([r[0] for r in u["beta0_deg"]], float)
        s.k = u["kappa"].to_numpy(float)
        s.lim_d = (s.d.min(), s.d.max())
        s.lim_b = (s.b.min(), s.b.max())
        s.grau = grau
        s._ajusta()

    def _base(s, d, b):
        """Bivariate polynomial in (log delta, log beta0), since both variables span more than a decade.

        Parameters
        ----------
        d : numpy.ndarray
            Depths delta.
        b : numpy.ndarray
            Half-widths beta0 (deg).

        Returns
        -------
        A : numpy.ndarray
            Design matrix of the polynomial.
        """
        ld, lb = np.log(d), np.log(b)
        cols = [np.ones_like(ld)]
        for i in range(1, s.grau + 1):
            for j in range(i + 1):
                cols.append(ld ** (i - j) * lb ** j)
        return np.stack(cols, 1)

    def _ajusta(s):
        """Least-squares fit of the polynomial and its training error."""
        A = s._base(s.d, s.b)
        s.coef, *_ = np.linalg.lstsq(A, s.k, rcond=None)
        p = A @ s.coef
        s.rmse = float(np.sqrt(np.mean((p - s.k) ** 2)))
        s.r2 = float(1 - ((p - s.k) ** 2).sum() / ((s.k - s.k.mean()) ** 2).sum())
        s.err = float(np.median(np.abs(p - s.k) / s.k) * 100)

    def __call__(s, delta_max, beta0, fora="nan"):
        """kappa_1 at the deepest defect. Outside the single-defect domain returns NaN (or clips,
        with `fora='clip'`), since a silent extrapolation is worse than a visible gap.

        Parameters
        ----------
        delta_max : float or numpy.ndarray
            Depth of the deepest defect.
        beta0 : float or numpy.ndarray
            Half-width of the deepest defect (deg).
        fora : str, {'nan', 'clip'}, default='nan'
            What to do outside the fitted domain.

        Returns
        -------
        kappa_wl : numpy.ndarray
            Weakest-link knockdown factor.
        """
        d = np.atleast_1d(np.asarray(delta_max, float)).copy()
        b = np.atleast_1d(np.asarray(beta0, float)).copy()
        dentro = ((d >= s.lim_d[0]) & (d <= s.lim_d[1]) &
                  (b >= s.lim_b[0]) & (b <= s.lim_b[1]))
        if fora == "clip":
            d = np.clip(d, *s.lim_d); b = np.clip(b, *s.lim_b); dentro[:] = True
        out = np.full(len(d), np.nan)
        if dentro.any():
            out[dentro] = s._base(d[dentro], b[dentro]) @ s.coef
        return out

    def para_tabela(s, df, fora="clip"):
        """kappa_WL for each row of the parametric master table.

        Parameters
        ----------
        df : pandas.DataFrame
            Parametric master table.
        fora : str, {'nan', 'clip'}, default='nan'
            What to do outside the fitted domain.

        Returns
        -------
        kappa_wl : numpy.ndarray
            Weakest-link knockdown factor of each row.
        """
        d = np.array([r[0] for r in df["delta"]], float)
        b = np.array([r[0] for r in df["beta0_deg"]], float)
        return s(d, b, fora=fora)


if __name__ == "__main__":
    df = pd.read_parquet(MESTRE)
    el = EloFraco()
    print('kappa_1 fitted on %d single-defect shells, polynomial of degree %d in (log d, log b)'
          % (len(el.d), el.grau))
    print("  ajuste nos proprios singles: R2 %.4f   erro %.3f%%   rmse %.5f"
          % (el.r2, el.err, el.rmse))
    print("  dominio: delta %.3f..%.3f   beta0 %.2f..%.2f graus" % (*el.lim_d, *el.lim_b))
    print()
    df["kappa_wl"] = el.para_tabela(df)
    print('=== first rung: the worst defect only ===')
    print("%-12s %6s %10s %10s %12s %10s" % ("origem", "n", "erro %", "R2", "vies medio", "dentro"))
    for o, g in df.groupby("origin"):
        k, w = g["kappa"].to_numpy(), g["kappa_wl"].to_numpy()
        m = np.isfinite(w)
        dd = np.array([r[0] for r in g["delta"]], float)
        bb = np.array([r[0] for r in g["beta0_deg"]], float)
        dentro = ((dd >= el.lim_d[0]) & (dd <= el.lim_d[1]) &
                  (bb >= el.lim_b[0]) & (bb <= el.lim_b[1])).mean()
        e = np.median(np.abs(w[m] - k[m]) / k[m]) * 100
        r2 = 1 - ((w[m] - k[m]) ** 2).sum() / ((k[m] - k[m].mean()) ** 2).sum()
        print("%-12s %6d %9.2f%% %10.4f %12.4f %9.0f%%"
              % (o, len(g), e, r2, np.median(w[m] - k[m]), 100 * dentro))
    print()
    print('The residual the models must learn is kappa - kappa_wl. On the single-defect shells it is noise of')
    print('fit; on the two-defect shells it is the interaction of a pair; on the multi-defect shells it is the question of the paper.')
