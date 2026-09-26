"""Models on the SET of defects: flat MLP, DeepSets and a pairwise interaction network.

All three use the same table and target, so that the difference between them is the model:

  flat       concatenates the slots by depth and pads the rest. The control: it sees what the tabular models see.
  deepsets   h = rho( sum_i phi(d_i), g )                    (Zaheer et al., NeurIPS 2017)
             permutation invariant and accepts any number of defects.
  pairwise   h = rho( sum_i phi(d_i) + sum_{i<j} psi(d_i, d_j, sep_ij), g )
             adds the pairwise term, the interaction the two-defect shells isolate.

TARGET. With `--target residual` the model learns Delta = kappa - kappa_WL and predicts kappa_WL + f(.), with
kappa_WL the weakest-link rule fitted on the single-defect shells (weakest_link.py). With `--target kappa` it
learns kappa directly.

The pairwise term (up to 3,240 pairs per shell for N = 81) is computed in masked batches.

Classes
-------
Flat
    Control: the slots concatenated by depth, plus the shell descriptors.
DeepSets
    DeepSets: h = rho(aggregation_i phi(d_i), g).
PairwiseNet
    h = rho( sum_i phi(d_i) + sum_{i<j} psi(d_i, d_j, sep_ij), g ).

Functions
---------
tensores
    From the parametric master table to masked tensors.
normaliza
    Mean and standard deviation over the REAL defects (the mask counts), and over the shells.
constroi
    Build an architecture from an Optuna trial. Each family has its own namespace.
"""
import math

import numpy as np
import torch
import torch.nn as nn

D_DEFEITO = 6
D_CASCA = 6


def tensores(df, idx, elo=None, device="cpu", Nmax=None):
    """From the parametric master table to masked tensors.

    Returns (X, G, M, y, kwl) with X (n, Nmax, 6), G (n, 6), M (n, Nmax) boolean,
    y the kappa and kwl the weakest-link reference.

    Parameters
    ----------
    df : pandas.DataFrame
        Parametric master table.
    idx : sequence
        Row labels to use.
    elo : EloFraco or None
        Weakest-link rule for the residual target.
    device : str
        Torch device.
    Nmax : int, default=None
        Number of defect slots; the largest N if None.

    Returns
    -------
    X, G, M, y, kwl : torch.Tensor
        Defect features, shell features, mask, kappa and weakest-link reference.
    """
    g = df.loc[idx]
    n = len(g)
    Nmax = int(Nmax or g["N"].max())
    X = np.zeros((n, Nmax, D_DEFEITO), np.float32)
    M = np.zeros((n, Nmax), bool)
    for a, (_, r) in enumerate(g.iterrows()):
        k = int(r["N"])
        X[a, :k, 0] = r["delta"]
        X[a, :k, 1] = r["lam"]
        X[a, :k, 2] = r["beta0_deg"]
        X[a, :k, 3] = r["x"]
        X[a, :k, 4] = r["y"]
        X[a, :k, 5] = r["z"]
        M[a, :k] = True
    G = g[["R", "t", "R_over_t", "nu", "p_classical", "N"]].to_numpy(np.float32)
    y = g["kappa"].to_numpy(np.float32)
    kwl = elo.para_tabela(g) .astype(np.float32) if elo is not None else np.full(n, np.nan, np.float32)
    t = lambda a, d=None: torch.tensor(a, dtype=d).to(device)
    return (t(X, torch.float32), t(G, torch.float32), t(M, torch.bool),
            t(y, torch.float32), t(kwl, torch.float32))


def normaliza(Xtr, Gtr, Mtr):
    """Mean and standard deviation over the REAL defects (the mask counts), and over the shells.

    Parameters
    ----------
    Xtr : torch.Tensor
        Training defect features.
    Gtr : torch.Tensor
        Training shell features.
    Mtr : torch.Tensor
        Training mask.

    Returns
    -------
    mu_x, sd_x, mu_g, sd_g : torch.Tensor
        Means and standard deviations.
    """
    v = Xtr[Mtr]
    mu_x, sd_x = v.mean(0), v.std(0) + 1e-6
    mu_g, sd_g = Gtr.mean(0), Gtr.std(0) + 1e-6
    return (mu_x, sd_x, mu_g, sd_g)


def _mlp(d_in, d_out, largura, prof, dr, norma=True):
    """An MLP with GELU, dropout and optional layer normalisation.

    Parameters
    ----------
    d_in : int
        Input width.
    d_out : int
        Output width.
    largura : int
        Hidden width.
    prof : int
        Number of hidden layers.
    dr : float
        Dropout.
    norma : bool, default=True
        Use layer normalisation.

    Returns
    -------
    net : torch.nn.Sequential
        The network.
    """
    lay, d = [], d_in
    for _ in range(prof):
        lay.append(nn.Linear(d, largura))
        if norma:
            lay.append(nn.LayerNorm(largura))
        lay += [nn.GELU(), nn.Dropout(dr)]
        d = largura
    return nn.Sequential(*lay, nn.Linear(d, d_out))


class Flat(nn.Module):
    """Control: the slots concatenated by depth, plus the shell descriptors."""

    def __init__(s, Nmax, largura=256, prof=4, dr=0.1):
        """Build the flat MLP.

        Parameters
        ----------
        Nmax : int
            Defect slots.
        largura : int
            Hidden width.
        prof : int
            Depth.
        dr : float
            Dropout.
        """
        super().__init__()
        s.Nmax = Nmax
        s.net = _mlp(Nmax * D_DEFEITO + Nmax + D_CASCA, 1, largura, prof, dr)

    def forward(s, X, G, M):
        """Predict the target of a batch of shells.

        Parameters
        ----------
        X : torch.Tensor, shape (B, Nmax, 6)
            Defect features.
        G : torch.Tensor, shape (B, 6)
            Shell features.
        M : torch.Tensor, shape (B, Nmax)
            Mask of the real defects.

        Returns
        -------
        y : torch.Tensor, shape (B,)
            Predicted target.
        """
        x = (X * M.unsqueeze(-1)).flatten(1)
        return s.net(torch.cat([x, M.float(), G], -1)).squeeze(-1)


class DeepSets(nn.Module):
    """DeepSets: h = rho(aggregation_i phi(d_i), g).

    Permutation invariant and defined for any number of defects (Zaheer et al., NeurIPS 2017).
    """

    def __init__(s, largura=256, prof_phi=3, prof_rho=3, dr=0.1, agreg="soma_e_max"):
        """Build phi and rho.

        Parameters
        ----------
        largura : int
            Hidden width.
        prof_phi : int
            Depth of phi.
        prof_rho : int
            Depth of rho.
        dr : float
            Dropout.
        agreg : str
            Aggregation over the defects ('soma' sum, 'soma_e_max' sum and max, ...).
        """
        super().__init__()
        s.phi = _mlp(D_DEFEITO, largura, largura, prof_phi, dr)
        s.agreg = agreg
        mult = 2 if agreg == "soma_e_max" else 1
        s.rho = _mlp(largura * mult + D_CASCA, 1, largura, prof_rho, dr)

    def _agrega(s, h, M):
        """Aggregate the per-defect features over the real defects.

        Parameters
        ----------
        h : torch.Tensor
            Per-defect features.
        M : torch.Tensor
            Mask.

        Returns
        -------
        a : torch.Tensor
            Aggregated features.
        """
        m = M.unsqueeze(-1)
        soma = (h * m).sum(1)
        if s.agreg == "soma":
            return soma
        if s.agreg == "media":
            return soma / M.sum(1, keepdim=True).clamp(min=1)
        mx = h.masked_fill(~m, -1e9).amax(1)
        return torch.cat([soma, mx], -1)

    def forward(s, X, G, M):
        """Predict the target of a batch of shells.

        Parameters
        ----------
        X : torch.Tensor, shape (B, Nmax, 6)
            Defect features.
        G : torch.Tensor, shape (B, 6)
            Shell features.
        M : torch.Tensor, shape (B, Nmax)
            Mask of the real defects.

        Returns
        -------
        y : torch.Tensor, shape (B,)
            Predicted target.
        """
        return s.rho(torch.cat([s._agrega(s.phi(X), M), G], -1)).squeeze(-1)


class PairwiseNet(nn.Module):
    """h = rho( sum_i phi(d_i) + sum_{i<j} psi(d_i, d_j, sep_ij), g ).

    The pairwise term is symmetrised (psi(a,b) + psi(b,a)) so that it does not depend on the order."""

    def __init__(s, largura=256, prof_phi=2, prof_psi=2, prof_rho=3, dr=0.1, max_pares=4096):
        """Build phi, psi and rho.

        Parameters
        ----------
        largura : int
            Hidden width.
        prof_phi : int
            Depth of phi.
        prof_psi : int
            Depth of psi.
        prof_rho : int
            Depth of rho.
        dr : float
            Dropout.
        max_pares : int
            Largest number of pairs evaluated at once.
        """
        super().__init__()
        s.phi = _mlp(D_DEFEITO, largura, largura, prof_phi, dr)
        s.psi = _mlp(2 * D_DEFEITO + 1, largura, largura, prof_psi, dr)
        s.rho = _mlp(2 * largura + D_CASCA, 1, largura, prof_rho, dr)
        s.max_pares = max_pares

    def forward(s, X, G, M):
        """Predict the target of a batch of shells.

        Parameters
        ----------
        X : torch.Tensor, shape (B, Nmax, 6)
            Defect features.
        G : torch.Tensor, shape (B, 6)
            Shell features.
        M : torch.Tensor, shape (B, Nmax)
            Mask of the real defects.

        Returns
        -------
        y : torch.Tensor, shape (B,)
            Predicted target.
        """
        B, N, _ = X.shape
        h1 = (s.phi(X) * M.unsqueeze(-1)).sum(1)
        iu = torch.triu_indices(N, N, offset=1, device=X.device)
        if iu.shape[1] > s.max_pares:
            iu = iu[:, torch.randperm(iu.shape[1], device=X.device)[:s.max_pares]]
        a, b = X[:, iu[0]], X[:, iu[1]]
        cos = (a[..., 3:6] * b[..., 3:6]).sum(-1).clamp(-1, 1)
        sep = torch.arccos(cos).unsqueeze(-1)
        mp = (M[:, iu[0]] & M[:, iu[1]]).unsqueeze(-1)
        p = s.psi(torch.cat([a, b, sep], -1)) + s.psi(torch.cat([b, a, sep], -1))
        h2 = (p * mp).sum(1) * 0.5
        return s.rho(torch.cat([h1, h2, G], -1)).squeeze(-1)


def constroi(nome, t, Nmax):
    """Build an architecture from an Optuna trial. Each family has its own namespace.

    Parameters
    ----------
    nome : str, {'flat', 'deepsets', 'pairwise'}
        Architecture.
    t : optuna.Trial
        The trial.
    Nmax : int
        Defect slots.

    Returns
    -------
    model : torch.nn.Module
        The architecture.
    """
    w = t.suggest_categorical(nome + "_largura", [64, 128, 256, 512])
    dr = t.suggest_float(nome + "_dropout", 0.0, 0.3)
    if nome == "flat":
        return Flat(Nmax, w, t.suggest_int("flat_prof", 2, 6), dr)
    if nome == "deepsets":
        return DeepSets(w, t.suggest_int("ds_prof_phi", 1, 4), t.suggest_int("ds_prof_rho", 1, 4),
                        dr, t.suggest_categorical("ds_agreg", ["soma", "media", "soma_e_max"]))
    if nome == "pairwise":
        return PairwiseNet(w, t.suggest_int("pw_prof_phi", 1, 3), t.suggest_int("pw_prof_psi", 1, 3),
                           t.suggest_int("pw_prof_rho", 1, 4), dr)
    raise ValueError(nome)
