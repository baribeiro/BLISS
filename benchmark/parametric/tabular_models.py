"""The tabular model families, with the sklearn fit/predict interface.

The families are those used by the reference comparisons on tabular data:

  Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022), "Why do tree-based models still outperform deep learning on
      tabular data?"
  Borisov et al. (2022), "Deep Neural Networks and Tabular Data: A Survey".
  McElfresh et al. (NeurIPS 2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
  Gorishniy, Rubachev, Khrulkov, Babenko (NeurIPS 2021), "Revisiting Deep Learning Models for Tabular Data"
      (tabular ResNet and FT-Transformer).
  Arik & Pfister (AAAI 2021), "TabNet".
  Hollmann et al. (Nature 2025), "TabPFN".

Gaussian processes are included as the usual surrogate in engineering design.

Interface: build(fam, trial, seed, device, n_features, n_train) returns an object with fit/predict. Each family
declares its hyperparameters in its own namespace, since equal names with different domains make Optuna reject
the trial.

Classes
-------
_Torch
    Training loop shared by the MLP, the ResNet and the FT-Transformer.
_TabNet
    TabNet (Arik & Pfister, AAAI 2021) with the fit/predict interface.
_TabPFN
    Network pre-fitted to a prior (Hollmann et al.). It has no training hyperparameters: fitting

Functions
---------
familias
    The families of the group, without those that do not fit this data set or are not installed.
build
    Build one model of a family, with its hyperparameters drawn from the trial.
"""
import os

import numpy as np

TABPFN_MAX_FEATURES = 500
TABPFN_MAX_AMOSTRAS = 10000

GRUPOS = {
    "chao": ["ridge", "elasticnet", "knn", "svr", "gp"],
    "arvores": ["rf", "extratrees", "sk_hgb", "xgboost", "lightgbm", "catboost"],
    "redes": ["mlp", "resnet", "ft_transformer", "tabnet", "tabpfn"],
}
GRUPOS["gpu_arvores"] = ["xgboost", "lightgbm", "catboost"]
# xgboost (1.050% contra 1.720% em 800 cascas).
GRUPOS["tab_gpu"] = ["xgboost", "lightgbm", "catboost", "ft_transformer"]
GRUPOS["tudo"] = GRUPOS["chao"] + GRUPOS["arvores"] + GRUPOS["redes"]
GRUPOS["trees"] = GRUPOS["chao"] + GRUPOS["arvores"]
GRUPOS["mlp"] = ["mlp"]
GRUPOS["cpu_resto"] = [m for m in GRUPOS["tudo"] if m not in GRUPOS["tab_gpu"]]

for _f in GRUPOS["tudo"]:
    GRUPOS.setdefault("so_" + _f, [_f])


def familias(grupo, n_features, n_train):
    """The families of the group, without those that do not fit this data set or are not installed.

    Parameters
    ----------
    grupo : str
        Name of the group of families.
    n_features : int
        Number of features.
    n_train : int
        Number of training shells.

    Returns
    -------
    families : list of str
        Families to search.
    """
    fs = list(GRUPOS[grupo])
    if "tabpfn" in fs:
        if n_features > TABPFN_MAX_FEATURES or n_train > TABPFN_MAX_AMOSTRAS:
            fs.remove("tabpfn")
        elif not os.environ.get("TABPFN_TOKEN"):
            fs.remove("tabpfn")
    return fs


# ---------------------------------------------------------------- redes, em torch
class _Torch:
    """Training loop shared by the MLP, the ResNet and the FT-Transformer.

    Missing values are imputed with the training MEAN, which is exactly 0 after normalisation."""

    def __init__(s, faz_rede, t, seed, device, prefixo, lr_max=3e-2):
        """Draw the training hyperparameters from the trial.

        Parameters
        ----------
        faz_rede : callable
            Builder of the network, faz_rede(d_in, dropout).
        t : optuna.Trial
            The trial.
        seed : int
            Seed.
        device : str
            Torch device.
        prefixo : str
            Namespace of the hyperparameters.
        lr_max : float, default=3e-2
            Upper bound of the learning rate.
        """
        s.faz_rede, s.seed, s.dev = faz_rede, seed, device
        s.dr = t.suggest_float(prefixo + "_dropout", 0.0, 0.5)
        s.wd = t.suggest_float(prefixo + "_weight_decay", 1e-6, 1e-1, log=True)
        s.lr = t.suggest_float(prefixo + "_lr", 1e-4, lr_max, log=True)
        s.ep = t.suggest_int(prefixo + "_epochs", 100, 1000)
        s.bs = t.suggest_categorical(prefixo + "_batch", [32, 64, 128, 256])

    def fit(s, X, y):
        """Train the network on standardised features with mean imputation.

        Parameters
        ----------
        X : numpy.ndarray
            Training features (NaN allowed).
        y : numpy.ndarray
            Training targets.

        Returns
        -------
        self : _Torch
            The fitted model.
        """
        import torch
        import torch.nn as nn
        torch.manual_seed(s.seed)
        torch.set_num_threads(8)
        dev = torch.device(s.dev)
        s.mu, s.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-8
        s._z = lambda M: torch.tensor((np.where(np.isnan(M), s.mu, M) - s.mu) / s.sd,
                                      dtype=torch.float32).to(dev)
        xa = s._z(X)
        ta = torch.tensor(y, dtype=torch.float32)[:, None].to(dev)
        s.net = s.faz_rede(X.shape[1], s.dr).to(dev)
        opt = torch.optim.AdamW(s.net.parameters(), lr=s.lr, weight_decay=s.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, s.ep, eta_min=s.lr * 0.01)
        g = torch.Generator().manual_seed(s.seed)
        for _ in range(s.ep):
            for j in torch.randperm(len(xa), generator=g).to(dev).split(s.bs):
                opt.zero_grad()
                nn.functional.mse_loss(s.net(xa[j]), ta[j]).backward()
                nn.utils.clip_grad_norm_(s.net.parameters(), 1.0)
                opt.step()
            sch.step()
        return s

    def predict(s, M):
        """Predict with the trained network.

        Parameters
        ----------
        M : numpy.ndarray
            Features.

        Returns
        -------
        y : numpy.ndarray
            Predictions.
        """
        import torch
        s.net.eval()
        with torch.no_grad():
            return s.net(s._z(M)).cpu().numpy().ravel()


def _mlp_builder(t, larga):
    """Draw an MLP architecture from the trial.

    Parameters
    ----------
    t : optuna.Trial
        The trial.
    larga : bool
        Allow the wider search range.

    Returns
    -------
    faz : callable
        Builder faz(d_in, dr).
    """
    h = t.suggest_categorical("mlp_hidden", [64, 128, 256, 512, 1024, 2048] if larga
                              else [64, 128, 256, 512])
    L = t.suggest_int("mlp_layers", 2, 10 if larga else 5)
    norm = t.suggest_categorical("mlp_norm", ["none", "layer"])
    act = t.suggest_categorical("mlp_act", ["gelu", "silu", "relu"])

    def faz(d_in, dr):
        """Build the network for a given input width and dropout.

        Parameters
        ----------
        d_in : int
            Number of input features.
        dr : float
            Dropout.

        Returns
        -------
        net : torch.nn.Module
            The network.
        """
        import torch.nn as nn
        A = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}[act]
        lay, d = [], d_in
        for _ in range(L):
            lay.append(nn.Linear(d, h))
            if norm == "layer":
                lay.append(nn.LayerNorm(h))
            lay += [A(), nn.Dropout(dr)]
            d = h
        return nn.Sequential(*lay, nn.Linear(d, 1))
    return faz


def _resnet_builder(t):
    """The tabular ResNet of Gorishniy et al. (2021), from the authors' library.

    Parameters
    ----------
    t : optuna.Trial
        The trial.

    Returns
    -------
    faz : callable
        Builder faz(d_in, dr).
    """
    d = t.suggest_categorical("res_d", [128, 256, 512])
    n = t.suggest_int("res_blocks", 1, 8)
    hid = t.suggest_float("res_hidden_mult", 1.0, 4.0)

    def faz(d_in, dr):
        """Build the network for a given input width and dropout.

        Parameters
        ----------
        d_in : int
            Number of input features.
        dr : float
            Dropout.

        Returns
        -------
        net : torch.nn.Module
            The network.
        """
        from rtdl_revisiting_models import ResNet
        return ResNet(d_in=d_in, d_out=1, n_blocks=n, d_block=d,
                      d_hidden=None, d_hidden_multiplier=hid,
                      dropout1=dr, dropout2=0.0)
    return faz


def _ft_builder(t):
    """The FT-Transformer of Gorishniy et al. (2021). All variables here are continuous.

    Parameters
    ----------
    t : optuna.Trial
        The trial.

    Returns
    -------
    faz : callable
        Builder faz(d_in, dr).
    """
    d = t.suggest_categorical("ft_d_block", [96, 128, 192, 256])
    n = t.suggest_int("ft_blocks", 1, 4)
    heads = t.suggest_categorical("ft_heads", [4, 8])

    def faz(d_in, dr):
        """Build the network for a given input width and dropout.

        Parameters
        ----------
        d_in : int
            Number of input features.
        dr : float
            Dropout.

        Returns
        -------
        net : torch.nn.Module
            The network.
        """
        from rtdl_revisiting_models import FTTransformer
        dd = d - d % heads
        return FTTransformer(n_cont_features=d_in, cat_cardinalities=[], d_out=1,
                             n_blocks=n, d_block=dd, attention_n_heads=heads,
                             attention_dropout=dr, ffn_d_hidden=None,
                             ffn_d_hidden_multiplier=4 / 3, ffn_dropout=dr,
                             residual_dropout=0.0)
    return faz


class _TabNet:
    """TabNet (Arik & Pfister, AAAI 2021) with the fit/predict interface."""
    def __init__(s, t, seed, device):
        """Draw the TabNet hyperparameters from the trial.

        Parameters
        ----------
        t : optuna.Trial
            The trial.
        seed : int
            Seed.
        device : str
            Torch device.
        """
        s.nd = t.suggest_categorical("tabnet_nd", [8, 16, 32, 64])
        s.steps = t.suggest_int("tabnet_steps", 3, 8)
        s.gamma = t.suggest_float("tabnet_gamma", 1.0, 2.0)
        s.lr = t.suggest_float("tabnet_lr", 1e-3, 5e-2, log=True)
        s.ep = t.suggest_int("tabnet_epochs", 50, 400)
        s.seed, s.dev = seed, device

    def fit(s, X, y):
        """Train TabNet on standardised features.

        Parameters
        ----------
        X : numpy.ndarray
            Training features (NaN allowed).
        y : numpy.ndarray
            Training targets.

        Returns
        -------
        self : _TabNet
            The fitted model.
        """
        from pytorch_tabnet.tab_model import TabNetRegressor
        import torch
        s.mu, s.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-8
        s._z = lambda M: ((np.where(np.isnan(M), s.mu, M) - s.mu) / s.sd).astype(np.float32)
        s.m = TabNetRegressor(n_d=s.nd, n_a=s.nd, n_steps=s.steps, gamma=s.gamma,
                              seed=s.seed, verbose=0,
                              optimizer_params=dict(lr=s.lr),
                              device_name="cuda" if s.dev.startswith("cuda") else "cpu")
        s.m.fit(s._z(X), y.reshape(-1, 1).astype(np.float32),
                max_epochs=s.ep, patience=0, batch_size=256, drop_last=False)
        return s

    def predict(s, M):
        """Predict with TabNet.

        Parameters
        ----------
        M : numpy.ndarray
            Features.

        Returns
        -------
        y : numpy.ndarray
            Predictions.
        """
        return s.m.predict(s._z(M)).ravel()


class _TabPFN:
    """Network pre-fitted to a prior (Hollmann et al.). It has no training hyperparameters: fitting
    stores the data and prediction runs one forward pass."""

    def __init__(s, t, seed, device):
        """Draw the ensemble size from the trial.

        Parameters
        ----------
        t : optuna.Trial
            The trial.
        seed : int
            Seed.
        device : str
            Torch device.
        """
        s.n_ens = t.suggest_categorical("tabpfn_ensemble", [1, 2, 4, 8])
        s.seed, s.dev = seed, device

    def fit(s, X, y):
        """Store the data (median imputation) in the pre-fitted network.

        Parameters
        ----------
        X : numpy.ndarray
            Training features (NaN allowed).
        y : numpy.ndarray
            Training targets.

        Returns
        -------
        self : _TabPFN
            The fitted model.
        """
        from tabpfn import TabPFNRegressor
        s.med = np.nanmedian(X, 0)
        s._c = lambda M: np.where(np.isnan(M), s.med, M)
        s.m = TabPFNRegressor(n_estimators=s.n_ens, random_state=s.seed,
                              device="cuda" if s.dev.startswith("cuda") else "cpu")
        s.m.fit(s._c(X), y)
        return s

    def predict(s, M):
        """Predict with one forward pass.

        Parameters
        ----------
        M : numpy.ndarray
            Features.

        Returns
        -------
        y : numpy.ndarray
            Predictions.
        """
        return np.asarray(s.m.predict(s._c(M))).ravel()


# ---------------------------------------------------------------- despacho
def build(fam, t, seed=0, device="cpu", n_features=0, n_train=0):
    """Build one model of a family, with its hyperparameters drawn from the trial.

    Parameters
    ----------
    fam : str
        Model family.
    t : optuna.Trial
        The trial.
    seed : int
        Seed.
    device : str
        Torch device.
    n_features : int
        Number of features.
    n_train : int
        Number of training shells.

    Returns
    -------
    model : object
        An object with fit and predict.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    imp = lambda *r: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), *r)
    cuda = device.startswith("cuda")

    if fam == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        return imp(PolynomialFeatures(t.suggest_categorical("ridge_poly", [1, 2])
                                      if n_features <= 40 else 1),
                   Ridge(alpha=t.suggest_float("ridge_alpha", 1e-4, 1e3, log=True)))
    if fam == "elasticnet":
        from sklearn.linear_model import ElasticNet
        return imp(ElasticNet(alpha=t.suggest_float("en_alpha", 1e-5, 1e1, log=True),
                              l1_ratio=t.suggest_float("en_l1", 0.0, 1.0), max_iter=5000))
    if fam == "knn":
        from sklearn.neighbors import KNeighborsRegressor
        return imp(KNeighborsRegressor(n_neighbors=t.suggest_int("knn_k", 1, 40),
                                       weights=t.suggest_categorical("knn_w", ["uniform", "distance"]),
                                       p=t.suggest_categorical("knn_p", [1, 2])))
    if fam == "svr":
        from sklearn.svm import SVR
        return imp(SVR(C=t.suggest_float("svr_C", 1e-2, 1e3, log=True),
                       gamma=t.suggest_float("svr_gamma", 1e-4, 1e0, log=True),
                       epsilon=t.suggest_float("svr_eps", 1e-4, 1e-1, log=True)))
    if fam == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
        nu = t.suggest_categorical("gp_nu", [0.5, 1.5, 2.5, "rbf"])
        base = RBF(1.0) if nu == "rbf" else Matern(1.0, nu=nu)
        k = ConstantKernel(1.0) * base + WhiteKernel(t.suggest_float("gp_noise", 1e-8, 1e-1, log=True))
        return imp(GaussianProcessRegressor(kernel=k, normalize_y=True, random_state=seed,
                                            n_restarts_optimizer=0))
    if fam in ("rf", "extratrees"):
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        cls = RandomForestRegressor if fam == "rf" else ExtraTreesRegressor
        pre = "rf" if fam == "rf" else "et"
        return make_pipeline(SimpleImputer(strategy="median"), cls(
            n_estimators=t.suggest_int(pre + "_n_estimators", 200, 1500),
            max_depth=t.suggest_categorical(pre + "_max_depth", [6, 10, 16, None]),
            min_samples_leaf=t.suggest_int(pre + "_min_samples_leaf", 1, 20),
            max_features=t.suggest_float(pre + "_max_features", 0.2, 1.0),
            random_state=seed, n_jobs=8))

    if fam in ("sk_hgb", "xgboost", "lightgbm", "catboost"):
        lr = t.suggest_float("gb_learning_rate", 1e-3, 0.3, log=True)
        n = t.suggest_int("gb_n_estimators", 200, 3000)
        md = t.suggest_categorical("gb_max_depth", [3, 4, 5, 6, 8, 10])
        l2 = t.suggest_float("gb_reg_lambda", 1e-4, 100, log=True)
        if fam == "sk_hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            return HistGradientBoostingRegressor(
                learning_rate=lr, max_iter=n, max_depth=md, l2_regularization=l2,
                min_samples_leaf=t.suggest_int("hgb_leaf", 5, 60), early_stopping=False,
                random_state=seed)
        sub = t.suggest_float("gb_subsample", 0.5, 1.0)
        col = t.suggest_float("gb_colsample", 0.4, 1.0)
        if fam == "catboost":
            from catboost import CatBoostRegressor
            kw = dict(learning_rate=lr, iterations=n, depth=min(md, 10), l2_leaf_reg=l2,
                      random_seed=seed, verbose=0, allow_writing_files=False,
                      task_type="GPU" if cuda else "CPU")
            if not cuda:
                kw["rsm"] = col
            return CatBoostRegressor(**kw)
        mcw = t.suggest_float("gb_min_child_weight", 1e-2, 30, log=True)
        if fam == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(learning_rate=lr, n_estimators=n, max_depth=md, reg_lambda=l2,
                                    subsample=sub, colsample_bytree=col, min_child_weight=mcw,
                                    tree_method="hist", device=device, random_state=seed,
                                    n_jobs=8, verbosity=0)
        import lightgbm as lgb
        return lgb.LGBMRegressor(learning_rate=lr, n_estimators=n, max_depth=md, reg_lambda=l2,
                                 subsample=sub, subsample_freq=1, colsample_bytree=col,
                                 min_child_weight=mcw, random_state=seed, n_jobs=8, verbose=-1,
                                 device="gpu" if cuda else "cpu")

    if fam == "mlp":
        return _Torch(_mlp_builder(t, cuda), t, seed, device, "mlp")
    if fam == "resnet":
        return _Torch(_resnet_builder(t), t, seed, device, "res", lr_max=5e-3)
    if fam == "ft_transformer":
        faz = _ft_builder(t)
        w = _Torch(faz, t, seed, device, "ft")
        base = w.faz_rede

        def faz_env(d_in, dr):
            """Wrap the FT-Transformer so that it takes one input tensor.

            Parameters
            ----------
            d_in : int
                Number of input features.
            dr : float
                Dropout.

            Returns
            -------
            net : torch.nn.Module
                The wrapped network.
            """
            import torch.nn as nn
            net = base(d_in, dr)

            class Env(nn.Module):
                """nn.Module wrapper that passes no categorical features."""
                def __init__(s2):
                    """Hold the wrapped network."""
                    super().__init__()
                    s2.net = net

                def forward(s2, x):
                    """Forward the continuous features only.

                    Parameters
                    ----------
                    x : torch.Tensor
                        Continuous features.

                    Returns
                    -------
                    y : torch.Tensor
                        Output of the wrapped network.
                    """
                    return s2.net(x, None)
            return Env()
        w.faz_rede = faz_env
        return w
    if fam == "tabnet":
        return _TabNet(t, seed, device)
    if fam == "tabpfn":
        return _TabPFN(t, seed, device)
    raise ValueError("familia desconhecida: %s" % fam)
