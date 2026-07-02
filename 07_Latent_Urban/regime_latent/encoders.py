#!/usr/bin/env python
"""latent.py —— 城市形态的潜空间编码器(PCA / AE / VAE)

把 measure.diagnose 的 9 维形态特征压到低维潜空间,三种方法对照:
  · PCA  线性、可解释(主轴 = 特征线性组合),重建 = 正交投影
  · AE   非线性自编码,瓶颈层 = 潜码;可压到更低重建损失
  · VAE  变分自编码,潜空间连续可采样(β-VAE 调 KL 正则)

诚实边界:样本小(教学语料 60 = 12 城 × 5 情形),AE/VAE 仅作演示,
非统计推断或真实城市预测。city-grouped 切分避免同城泄漏。
"""
from __future__ import annotations
import numpy as np

FEATURES = ["far", "coverage", "h_mean", "h_max", "h_cv",
            "grain", "slender", "concentration", "n"]

# n / grain 跨城量级差大 → log1p 稳定;标准化前先做
LOG_FEATURES = {"n", "grain", "h_max"}


# ============================================================ 标准化
class Standardizer:
    """log1p(选定欄) + z-score。fit/transform/inverse 都可逆,供重建回真实尺度。"""

    def __init__(self, features=FEATURES, log_features=LOG_FEATURES):
        self.features = list(features)
        self.log_idx = [i for i, f in enumerate(self.features) if f in log_features]
        self.mu = None
        self.sd = None

    def _log(self, X):
        X = X.astype(float).copy()
        for j in self.log_idx:
            X[:, j] = np.log1p(X[:, j])
        return X

    def _unlog(self, X):
        X = X.copy()
        for j in self.log_idx:
            X[:, j] = np.expm1(X[:, j])
        return X

    def fit(self, X):
        Xl = self._log(np.asarray(X))
        self.mu = Xl.mean(0)
        self.sd = Xl.std(0) + 1e-8
        return self

    def transform(self, X):
        return (self._log(np.asarray(X)) - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse(self, Z):
        return self._unlog(np.asarray(Z) * self.sd + self.mu)


# ============================================================ PCA(numpy SVD)
class PCAModel:
    """SVD-based PCA,无外部依赖。whiten 可选。"""

    def __init__(self, k=2, whiten=False):
        self.k = k
        self.whiten = whiten

    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.k]                 # (k, d)
        self.singular_ = S
        var = (S ** 2) / (len(X) - 1)
        self.explained_ = var[:self.k]
        self.explained_ratio_ = var / var.sum()
        return self

    def transform(self, X):
        Xc = np.asarray(X, float) - self.mean_
        Z = Xc @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_ + 1e-8)
        return Z

    def inverse(self, Z):
        Z = np.asarray(Z, float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_ + 1e-8)
        return Z @ self.components_ + self.mean_

    def reconstruct(self, X):
        return self.inverse(self.transform(X))


# ============================================================ AE / VAE(torch)
def _torch():
    import torch
    return torch


class _MLP:
    """共用:建 encoder/decoder。惰性 import torch。"""
    pass


def build_ae(d_in, latent=2, hidden=(16, 8), bn=True, seed=0):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)

    def block(i, o, act=True):
        layers = [nn.Linear(i, o)]
        if bn:
            layers.append(nn.BatchNorm1d(o))
        if act:
            layers.append(nn.ReLU())
        return layers

    enc, prev = [], d_in
    for h in hidden:
        enc += block(prev, h); prev = h
    enc += [nn.Linear(prev, latent)]
    dec, prev = [], latent
    for h in reversed(hidden):
        dec += block(prev, h); prev = h
    dec += [nn.Linear(prev, d_in)]

    class AE(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = nn.Sequential(*enc)
            s.dec = nn.Sequential(*dec)

        def encode(s, x):
            return s.enc(x)

        def forward(s, x):
            z = s.enc(x)
            return s.dec(z), z

    return AE()


def build_vae(d_in, latent=2, hidden=(16, 8), seed=0):
    torch = _torch()
    import torch.nn as nn
    torch.manual_seed(seed)

    class VAE(nn.Module):
        def __init__(s):
            super().__init__()
            enc, prev = [], d_in
            for h in hidden:
                enc += [nn.Linear(prev, h), nn.ReLU()]; prev = h
            s.enc = nn.Sequential(*enc)
            s.mu = nn.Linear(prev, latent)
            s.logvar = nn.Linear(prev, latent)
            dec, prev = [], latent
            for h in reversed(hidden):
                dec += [nn.Linear(prev, h), nn.ReLU()]; prev = h
            dec += [nn.Linear(prev, d_in)]
            s.dec = nn.Sequential(*dec)

        def encode(s, x):
            h = s.enc(x)
            return s.mu(h), s.logvar(h)

        def reparam(s, mu, logvar):
            std = _torch().exp(0.5 * logvar)
            return mu + std * _torch().randn_like(std)

        def forward(s, x):
            mu, logvar = s.encode(x)
            z = s.reparam(mu, logvar)
            return s.dec(z), mu, logvar

    return VAE()


def train_ae(model, Xtr, Xval=None, epochs=400, lr=3e-3, weight_decay=1e-4,
             denoise=0.0, seed=0, huber=True):
    """回传 history: {'train':[...], 'val':[...]}。denoise=输入噪声 std。"""
    torch = _torch()
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xv = None if Xval is None else torch.tensor(Xval, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = Fn.smooth_l1_loss if huber else Fn.mse_loss
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        xin = Xtr + denoise * torch.randn_like(Xtr) if denoise else Xtr
        xhat, _ = model(xin)
        loss = loss_fn(xhat, Xtr)
        opt.zero_grad(); loss.backward(); opt.step()
        hist["train"].append(float(loss))
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                xh, _ = model(Xv)
                hist["val"].append(float(loss_fn(xh, Xv)))
    return hist


def train_vae(model, Xtr, Xval=None, epochs=600, lr=3e-3, beta=1.0,
              anneal=0.0, seed=0):
    """β-VAE + KL annealing。anneal=达到满 β 的 epoch 比例(0=不退火)。
    回传 history: total/recon/kl(train)+ val_total。"""
    torch = _torch()
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    Xv = None if Xval is None else torch.tensor(Xval, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"total": [], "recon": [], "kl": [], "val": []}
    warm = max(1, int(anneal * epochs)) if anneal else 1
    for ep in range(epochs):
        model.train()
        b = beta * min(1.0, (ep + 1) / warm)
        xhat, mu, logvar = model(Xtr)
        recon = Fn.mse_loss(xhat, Xtr, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + b * kl
        opt.zero_grad(); loss.backward(); opt.step()
        hist["total"].append(float(loss)); hist["recon"].append(float(recon))
        hist["kl"].append(float(kl))
        if Xv is not None:
            model.eval()
            with torch.no_grad():
                xh, muv, lvv = model(Xv)
                hist["val"].append(float(Fn.mse_loss(xh, Xv)))
    return hist


def ae_reconstruct(model, X):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xh, _ = model(torch.tensor(X, dtype=torch.float32))
    return xh.numpy()


def vae_reconstruct(model, X, sample=False):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32)
        mu, logvar = model.encode(xt)
        z = model.reparam(mu, logvar) if sample else mu
        xh = model.dec(z)
    return xh.numpy()


def ae_encode(model, X):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.tensor(X, dtype=torch.float32))
    return z.numpy()


def vae_encode(model, X):
    torch = _torch()
    model.eval()
    with torch.no_grad():
        mu, _ = model.encode(torch.tensor(X, dtype=torch.float32))
    return mu.numpy()


def ae_decode(model, Z):
    """由任意潜码 Z 解码回(标准化)特征空间。供采样/生成/插值用。"""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xh = model.dec(torch.tensor(np.asarray(Z), dtype=torch.float32))
    return xh.numpy()


def vae_decode(model, Z):
    """VAE 解码器:z → (标准化)特征。z 可来自先验 N(0,I)。"""
    torch = _torch()
    model.eval()
    with torch.no_grad():
        xh = model.dec(torch.tensor(np.asarray(Z), dtype=torch.float32))
    return xh.numpy()


# ============================================================ 指标
def recon_mse(X, Xhat, per_feature=False):
    """标准化空间的 MSE。per_feature=True 回传每维。"""
    err = (np.asarray(X) - np.asarray(Xhat)) ** 2
    return err.mean(0) if per_feature else float(err.mean())
