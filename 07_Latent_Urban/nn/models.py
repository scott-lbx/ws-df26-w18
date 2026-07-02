# -*- coding: utf-8 -*-
"""
models.py — 三種把資料壓到 2D 潛在空間的方法（繁體中文註解）

    PCA  : 線性、封閉解（numpy SVD），沒有訓練迴圈。
    AE   : 自編碼器，非線性但無機率結構。
    VAE  : 變分自編碼器，潛在空間有機率意義，可從潛在網格「生成」。

刻意保持精簡、可讀，方便助教逐行講解。
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from .. import config
except (ImportError, ValueError):
    import config


# ════════════════════════════════════════════════════════════
# PCA — 用 numpy SVD 手刻，不依賴 sklearn
# ════════════════════════════════════════════════════════════
class PCA:
    """
    主成分分析（線性降維）。

    介面：
        fit(X)                     配適，計算主成分方向。
        transform(X)   -> Z[N, k]  投影到前 k 個主成分。
        inverse_transform(Z) -> Xhat  由潛在空間重建回原空間。
        explained_variance_ratio_     每個主成分解釋的變異比例。
    """

    def __init__(self, n_components=None):
        self.n_components = n_components or config.LATENT_DIM
        self.mean_ = None
        self.components_ = None            # shape [k, D]
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # 經濟型 SVD：Xc = U S Vt，Vt 的每一列即一個主成分方向
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        k = self.n_components
        self.components_ = Vt[:k]                       # [k, D]

        var = (S ** 2) / (X.shape[0] - 1)               # 各成分變異
        total = var.sum()
        self.explained_variance_ratio_ = (var[:k] / total) if total > 0 else var[:k]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) @ self.components_.T       # [N, k]
        return Z.astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        Xhat = Z @ self.components_ + self.mean_        # [N, D]
        return Xhat.astype(np.float32)


# ════════════════════════════════════════════════════════════
# AE — 自編碼器
# ════════════════════════════════════════════════════════════
class AE(nn.Module):
    """
    自編碼器：D -> 16 -> LATENT_DIM -> 16 -> D，隱藏層用 ReLU。
    以 MSE 重建誤差訓練。

    介面：
        forward(x)     -> xhat
        encode(x)      -> z     （回傳潛在座標）
        reconstruct(x) -> xhat  （重建，回傳 numpy）
    """

    def __init__(self, input_dim, latent_dim=None, hidden=16):
        super().__init__()
        latent_dim = latent_dim or config.LATENT_DIM
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    @torch.no_grad()
    def encode(self, X):
        self.eval()
        x = _as_tensor(X)
        return self.encoder(x).cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, X):
        self.eval()
        x = _as_tensor(X)
        return self.forward(x).cpu().numpy()


# ════════════════════════════════════════════════════════════
# VAE — 變分自編碼器
# ════════════════════════════════════════════════════════════
class VAE(nn.Module):
    """
    變分自編碼器：編碼器輸出 mu 與 logvar，經 reparameterization 抽樣，
    再解碼回原空間。損失 = MSE 重建 + beta * KL。

    介面：
        forward(x)     -> (xhat, mu, logvar)
        encode(x)      -> mu    （潛在座標取平均值即可）
        reconstruct(x) -> xhat  （重建，回傳 numpy）
        decode(Z)      -> Xhat  （由潛在座標生成，供 latent-grid 使用）
    """

    def __init__(self, input_dim, latent_dim=None, hidden=16):
        super().__init__()
        self.latent_dim = latent_dim or config.LATENT_DIM
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden, self.latent_dim)
        self.fc_logvar = nn.Linear(hidden, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def reparameterize(self, mu, logvar):
        # z = mu + sigma * eps，sigma = exp(0.5 * logvar)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        xhat = self.decoder(z)
        return xhat, mu, logvar

    @torch.no_grad()
    def encode(self, X):
        self.eval()
        x = _as_tensor(X)
        h = self.enc(x)
        return self.fc_mu(h).cpu().numpy()   # 以 mu 當潛在座標

    @torch.no_grad()
    def reconstruct(self, X):
        self.eval()
        x = _as_tensor(X)
        xhat, _, _ = self.forward(x)
        return xhat.cpu().numpy()

    @torch.no_grad()
    def decode(self, Z):
        self.eval()
        z = _as_tensor(Z)
        return self.decoder(z).cpu().numpy()


# ════════════════════════════════════════════════════════════
# 訓練輔助函式
# ════════════════════════════════════════════════════════════
def _as_tensor(X):
    """把 numpy / tensor 統一轉成 float32 CPU tensor。"""
    if isinstance(X, torch.Tensor):
        return X.float()
    return torch.as_tensor(np.asarray(X, dtype=np.float32))


def train_ae(X, epochs=None, lr=1e-3, seed=None, hidden=16):
    """
    訓練 AE（全批次梯度下降，~3k 筆資料足夠）。

    回傳
    ----
    (model, losses) : (AE, list[float])
    """
    epochs = config.EPOCHS if epochs is None else epochs
    seed = config.SEED if seed is None else seed
    torch.manual_seed(seed)

    x = _as_tensor(X)
    model = AE(input_dim=x.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        xhat = model(x)
        loss = loss_fn(xhat, x)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return model, losses


def train_vae(X, epochs=None, lr=1e-3, seed=None, beta=1.0, hidden=16):
    """
    訓練 VAE（全批次）。beta 控制 KL 項權重（越小越像 AE）。

    回傳
    ----
    (model, losses) : (VAE, list[float])   losses 為每回合的總損失。
    """
    epochs = config.EPOCHS if epochs is None else epochs
    seed = config.SEED if seed is None else seed
    torch.manual_seed(seed)

    x = _as_tensor(X)
    model = VAE(input_dim=x.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        xhat, mu, logvar = model(x)
        recon = torch.mean((xhat - x) ** 2)                       # MSE 重建
        # KL(q(z|x) || N(0,I))，對 batch 取平均
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kl
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return model, losses
