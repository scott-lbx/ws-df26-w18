# -*- coding: utf-8 -*-
"""conv.py —— Conv-VAE:CNN 流派的潛空間方法(看空間紋理,和表格 VAE 對照)

輸入是城市高度 patch 影像 [B,1,64,64],卷積編碼 → 潛向量 → 反卷積解碼。
和前面「每棟樓一列」的表格 AE/VAE 不同:卷積吃**空間排列**(街廓紋理、疏密),
是 urban form 最自然的 NN 表示。教學重點:同資料換影像張量,CNN 能學形態紋理。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """64×64 單通道高度圖的變分自編碼器。潛維預設 8。"""

    def __init__(self, latent=8, ch=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = ch
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Conv2d(1, c1, 4, 2, 1), nn.ReLU(),    # 64→32
            nn.Conv2d(c1, c2, 4, 2, 1), nn.ReLU(),   # 32→16
            nn.Conv2d(c2, c3, 4, 2, 1), nn.ReLU(),   # 16→8
        )
        self.flat = c3 * 8 * 8
        self.fc_mu = nn.Linear(self.flat, latent)
        self.fc_lv = nn.Linear(self.flat, latent)
        self.fc_dec = nn.Linear(latent, self.flat)
        self._c3 = c3
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.ReLU(),   # 8→16
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.ReLU(),   # 16→32
            nn.ConvTranspose2d(c1, 1, 4, 2, 1), nn.Sigmoid(),  # 32→64, [0,1]
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def reparam(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(lv)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self._c3, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return self.decode(z), mu, lv


def train_conv_vae(model, X, epochs=80, lr=1e-3, beta=0.1, anneal=0.25,
                   batch=64, seed=0, augment=True, val=None, free_bits=0.0,
                   pos_weight=9.0, verbose=False):
    """訓練 Conv-VAE。X:[N,1,64,64] float32 ∈[0,1]。回傳 history。
    augment=水平/垂直翻轉增樣(小資料必要)。
    pos_weight=建築像素(非零)的重建權重:高度圖 ~78% 空白,不加權時解碼器
    只要全輸出 0 就贏,潛碼失效(posterior collapse);加權逼它學建築位置。
    free_bits=每維 KL 下限(nats),另一種防塌縮手段(此處預設關,靠 pos_weight)。"""
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    Vt = None if val is None else torch.tensor(val, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    warm = max(1, int(anneal * epochs))
    hist = {"total": [], "recon": [], "kl": [], "val": []}
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        b = beta * min(1.0, (ep + 1) / warm)
        tot = rec = kl_ = 0.0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i+batch]]
            if augment:
                if torch.rand(1, generator=g) < 0.5:
                    xb = torch.flip(xb, [3])
                if torch.rand(1, generator=g) < 0.5:
                    xb = torch.flip(xb, [2])
            xh, mu, lv = model(xb)
            w = 1.0 + pos_weight * (xb > 0).float()          # 建築像素加權
            recon = (w * (xh - xb) ** 2).mean()
            kl_dim = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp(), dim=0)  # [latent]
            kl = torch.clamp(kl_dim, min=free_bits).sum() if free_bits else \
                -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
            loss = recon + b * kl
            opt.zero_grad(); loss.backward(); opt.step()
            w = len(xb) / n
            tot += loss.item() * w; rec += recon.item() * w; kl_ += kl.item() * w
        hist["total"].append(tot); hist["recon"].append(rec); hist["kl"].append(kl_)
        if Vt is not None:
            model.eval()
            with torch.no_grad():
                xh, mu, lv = model(Vt)
                hist["val"].append(float(F.mse_loss(xh, Vt)))
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  ep{ep:3d} total={tot:.4f} recon={rec:.4f} kl={kl_:.4f}")
    return hist


@torch.no_grad()
def encode_mu(model, X):
    model.eval()
    mu, _ = model.encode(torch.tensor(np.asarray(X), dtype=torch.float32))
    return mu.numpy()


@torch.no_grad()
def reconstruct(model, X):
    model.eval()
    xh, _, _ = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    return xh.numpy()


@torch.no_grad()
def sample(model, n, seed=0):
    """從先驗 N(0,I) 採樣生成高度 patch。"""
    torch.manual_seed(seed)
    z = torch.randn(n, model.latent)
    return model.decode(z).numpy()
