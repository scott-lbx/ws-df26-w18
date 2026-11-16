"""
vae.py(精簡註解版)— Beta-VAE:AE 再加「分布 + 重參數化 + KL」,於是能生成

和 AE 的差別只有三點(★):
  ★1 瓶頸吐 μ 和 logσ²(不是一個點,而是一團分布)
  ★2 重參數化抽樣 z = μ + σ·ε     (讓抽樣可微,模型才訓得動)
  ★3 loss 多一項 β·KL            (把 latent 逼成標準常態雲 → 可從雲裡抽樣生成)
把 USE_PRETRAINED 改 True 跳過訓練。自包含,`python vae.py`。

⚠️ 資料注意:本範例與 pretrained/vae.pt 用的是「示範資料」(v03/cache/cells.parquet),
   不是最完整/最正確的原始資料。正式資料(見 ../../06_INTERFACE_HANDOFF.md)到位後,
   務必把 DATA 指到正式資料、USE_PRETRAINED=False 重新訓練,數字與模型才算數。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 設定 =====
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "v03", "cache", "cells.parquet")
OUT = os.path.join(HERE, "..", "outputs"); os.makedirs(OUT, exist_ok=True)
PRE = os.path.join(HERE, "..", "pretrained")
# CHANNELS=None → 自動使用「所有數值欄」當 channel,不限制數量;要手動指定就填成 list
CHANNELS = None
NON_CHANNEL = {"window", "row", "col", "role", "name", "geometry", "bid", "bidx",
               "n_bld", "token", "token_v03", "token_v04", "judge", "semantic", "family"}
LATENT = 2; EPOCHS = 150; LR = 1e-3; BATCH = 256
BETA = 1.0                                          # ★ 重建 vs 整齊的旋鈕(試 0.1 / 5 看差別)
USE_PRETRAINED = False

# ===== 1~3. 讀表、丟空格、自動決定 channel、切分、標準化 =====
df = pd.read_parquet(DATA)
if "coverage" in df.columns:
    df = df[df["coverage"] >= 0.02].reset_index(drop=True)
FEATS = CHANNELS or [c for c in df.columns
                     if c not in NON_CHANNEL and pd.api.types.is_numeric_dtype(df[c])]
X = df[FEATS].to_numpy(np.float32); N = len(X); C = len(FEATS)
rng = np.random.default_rng(0); idx = rng.permutation(N); va = idx[:N // 5]; tr = idx[N // 5:]
mean = X[tr].mean(0); std = X[tr].std(0); std[std == 0] = 1
Xz = ((X - mean) / std).astype(np.float32)
np.savez(os.path.join(PRE, "mean_std.npz"), mean=mean, std=std)
print(f"N={N}  C={C}  beta={BETA}  channels={FEATS}")

# ===== 4. 模型 =====
class VAE(nn.Module):
    def __init__(s, c, L):
        super().__init__()
        s.body = nn.Sequential(nn.Linear(c, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        s.mu = nn.Linear(16, L)                     # ★1 中心
        s.logvar = nn.Linear(16, L)                 # ★1 散開度(log,保證正)
        s.dec = nn.Sequential(nn.Linear(L, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, c))
    def forward(s, x):
        h = s.body(x); mu = s.mu(h); logvar = s.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # ★2 重參數化
        return s.dec(z), mu, logvar

def loss_fn(xhat, x, mu, logvar, beta):
    recon = ((xhat - x) ** 2).sum(1).mean()                      # 重建項:對 channel 加總,再對 batch 平均
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)).mean()  # ★3 KL
    return recon + beta * kl, recon.item(), kl.item()

model = VAE(C, LATENT)

# ===== 5. 訓練(或載入預訓練)=====
if USE_PRETRAINED:
    ck = torch.load(os.path.join(PRE, "vae.pt"))
    assert ck["c_in"] == C, (f"預訓練模型是 {ck['c_in']} 個 channel,但這份資料有 {C} 個。"
                             f"channel 數變了就不能用舊模型 → 請 USE_PRETRAINED=False 重新訓練。")
    model.load_state_dict(ck["state_dict"])
    print("載入預訓練 vae.pt")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr = torch.tensor(Xz[tr]); Xva = torch.tensor(Xz[va])
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]]
            opt.zero_grad()
            xhat, mu, logvar = model(xb)
            loss, _, _ = loss_fn(xhat, xb, mu, logvar, BETA)
            loss.backward(); opt.step()
        if ep % 30 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                xh, mu, lv = model(Xva); _, r, k = loss_fn(xh, Xva, mu, lv, BETA)
            print(f"  epoch {ep:3d}  recon={r:.4f}  kl={k:.4f}")
    torch.save({"state_dict": model.state_dict(), "c_in": C, "latent": LATENT}, os.path.join(OUT, "vae.pt"))

# ===== 6. 重建誤差 + latent(eval 用 μ 做確定性重建,不抽樣)=====
model.eval()
with torch.no_grad():
    h = model.body(torch.tensor(Xz)); MU = model.mu(h)
    Xhat = model.dec(MU).numpy()
Z = MU.numpy()
print(f"重建誤差(val)={np.mean((Xz[va]-Xhat[va])**2):.4f}")

# ===== 7. ★ 生成:從 N(0,I) 抽 z → decode → 反標準化 → 全新格子 =====
with torch.no_grad():
    znew = torch.randn(8, LATENT)                   # 抽 8 個點
    gen = model.dec(znew).numpy() * std + mean      # 反標準化回原始尺度
np.savez(os.path.join(OUT, "vae_generated.npz"), gen=gen)
print("生成 8 筆新格子 → outputs/vae_generated.npz")

# ===== 8. 畫 latent =====
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5)); plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.4)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title(f"VAE latent (beta={BETA})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "vae_latent.png"), dpi=120)
    print("圖 → outputs/vae_latent.png")
except ImportError:
    print("(未裝 matplotlib,略過畫圖)")
