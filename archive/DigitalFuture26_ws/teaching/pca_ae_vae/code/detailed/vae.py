"""
============================================================================
 vae.py(詳細註解版)— Beta-VAE:會「生成」全新城市格子的模型
============================================================================
給完全不懂 Python / 神經網路的同學。逐行解釋。

VAE 不是全新的東西,它就是「AE 再加三樣東西」:
  ★1 瓶頸不再吐一個點,而是吐一團「分布」(用 μ 中心 + logσ² 散開度 描述)
  ★2 重參數化:用 z = μ + σ·ε 來抽樣(這招讓「抽籤」也能訓練)
  ★3 損失多一項 KL:逼那團分布靠近「標準常態雲」,於是 latent 變得整齊、沒有洞
為什麼要這三樣?因為只有「整齊沒有洞」的 latent,才能隨便抽一個點就解出合理結果
——這就是「生成」。AE 做不到(它的 latent 有洞),VAE 做得到。

執行:  python vae.py
試試把 BETA 改成 0.1 或 5,看「重建清晰」和「生成合理」怎麼拉鋸。

⚠️ 資料注意:本範例與 pretrained/vae.pt,用的是 repo 內的「示範資料」
   (v03/cache/cells.parquet),那不是最完整、最正確的原始資料。
   等正式資料(由上游依 ../../06_INTERFACE_HANDOFF.md 整理)到位後,
   一定要把設定區的 DATA 改成正式資料、USE_PRETRAINED 設 False 重新訓練,
   這裡印出的數字、存下的模型、生成的格子才有效。
============================================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F      # F 裡有一些現成的數學函式


# ============================================================
# 設定區
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "v03", "cache", "cells.parquet")
OUT = os.path.join(HERE, "..", "outputs"); os.makedirs(OUT, exist_ok=True)
PRE = os.path.join(HERE, "..", "pretrained")

# 不限制上游給幾個 channel。CHANNELS = None 表示「自動使用所有數值欄」;
# 要手動指定就填成 list。NON_CHANNEL 是「不是 channel 的欄」(位置/名稱/幾何/舊標籤),會跳過。
CHANNELS = None
NON_CHANNEL = {"window", "row", "col", "role", "name", "geometry", "bid", "bidx",
               "n_bld", "token", "token_v03", "token_v04", "judge", "semantic", "family"}

LATENT = 2
EPOCHS = 150
LR = 1e-3
BATCH = 256
BETA = 1.0          # ★ 關鍵旋鈕:KL 這項的權重。大=latent 整齊但重建糊;小=重建清楚但難生成。
USE_PRETRAINED = False


# ============================================================
# 第 1~3 步:讀表、丟空格、分組、標準化(和 AE 完全一樣)
# ============================================================
df = pd.read_parquet(DATA)
if "coverage" in df.columns:            # 有 coverage 欄才丟空格(沒給也不會錯)
    df = df[df["coverage"] >= 0.02].reset_index(drop=True)
# 自動挑出要當 channel 的欄(所有數值欄,排除 NON_CHANNEL);不限制數量
FEATS = CHANNELS or [c for c in df.columns
                     if c not in NON_CHANNEL and pd.api.types.is_numeric_dtype(df[c])]
X = df[FEATS].to_numpy(np.float32)
N = len(X); C = len(FEATS)
print(f"自動選到的 channel({C} 個):{FEATS}")
rng = np.random.default_rng(0); idx = rng.permutation(N)
va = idx[:N // 5]; tr = idx[N // 5:]
mean = X[tr].mean(0); std = X[tr].std(0); std[std == 0] = 1
Xz = ((X - mean) / std).astype(np.float32)
os.makedirs(PRE, exist_ok=True)
np.savez(os.path.join(PRE, "mean_std.npz"), mean=mean, std=std)   # 生成時要用來還原數字
print(f"資料筆數 N={N}，每筆 C={C} 個數字，beta={BETA}")


# ============================================================
# 第 4 步:定義 VAE(對照 AE,只多了標 ★ 的地方)
# ============================================================
class VAE(nn.Module):
    def __init__(self, c, L):
        super().__init__()
        # 編碼器的「身體」:9 → 32 → 16(和 AE 一樣)
        self.body = nn.Sequential(
            nn.Linear(c, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        # ★1 不再用一層輸出 z,而是用「兩個頭」分別輸出 μ 和 logσ²。
        self.mu = nn.Linear(16, L)        # μ:這團分布的「中心」
        self.logvar = nn.Linear(16, L)    # logσ²:這團分布的「散開度」(取 log 是為了保證之後算出來是正的)

        # 解碼器:2 → 16 → 32 → 9(和 AE 一樣)
        self.dec = nn.Sequential(
            nn.Linear(L, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, c),
        )

    def forward(self, x):
        h = self.body(x)                  # 先過身體
        mu = self.mu(h)                   # 算出中心 μ
        logvar = self.logvar(h)           # 算出散開度 logσ²

        # ★2 重參數化(reparameterization):這是 VAE 最重要的一招。
        #   我們想從 N(μ, σ²) 這團分布裡「抽」一個 z。
        #   但「抽籤」是隨機的,沒辦法回傳梯度 → 模型學不到 μ、σ 該設多少。
        #   解法:先抽一個和模型無關的標準亂數 ε,再用公式 z = μ + σ·ε 把它「搬」到位。
        #   這樣 z 變成 μ、σ 的普通運算 + 一個外來常數 ε,梯度就能順利流回去。
        std = torch.exp(0.5 * logvar)               # 由 logσ² 算出標準差 σ(exp 保證為正)
        eps = torch.randn_like(mu)                  # 抽一個形狀和 μ 一樣的標準常態亂數 ε
        z = mu + std * eps                          # ← 重參數化公式

        xhat = self.dec(z)                # 用抽到的 z 解碼還原
        return xhat, mu, logvar           # 回傳:還原結果、中心、散開度


# ★3 損失函數:重建項 + β × KL 項
def loss_fn(xhat, x, mu, logvar, beta):
    # 重建項:還原得像不像。對 9 個 channel「加總」,再對整批「平均」。
    # (加總而不是平均,是為了讓重建項夠有份量,不會被 KL 壓垮 → 避免模型擺爛。)
    recon = ((xhat - x) ** 2).sum(dim=1).mean()

    # KL 項:衡量「這團分布」離「標準常態雲 N(0,1)」有多遠。這是固定公式,背起來即可。
    # 它在逼 μ→0、σ→1,把所有資料的 latent 收攏成一團整齊的雲。
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

    # 總損失:兩項相加,β 控制 KL 的份量。
    return recon + beta * kl, recon.item(), kl.item()


model = VAE(C, LATENT)


# ============================================================
# 第 5 步:訓練(或載入現成模型)
# ============================================================
if USE_PRETRAINED:
    ck = torch.load(os.path.join(PRE, "vae.pt"))
    assert ck["c_in"] == C, (f"預訓練模型是 {ck['c_in']} 個 channel,但現在資料有 {C} 個。"
                             f"channel 數變了 → 請把 USE_PRETRAINED 設 False 重新訓練。")
    model.load_state_dict(ck["state_dict"])
    print("已載入預訓練模型 vae.pt(跳過訓練)")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr = torch.tensor(Xz[tr]); Xva = torch.tensor(Xz[va])
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]]
            opt.zero_grad()                              # 1. 清梯度
            xhat, mu, logvar = model(xb)                 # 2. 前向(內含抽樣)
            loss, _, _ = loss_fn(xhat, xb, mu, logvar, BETA)   # 3. 算損失(重建 + KL)
            loss.backward()                              # 4a. 反向
            opt.step()                                   # 4b. 更新
        if ep % 30 == 0 or ep == 1:
            model.eval()
            with torch.no_grad():
                xh, mu, lv = model(Xva)
                _, r, k = loss_fn(xh, Xva, mu, lv, BETA)
            # recon 往下掉 = 越還原越像;kl 穩定在某個正值 = latent 有被好好使用(沒擺爛)
            print(f"  第 {ep:3d} 輪   重建={r:.4f}   KL={k:.4f}")
    torch.save({"state_dict": model.state_dict(), "c_in": C, "latent": LATENT},
               os.path.join(OUT, "vae.pt"))


# ============================================================
# 第 6 步:評估重建,並取出 latent(這裡用 μ 當座標,不抽樣,結果才穩定)
# ============================================================
model.eval()
with torch.no_grad():
    h = model.body(torch.tensor(Xz))
    MU = model.mu(h)                    # 每個格子 latent 的中心
    Xhat = model.dec(MU).numpy()        # 用中心解碼(確定性重建)
Z = MU.numpy()
print(f"重建誤差(驗證組)={np.mean((Xz[va] - Xhat[va]) ** 2):.4f}")
print("(通常比 AE 略大一點點 —— 這是用『一點點重建品質』換來『能生成』,很划算)")


# ============================================================
# 第 7 步:★ 生成 —— VAE 才有的本領
# ============================================================
# 因為 KL 已經把所有資料的 latent 收進「標準常態雲」,雲裡到處都合理,
# 所以我們直接從 N(0,1) 隨便抽幾個點,丟進解碼器,就會「長出」全新的格子。
with torch.no_grad():
    znew = torch.randn(8, LATENT)               # 從標準常態雲抽 8 個點 [8, 2]
    gen_std = model.dec(znew).numpy()           # 解碼成 8 筆(標準化尺度的)新格子
    gen = gen_std * std + mean                  # 反標準化:乘回 std、加回 mean,還原成真實尺度
np.savez(os.path.join(OUT, "vae_generated.npz"), gen=gen)
print("已生成 8 筆全新格子 → outputs/vae_generated.npz")
print("(這 8 筆是模型『無中生有』的,可交給生成/渲染線變成新城市)")


# ============================================================
# 第 8 步:畫 latent 散點圖
# ============================================================
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.4)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title(f"VAE latent (beta={BETA})")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "vae_latent.png"), dpi=120)
    print("已存圖 → outputs/vae_latent.png")
except ImportError:
    print("(沒裝 matplotlib,跳過畫圖。)")
