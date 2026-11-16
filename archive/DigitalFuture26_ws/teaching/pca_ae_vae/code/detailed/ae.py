"""
============================================================================
 ae.py(詳細註解版)— Autoencoder:用「神經網路」把 9 個數字壓成 2 個
============================================================================
給完全沒寫過 Python / 沒碰過神經網路的同學。逐行解釋。

和 PCA 的差別:PCA 用一個固定矩陣壓;AE 用一個「會自己學」的神經網路壓,
所以它能描述彎曲的關係,通常壓得比 PCA 好。代價是:要「訓練」(反覆調整)。

執行:  python ae.py
想跳過訓練、直接用現成模型:把下面的 USE_PRETRAINED 改成 True。

⚠️ 資料注意:本範例與 pretrained/ae.pt,用的是 repo 內的「示範資料」
   (v03/cache/cells.parquet),那不是最完整、最正確的原始資料。
   等正式資料(由上游依 ../../06_INTERFACE_HANDOFF.md 整理)到位後,
   一定要把設定區的 DATA 改成正式資料、USE_PRETRAINED 設 False 重新訓練,
   這裡印出的數字與存下的模型才有效。
============================================================================
"""

import os
import numpy as np
import pandas as pd

# torch 是 PyTorch,做神經網路的工具箱。
import torch
import torch.nn as nn        # nn 裡面有現成的「神經網路層」可以堆


# ============================================================
# 設定區
# ============================================================
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "v03", "cache", "cells.parquet")
OUT = os.path.join(HERE, "..", "outputs"); os.makedirs(OUT, exist_ok=True)
PRE = os.path.join(HERE, "..", "pretrained")        # 預訓練模型放這裡

# 不限制上游給幾個 channel。CHANNELS = None 表示「自動使用所有數值欄」;
# 要手動指定就填成 list。NON_CHANNEL 是「不是 channel 的欄」(位置/名稱/幾何/舊標籤),會跳過。
CHANNELS = None
NON_CHANNEL = {"window", "row", "col", "role", "name", "geometry", "bid", "bidx",
               "n_bld", "token", "token_v03", "token_v04", "judge", "semantic", "family"}

LATENT = 2          # 壓成幾維
EPOCHS = 100        # 訓練輪數:把全部資料看過幾遍
LR = 1e-3           # 學習率:每次調整權重「跨多大步」(太大會壞,太小很慢)
BATCH = 256         # 一次餵幾筆資料給模型
USE_PRETRAINED = False    # True = 載入現成模型、跳過訓練(現場來不及訓練時用)


# ============================================================
# 第 1~3 步:讀表、丟空格、分組、標準化(和 PCA 完全一樣)
# ============================================================
df = pd.read_parquet(DATA)
if "coverage" in df.columns:            # 有 coverage 欄才丟空格(沒給也不會錯)
    df = df[df["coverage"] >= 0.02].reset_index(drop=True)
# 自動挑出要當 channel 的欄(所有數值欄,排除 NON_CHANNEL);不限制數量
FEATS = CHANNELS or [c for c in df.columns
                     if c not in NON_CHANNEL and pd.api.types.is_numeric_dtype(df[c])]
X = df[FEATS].to_numpy(np.float32)
N = len(X)                              # 格子數
C = len(FEATS)                          # 每格幾個數字(channel),由資料自動決定
print(f"自動選到的 channel({C} 個):{FEATS}")

rng = np.random.default_rng(0)
idx = rng.permutation(N)
va = idx[:N // 5]                       # 驗證組
tr = idx[N // 5:]                       # 訓練組

mean = X[tr].mean(0); std = X[tr].std(0); std[std == 0] = 1
Xz = ((X - mean) / std).astype(np.float32)      # 標準化後的資料 [N, 9]

# 把標準化用的 mean / std 存起來(之後 VAE 生成、還原數字時要用同一組)。
os.makedirs(PRE, exist_ok=True)
np.savez(os.path.join(PRE, "mean_std.npz"), mean=mean, std=std)
print(f"資料筆數 N={N}，每筆 C={C} 個數字")


# ============================================================
# 第 4 步:定義神經網路(這就是 AE 的「形狀」)
# ============================================================
# 一個「class」是一個自訂的東西。這裡定義一個叫 AE 的神經網路。
# nn.Linear(a, b) 是一層「全連接層」:輸入 a 個數字、輸出 b 個數字。
# nn.ReLU() 是「激活函數」:把負數變 0,讓網路能描述彎曲(非線性)的關係。

class AE(nn.Module):
    def __init__(self, c, L):           # __init__ 在建立模型時執行,負責「搭骨架」
        super().__init__()              # 這行是固定寫法,呼叫父類別的初始化

        # 編碼器:9 → 32 → 16 → 2,一層層把資料壓小。
        # 注意最後一層後面「沒有 ReLU」,因為壓縮後的 2 個數字要能是負的。
        self.enc = nn.Sequential(
            nn.Linear(c, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, L),           # ← 瓶頸:只剩 L(=2)個數字
        )
        # 解碼器:2 → 16 → 32 → 9,把 2 個數字還原回 9 個。
        self.dec = nn.Sequential(
            nn.Linear(L, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, c),
        )

    def forward(self, x):               # forward 定義「資料怎麼流過模型」
        z = self.enc(x)                 # 先編碼成 z(壓縮)
        xhat = self.dec(z)              # 再解碼成 xhat(還原)
        return xhat, z                  # 回傳「還原結果」和「壓縮結果」


model = AE(C, LATENT)                    # 真的把模型建出來


# ============================================================
# 第 5 步:訓練(或直接載入現成模型)
# ============================================================
if USE_PRETRAINED:
    # torch.load 把存好的模型參數讀回來;load_state_dict 把參數裝回模型。
    ck = torch.load(os.path.join(PRE, "ae.pt"))
    # 安全檢查:預訓練模型的 channel 數要和現在資料一致,否則裝不回去。
    assert ck["c_in"] == C, (f"預訓練模型是 {ck['c_in']} 個 channel,但現在資料有 {C} 個。"
                             f"channel 數變了 → 請把 USE_PRETRAINED 設 False 重新訓練。")
    model.load_state_dict(ck["state_dict"])
    print("已載入預訓練模型 ae.pt(跳過訓練)")
else:
    # optimizer(優化器)負責「根據誤差去調整模型權重」。Adam 是常用的一種。
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()                # 損失函數:用「均方誤差」衡量還原得像不像

    # 把 numpy 資料轉成 torch 的 tensor(神經網路只吃 tensor)。
    Xtr = torch.tensor(Xz[tr])
    Xva = torch.tensor(Xz[va])

    # 訓練主迴圈:把資料看 EPOCHS 遍。
    for ep in range(1, EPOCHS + 1):
        model.train()                                   # 切到「訓練模式」
        perm = torch.randperm(len(Xtr))                 # 每輪把訓練資料重新洗牌
        # 一次拿 BATCH 筆出來訓練。
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]]                 # 這個小批次的資料

            # ↓↓↓ 這四行是「所有神經網路訓練的心臟」,背起來 ↓↓↓
            opt.zero_grad()                             # 1. 把上一輪的梯度清乾淨
            xhat, _ = model(xb)                         # 2. 前向:算出還原結果
            loss = lossf(xhat, xb)                      # 3. 算誤差(還原差多少)
            loss.backward()                             # 4a. 反向:算每個權重該往哪調
            opt.step()                                  # 4b. 真的調整權重一小步
            # ↑↑↑ 重複幾千次,loss 慢慢變小 = 模型在學 ↑↑↑

        # 每 20 輪印一次「驗證組」的誤差,看有沒有進步。
        if ep % 20 == 0 or ep == 1:
            model.eval()                                # 切到「評估模式」
            with torch.no_grad():                       # 評估時不需要算梯度,省資源
                v = lossf(model(Xva)[0], Xva).item()
            print(f"  第 {ep:3d} 輪   驗證誤差={v:.4f}")

    # 訓練完,把模型存起來。
    torch.save({"state_dict": model.state_dict(), "c_in": C, "latent": LATENT},
               os.path.join(OUT, "ae.pt"))


# ============================================================
# 第 6 步:用訓練好的模型,算重建誤差、取出壓縮結果
# ============================================================
model.eval()
with torch.no_grad():
    Xhat, Z = model(torch.tensor(Xz))   # 把全部資料丟進去
Xhat = Xhat.numpy()                     # 轉回 numpy 方便後面用
Z = Z.numpy()

def err(a, b):
    return float(np.mean((a - b) ** 2))

print(f"重建誤差  訓練組={err(Xz[tr], Xhat[tr]):.4f}   驗證組={err(Xz[va], Xhat[va]):.4f}")
print("(應該比 PCA 小 —— 這就是『會彎的神經網路』比『直線 PCA』強的地方)")


# ============================================================
# 第 7 步:畫壓縮結果散點圖
# ============================================================
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.4)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title("AE latent")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ae_latent.png"), dpi=120)
    print("已存圖 → outputs/ae_latent.png")
except ImportError:
    print("(沒裝 matplotlib,跳過畫圖。)")
