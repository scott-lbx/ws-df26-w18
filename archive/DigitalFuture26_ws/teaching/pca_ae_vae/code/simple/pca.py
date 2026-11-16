"""
pca.py(精簡註解版)— 用 PCA 把 9 維城市格子壓成 2 維

流程:讀表 → 丟空格 → 標準化 → SVD 求主軸 → 編碼/解碼 → 看重建誤差 → 畫圖
PCA 不用訓練,一次線性代數算完。自包含,直接 `python pca.py`。

⚠️ 資料注意:本範例與 pretrained/ 模型用的是「示範資料」(v03/cache/cells.parquet),
   不是最完整/最正確的原始資料。正式資料(見 ../../06_INTERFACE_HANDOFF.md)到位後,
   務必把 DATA 指到正式資料、重新跑一次,數字與模型才算數。
"""
import os
import numpy as np
import pandas as pd

# ===== 設定 =====
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "v03", "cache", "cells.parquet")
OUT = os.path.join(HERE, "..", "outputs"); os.makedirs(OUT, exist_ok=True)
# CHANNELS=None → 自動使用「所有數值欄」當 channel,不限制數量;要手動指定就填成 list
CHANNELS = None
NON_CHANNEL = {"window", "row", "col", "role", "name", "geometry", "bid", "bidx",
               "n_bld", "token", "token_v03", "token_v04", "judge", "semantic", "family"}
LATENT = 2

# ===== 1. 讀資料;若有 coverage 欄就丟空格;自動決定 channel 欄位 =====
df = pd.read_parquet(DATA)
if "coverage" in df.columns:
    df = df[df["coverage"] >= 0.02].reset_index(drop=True)
FEATS = CHANNELS or [c for c in df.columns
                     if c not in NON_CHANNEL and pd.api.types.is_numeric_dtype(df[c])]
X = df[FEATS].to_numpy(np.float32)                 # X: [N, C]
N, C = X.shape
print(f"N={N}  C={C}  channels={FEATS}")

# ===== 2. 切 train/val(固定 seed)=====
rng = np.random.default_rng(0)
idx = rng.permutation(N); va = idx[:N // 5]; tr = idx[N // 5:]

# ===== 3. 標準化:每個 channel 各自 (x-mean)/std,只用 train 算 =====
mean = X[tr].mean(0); std = X[tr].std(0); std[std == 0] = 1
Xz = (X - mean) / std                              # 之後反標準化用 mean/std

# ===== 4. PCA:對 train 做 SVD,取前 2 條主軸 W =====
Xc = Xz[tr] - Xz[tr].mean(0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
W = Vt[:LATENT].T                                  # W: [9, 2] 編碼器(就是個矩陣)
explained = (S[:LATENT] ** 2) / (S ** 2).sum()
print(f"W={W.shape}  各軸解釋變異={np.round(explained,3)}")

# ===== 5. 編碼 z=X·W,解碼 x̂=z·Wᵀ =====
Z = Xz @ W                                         # Z: [N, 2] 壓縮結果
Xhat = Z @ W.T                                     # Xhat: [N, 9] 重建

# ===== 6. 重建誤差(標準化尺度,越小越好)=====
err = lambda a, b: float(np.mean((a - b) ** 2))
print(f"重建誤差 train={err(Xz[tr],Xhat[tr]):.4f}  val={err(Xz[va],Xhat[va]):.4f}")

# ===== 7. 畫 latent 散點 =====
np.savez(os.path.join(OUT, "pca_result.npz"), W=W, Z=Z)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5)); plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.4)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA latent")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pca_latent.png"), dpi=120)
    print("圖 → outputs/pca_latent.png")
except ImportError:
    print("(未裝 matplotlib,略過畫圖)")
