"""
ae.py(精簡註解版)— Autoencoder:用神經網路把 9 維壓成 2 維

流程:讀表 → 標準化 → 建 Encoder/Decoder → 訓練(loss 下降)→ 重建誤差 → 畫 latent
把 USE_PRETRAINED 改成 True 可載入現成模型、跳過訓練。自包含,`python ae.py`。

⚠️ 資料注意:本範例與 pretrained/ae.pt 用的是「示範資料」(v03/cache/cells.parquet),
   不是最完整/最正確的原始資料。正式資料(見 ../../06_INTERFACE_HANDOFF.md)到位後,
   務必把 DATA 指到正式資料、USE_PRETRAINED=False 重新訓練,數字與模型才算數。
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ===== 設定 =====
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "..", "..", "v03", "cache", "cells.parquet")
OUT = os.path.join(HERE, "..", "outputs"); os.makedirs(OUT, exist_ok=True)
PRE = os.path.join(HERE, "..", "pretrained")
# CHANNELS=None → 自動使用「所有數值欄」當 channel,不限制數量;要手動指定就填成 list
CHANNELS = None
NON_CHANNEL = {"window", "row", "col", "role", "name", "geometry", "bid", "bidx",
               "n_bld", "token", "token_v03", "token_v04", "judge", "semantic", "family"}
LATENT = 2; EPOCHS = 100; LR = 1e-3; BATCH = 256
USE_PRETRAINED = False                              # 改 True 可跳過訓練

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
np.savez(os.path.join(PRE, "mean_std.npz"), mean=mean, std=std)  # 反標準化用
print(f"N={N}  C={C}  channels={FEATS}")

# ===== 4. 模型:Encoder 9→32→16→2(瓶頸,線性);Decoder 2→16→32→9 =====
class AE(nn.Module):
    def __init__(s, c, L):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(c, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, L))
        s.dec = nn.Sequential(nn.Linear(L, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, c))
    def forward(s, x):
        z = s.enc(x); return s.dec(z), z

model = AE(C, LATENT)

# ===== 5. 訓練(或載入預訓練)=====
if USE_PRETRAINED:
    ck = torch.load(os.path.join(PRE, "ae.pt"))
    assert ck["c_in"] == C, (f"預訓練模型是 {ck['c_in']} 個 channel,但這份資料有 {C} 個。"
                             f"channel 數變了就不能用舊模型 → 請 USE_PRETRAINED=False 重新訓練。")
    model.load_state_dict(ck["state_dict"])
    print("載入預訓練 ae.pt")
else:
    opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.MSELoss()
    Xtr = torch.tensor(Xz[tr]); Xva = torch.tensor(Xz[va])
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            xb = Xtr[perm[i:i + BATCH]]
            opt.zero_grad()                         # 清梯度
            xhat, _ = model(xb)                     # 前向
            loss = lossf(xhat, xb)                  # MSE
            loss.backward(); opt.step()             # 反向 + 更新
        if ep % 20 == 0 or ep == 1:
            model.eval()
            with torch.no_grad(): v = lossf(model(Xva)[0], Xva).item()
            print(f"  epoch {ep:3d}  val_mse={v:.4f}")
    torch.save({"state_dict": model.state_dict(), "c_in": C, "latent": LATENT}, os.path.join(OUT, "ae.pt"))

# ===== 6. 重建誤差 + 取 latent =====
model.eval()
with torch.no_grad(): Xhat, Z = model(torch.tensor(Xz))
Xhat, Z = Xhat.numpy(), Z.numpy()
err = lambda a, b: float(np.mean((a - b) ** 2))
print(f"重建誤差 train={err(Xz[tr],Xhat[tr]):.4f}  val={err(Xz[va],Xhat[va]):.4f}")

# ===== 7. 畫 latent =====
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5)); plt.scatter(Z[:, 0], Z[:, 1], s=4, alpha=.4)
    plt.xlabel("z1"); plt.ylabel("z2"); plt.title("AE latent")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ae_latent.png"), dpi=120)
    print("圖 → outputs/ae_latent.png")
except ImportError:
    print("(未裝 matplotlib,略過畫圖)")
