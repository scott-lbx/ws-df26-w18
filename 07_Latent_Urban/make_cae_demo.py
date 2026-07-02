# -*- coding: utf-8 -*-
"""make_cae_demo.py —— CNN 在本資料上「重建=模糊」的誠實證據圖

訓練一個 Conv-AE(已是無 KL、最有利重建的設定),對照:
  上排 = 真實高度 patch(輸入)
  下排 = CNN 重建(只抓密度/位置,細節=模糊)
說明:12 城 + 稀疏高頻 footprint → 小模型必然糊成密度圖。
存 out/figs/concept_ml/m_cae_recon.png。執行: python make_cae_demo.py
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn._font import set_cjk_font
from nn import raster
set_cjk_font()

OUT = Path(__file__).resolve().parent / "out" / "figs" / "concept_ml"
BG = "#12253C"; TX = "#F2F5FA"
SLUGS = ["lujiazui", "caoyang", "yuyuan", "waitan", "nanjingxi", "kongjiang",
         "pengpu", "laoximen", "dapuqiao", "xujiahui", "nanjingdong", "beiwaitan"]


class CAE(nn.Module):
    def __init__(s, lat=32):
        super().__init__()
        s.enc = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
                              nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                              nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())
        s.fm = 128 * 4 * 4
        s.fc1 = nn.Linear(s.fm, lat); s.fc2 = nn.Linear(lat, s.fm)
        s.dec = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
                              nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                              nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())

    def forward(s, x):
        z = s.fc1(s.enc(x).flatten(1))
        return s.dec(s.fc2(z).view(-1, 128, 4, 4))


def main():
    X, _ = raster.city_patch_dataset(SLUGS, patch=32, stride=8, min_fill=0.08)
    Xt = torch.tensor(X)
    m = CAE(32); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator().manual_seed(0); n = len(Xt)
    for ep in range(120):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 128):
            xb = Xt[perm[i:i+128]]
            if torch.rand(1, generator=g) < 0.5:
                xb = torch.flip(xb, [3])
            xh = m(xb); w = 1 + 12 * (xb > 0).float()
            (w * (xh - xb) ** 2).mean().backward(); opt.step(); opt.zero_grad()
    m.eval()
    with torch.no_grad():
        fill = (X > 0).reshape(len(X), -1).mean(1)
        pick = np.argsort(-fill)[:6]
        rec = m(torch.tensor(X[pick])).numpy()[:, 0]
    fig, ax = plt.subplots(2, 6, figsize=(9.6, 3.4)); fig.patch.set_facecolor(BG)
    for k in range(6):
        ax[0, k].imshow(X[pick[k], 0], cmap="inferno", vmin=0, vmax=1)
        ax[1, k].imshow(rec[k], cmap="inferno", vmin=0, vmax=1)
        for a in (ax[0, k], ax[1, k]):
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#33404F")
    ax[0, 0].set_ylabel("輸入\n真實高度", color=TX, fontsize=13, rotation=0, ha="right", va="center", labelpad=22)
    ax[1, 0].set_ylabel("CNN\n重建", color="#F96167", fontsize=13, rotation=0, ha="right", va="center", labelpad=22)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.savefig(OUT / "m_cae_recon.png", dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig); print("[cae] m_cae_recon.png")


if __name__ == "__main__":
    main()
