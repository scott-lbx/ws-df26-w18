# -*- coding: utf-8 -*-
"""slides_bldg.py —— 建築層降維裸面板(每棟建築=一個點,對齊實際 dataset)

lujiazui 3263 棟 × 16 特徵 → PCA / t-SNE / Isomap 2D,按高度著色。
裸面板(無標題/圖例),透明深底友善。存 out/figs/slides/p_bldg_*.png。
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nn

OUT = Path(__file__).resolve().parent / "out" / "figs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)
TICK = "#C9D1D9"; GRID = "#33404F"


def _bare(ax):
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xticks([]); ax.set_yticks([])


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig); print("[bldg]", name)


def main():
    X, feat, aux = nn.make_dataset("lujiazui")
    h = np.asarray(aux["height_m"], dtype=float)
    vmax = np.nanpercentile(h, 97)
    from sklearn.manifold import TSNE, Isomap
    Zp = nn.PCA(n_components=2).fit(X).transform(X)
    Zt = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    Zi = Isomap(n_components=2, n_neighbors=15).fit_transform(X)
    for nm, Z in [("pca", Zp), ("tsne", Zt), ("isomap", Zi)]:
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        ax.scatter(Z[:, 0], Z[:, 1], c=h, s=6, alpha=0.55, cmap="viridis", vmax=vmax, edgecolor="none")
        _bare(ax)
        _save(fig, f"p_bldg_{nm}.png")
    print("[DONE]", OUT)


if __name__ == "__main__":
    main()
