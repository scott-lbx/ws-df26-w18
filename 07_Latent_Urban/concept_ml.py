# -*- coding: utf-8 -*-
"""concept_ml.py —— ML/NN 原理概念圖(教科書式,深色房風,對齊簡報 BG=12253C)

參考網路上通用畫法後重畫:
  m_pca      主軸投影(最大方差方向 + 投影)
  m_tsne     高維纏繞 → 2D 分明簇(局部保留,簇間距離無意義)
  m_isomap   瑞士捲:測地線 vs 直線「穿牆」→ 攤平
  m_ae       自編碼器神經網路架構(編碼器→瓶頸→解碼器)
  m_vae      變分自編碼器(μ/σ → 採樣 z → 解碼;潛空間可採樣)
  m_cnn      Conv-VAE:高度影像 → 卷積特徵圖 → 潛碼 → 生成

存 out/figs/concept_ml/。執行: python concept_ml.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Ellipse, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn._font import set_cjk_font
set_cjk_font()

OUT = Path(__file__).resolve().parent / "out" / "figs" / "concept_ml"
OUT.mkdir(parents=True, exist_ok=True)

BG = "#12253C"
TX = "#F2F5FA"; MUT = "#8B949E"; GRID = "#33404F"
BLUE = "#5AB0FF"; GOLD = "#F2C14E"; TEAL = "#4EC9B0"; CORAL = "#F96167"; PUR = "#C586C0"


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.axis("off"); return fig, ax


def _arrow(ax, p1, p2, c=TX, lw=2.2, style="-|>", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, lw=lw, color=c))


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig); print("[concept-ml]", name)


def _layer(ax, x, ys, r=0.16, fc=BLUE, ec=None):
    pts = []
    for y in ys:
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec or fc, lw=1.2, zorder=3)); pts.append((x, y))
    return pts


def _connect(ax, L1, L2, c=GRID, lw=0.6):
    for a in L1:
        for b in L2:
            ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, zorder=1, alpha=0.5)


# -------------------------------------------------- PCA
def m_pca():
    fig, ax = _fig(7.6, 5.2)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 2)) @ [[2.4, 0], [0, 0.7]]
    ang = np.deg2rad(28); R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    X = X @ R.T
    ax.scatter(X[:, 0], X[:, 1], s=20, color=BLUE, alpha=0.55, zorder=2, edgecolor="none")
    v1 = np.array([np.cos(ang), np.sin(ang)]); v2 = np.array([-np.sin(ang), np.cos(ang)])
    _arrow(ax, (0, 0), tuple(v1 * 4.2), c=CORAL, lw=3.2)
    _arrow(ax, (0, 0), tuple(v2 * 1.7), c=GOLD, lw=3.2)
    # 投影示範:一個點投到 PC1
    p = X[np.argmax(X @ v1) - 3]
    proj = (p @ v1) * v1
    ax.scatter(*p, s=90, color=TX, zorder=5, edgecolor=BG, lw=1)
    ax.plot([p[0], proj[0]], [p[1], proj[1]], "--", color=MUT, lw=1.6, zorder=4)
    ax.scatter(*proj, s=60, color=CORAL, zorder=5)
    ax.text(v1[0]*4.4, v1[1]*4.4, "PC1\n最大方差方向", color=CORAL, fontsize=13, fontweight="bold", va="center")
    ax.text(v2[0]*1.9-0.3, v2[1]*1.9+0.2, "PC2", color=GOLD, fontsize=13, fontweight="bold")
    ax.text(p[0]+0.2, p[1]+0.1, "建築", color=TX, fontsize=11)
    ax.text(proj[0]+0.1, proj[1]-0.45, "投影(降維後座標)", color=MUT, fontsize=10)
    ax.set_xlim(-5.2, 6.2); ax.set_ylim(-3.5, 4.6); ax.set_aspect("equal")
    _save(fig, "m_pca.png")


# -------------------------------------------------- t-SNE
def m_tsne():
    fig, ax = _fig(9.2, 4.6)
    rng = np.random.default_rng(1)
    cols = [BLUE, GOLD, TEAL, CORAL]
    # 左:高維纏繞(重疊)
    for c, (cx, cy) in zip(cols, [(2.0, 2.6), (2.6, 2.9), (2.3, 2.2), (2.9, 2.5)]):
        p = rng.normal(size=(22, 2)) * 0.75 + [cx, cy]
        ax.scatter(p[:, 0], p[:, 1], s=26, color=c, alpha=0.65, edgecolor="none")
    ax.text(2.4, 4.4, "高維:纏繞、看不清", color=MUT, fontsize=13, ha="center")
    # 箭頭
    _arrow(ax, (4.4, 2.6), (5.6, 2.6), c=TEAL, lw=3)
    ax.text(5.0, 3.0, "t-SNE", color=TEAL, fontsize=13, fontweight="bold", ha="center")
    # 右:2D 分明簇
    for c, (cx, cy) in zip(cols, [(6.6, 1.5), (7.0, 3.8), (8.9, 3.6), (8.6, 1.4)]):
        p = rng.normal(size=(22, 2)) * 0.35 + [cx, cy]
        ax.scatter(p[:, 0], p[:, 1], s=30, color=c, alpha=0.85, edgecolor="none")
    ax.text(7.7, 4.4, "2D:局部簇分明", color=TX, fontsize=13, ha="center")
    ax.annotate("", xy=(8.2, 2.6), xytext=(7.4, 2.6), arrowprops=dict(arrowstyle="<->", color=MUT, lw=1.4))
    ax.text(7.8, 2.35, "簇間距離無意義", color=CORAL, fontsize=10.5, ha="center")
    ax.set_xlim(0.8, 10); ax.set_ylim(0.5, 4.8)
    _save(fig, "m_tsne.png")


# -------------------------------------------------- Isomap
def m_isomap():
    fig = plt.figure(figsize=(9.6, 4.3)); fig.patch.set_facecolor(BG)
    axL = fig.add_axes([0.02, 0.08, 0.44, 0.84]); axR = fig.add_axes([0.60, 0.34, 0.38, 0.30])
    for a in (axL, axR):
        a.set_facecolor(BG); a.axis("off")
    t = np.linspace(1.4, 4.2 * np.pi, 500)
    x, y = t * np.cos(t) * 0.22, t * np.sin(t) * 0.22
    axL.scatter(x, y, c=t, cmap="viridis", s=12, zorder=2)
    i, j = 70, 360
    # 測地線:沿紙面(粗白線疊在螺旋上,和資料色區隔)
    axL.plot(x[i:j], y[i:j], color=TX, lw=3.4, alpha=0.85, zorder=3, solid_capstyle="round")
    # 直線:穿牆(紅虛線)
    axL.plot([x[i], x[j]], [y[i], y[j]], color=CORAL, lw=2.4, ls="--", zorder=4)
    axL.scatter([x[i], x[j]], [y[i], y[j]], s=80, color=TX, zorder=5, edgecolor=BG, lw=1.5)
    axL.text(x[i]-1.2, y[i]+0.3, "直線\n穿牆(錯)", color=CORAL, fontsize=11, ha="center", fontweight="bold")
    axL.text(-0.3, -2.9, "測地線=沿紙面(對)", color=TX, fontsize=11, ha="center", fontweight="bold")
    axL.text(0, 3.3, "資料像捲起來的紙", color=MUT, fontsize=13, ha="center")
    axL.set_aspect("equal"); axL.set_xlim(-3.4, 3.4); axL.set_ylim(-3.4, 3.7)
    axR.scatter(t, t * 0, c=t, cmap="viridis", s=40)
    axR.set_xlim(t.min()-1, t.max()+1); axR.set_ylim(-1, 1)
    axR.text((t.min()+t.max())/2, 0.75, "Isomap 攤平 → 保全局結構", color=TX, fontsize=12, ha="center", fontweight="bold")
    # 中間箭頭
    axA = fig.add_axes([0.46, 0.40, 0.14, 0.2]); axA.axis("off"); axA.set_xlim(0, 1); axA.set_ylim(0, 1)
    _arrow(axA, (0.1, 0.5), (0.9, 0.5), c=TEAL, lw=3, ms=20)
    axA.text(0.5, 0.72, "攤平", color=TEAL, fontsize=12, ha="center", fontweight="bold")
    _save(fig, "m_isomap.png")


# -------------------------------------------------- AE
def m_ae():
    fig, ax = _fig(9.6, 5.0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    L = [
        (1.0, np.linspace(1.2, 4.8, 7), BLUE),
        (2.8, np.linspace(1.9, 4.1, 4), BLUE),
        (4.6, np.linspace(2.6, 3.4, 2), CORAL),   # 瓶頸
        (6.4, np.linspace(1.9, 4.1, 4), GOLD),
        (8.2, np.linspace(1.2, 4.8, 7), GOLD),
    ]
    pts = [_layer(ax, x, ys, fc=c) for x, ys, c in L]
    for a, b in zip(pts, pts[1:]):
        _connect(ax, a, b)
    ax.text(1.9, 5.4, "編碼器", color=BLUE, fontsize=14, fontweight="bold", ha="center")
    ax.text(4.6, 5.4, "潛碼(瓶頸)", color=CORAL, fontsize=14, fontweight="bold", ha="center")
    ax.text(7.3, 5.4, "解碼器", color=GOLD, fontsize=14, fontweight="bold", ha="center")
    ax.text(1.0, 0.6, "原城市\n(高維)", color=TX, fontsize=11, ha="center")
    ax.text(8.2, 0.6, "還原\n(高維)", color=TX, fontsize=11, ha="center")
    ax.text(4.6, 1.9, "2 個數字", color=CORAL, fontsize=11, ha="center")
    _save(fig, "m_ae.png")


# -------------------------------------------------- VAE
def m_vae():
    fig, ax = _fig(10.0, 5.0)
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 6)
    # 編碼器(梯形)
    ax.add_patch(Polygon([(0.6, 1.4), (2.2, 2.3), (2.2, 3.7), (0.6, 4.6)], fc=BLUE, alpha=0.9))
    ax.text(1.3, 3.0, "編碼器", color="#0C1A2E", fontsize=13, fontweight="bold", ha="center", va="center", rotation=90)
    # μ, σ
    for yy, lab, c in [(3.7, "μ 平均", GOLD), (2.3, "σ 標準差", TEAL)]:
        for k in range(3):
            ax.add_patch(Circle((2.9 + k * 0.34, yy), 0.13, fc=c))
        ax.text(3.9, yy, lab, color=c, fontsize=12, va="center")
    # 採樣 z
    ax.text(5.1, 4.5, "採樣 z = μ + σ·ε", color=TX, fontsize=12, ha="center")
    ax.text(5.1, 4.1, "ε ~ N(0, I)", color=MUT, fontsize=11, ha="center")
    for k in range(3):
        ax.add_patch(Circle((4.9 + k * 0.34, 3.0), 0.14, fc=CORAL))
    ax.text(5.25, 2.5, "z", color=CORAL, fontsize=13, fontweight="bold", ha="center")
    _arrow(ax, (4.2, 3.5), (4.75, 3.15), c=MUT, lw=1.6); _arrow(ax, (4.2, 2.4), (4.75, 2.85), c=MUT, lw=1.6)
    # 解碼器
    ax.add_patch(Polygon([(6.4, 2.3), (8.0, 1.4), (8.0, 4.6), (6.4, 3.7)], fc=GOLD, alpha=0.9))
    ax.text(7.2, 3.0, "解碼器", color="#0C1A2E", fontsize=13, fontweight="bold", ha="center", va="center", rotation=90)
    _arrow(ax, (5.7, 3.0), (6.35, 3.0), c=MUT, lw=1.8)
    ax.text(8.9, 3.0, "生成\n城市", color=TX, fontsize=12, ha="center", va="center")
    # 潛空間高斯雲(小圖)
    axc = fig.add_axes([0.72, 0.08, 0.26, 0.30]); axc.set_facecolor(BG); axc.axis("off")
    th = np.linspace(0, 2*np.pi, 80)
    for r, a in [(1.0, 0.12), (2.0, 0.07)]:
        axc.fill(r*np.cos(th), r*np.sin(th), color=TEAL, alpha=a)
    rng = np.random.default_rng(0); pp = rng.normal(size=(60, 2))
    axc.scatter(pp[:, 0], pp[:, 1], s=10, color=TEAL, alpha=0.6)
    axc.set_xlim(-3, 3); axc.set_ylim(-2.6, 2.6)
    axc.text(0, 2.4, "潛空間連續 → 處處可採樣", color=TX, fontsize=10.5, ha="center")
    ax.text(2.0, 5.4, "不是一個點,是一個分布", color=CORAL, fontsize=12.5, fontweight="bold")
    _save(fig, "m_vae.png")


# -------------------------------------------------- CNN
def m_cnn():
    fig, ax = _fig(10.2, 4.6)
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 5)
    rng = np.random.default_rng(2)
    # 輸入影像(格)
    img = (rng.random((8, 8)) > 0.72).astype(float)
    ax.imshow(img, extent=(0.3, 1.9, 1.6, 3.2), cmap="inferno", aspect="auto", zorder=2)
    ax.text(1.1, 1.2, "高度影像", color=TX, fontsize=11, ha="center")
    # 卷積特徵圖(縮小堆疊)
    def stack(cx, n, w, c):
        for k in range(n):
            ax.add_patch(Rectangle((cx + k*0.09, 2.0 + k*0.09), w, w, fc=c, ec="#0C1A2E", lw=0.6, alpha=0.9, zorder=3))
    stack(2.7, 4, 1.0, BLUE); stack(4.3, 6, 0.7, BLUE)
    ax.text(3.4, 1.2, "卷積特徵圖", color=BLUE, fontsize=11, ha="center")
    ax.text(3.4, 3.5, "讀紋理/疏密", color=MUT, fontsize=10, ha="center")
    # 潛碼
    for k in range(4):
        ax.add_patch(Circle((6.1, 1.9 + k*0.45), 0.14, fc=CORAL, zorder=3))
    ax.text(6.1, 1.1, "潛碼", color=CORAL, fontsize=11, ha="center")
    # 反卷積 → 生成影像
    stack(6.9, 6, 0.7, GOLD); stack(8.4, 4, 1.0, GOLD)
    img2 = (rng.random((8, 8)) > 0.7).astype(float)
    ax.text(8.9, 1.2, "生成新紋理", color=GOLD, fontsize=11, ha="center")
    for xa, xb in [(1.95, 2.65), (5.15, 5.9), (6.35, 6.85)]:
        _arrow(ax, (xa, 2.6), (xb, 2.6), c=MUT, lw=1.8)
    ax.text(5.2, 4.4, "卷積像看空照圖一樣讀「街廓排列」——urban form 最自然的 NN 表示", color=TX, fontsize=12, ha="center")
    _save(fig, "m_cnn.png")


# -------------------------------------------------- 城市 = 一個點(向量化)
def m_point():
    import matplotlib.cm as cm
    fig, ax = _fig(10.0, 4.6); ax.set_xlim(0, 10.4); ax.set_ylim(0, 5)
    feats = ["height_m 高度", "area_m2 面積", "slender 瘦高", "age 屋齡",
             "aoi_price 地價", "euluc 用途(商/住/公…)"]
    vals = [0.8, 0.45, 0.6, 0.3, 0.7, 0.5]
    ax.text(2.1, 4.6, "一棟建築", color=TX, fontsize=14, fontweight="bold", ha="center")
    for i, (f, v) in enumerate(zip(feats, vals)):
        y = 3.9 - i * 0.55
        ax.add_patch(Rectangle((1.4, y), 2.2, 0.34, fc=GRID))
        ax.add_patch(Rectangle((1.4, y), 2.2 * v, 0.34, fc=BLUE))
        ax.text(1.35, y + 0.17, f, color=TX, fontsize=10, ha="right", va="center")
    ax.text(2.5, 0.25, "十幾個建築特徵", color=MUT, fontsize=11, ha="center")
    _arrow(ax, (3.9, 2.3), (5.4, 2.3), c=GOLD, lw=3, ms=20)
    ax.text(4.65, 2.7, "打包成\n一個座標", color=GOLD, fontsize=12, fontweight="bold", ha="center")
    # 右:幾千棟建築 = 幾千個點,highlight 一棟
    rng = np.random.default_rng(4)
    cloud = rng.normal(size=(500, 2)) @ [[1.4, 0.4], [0, 1.0]] + [8.0, 3.0]
    hcol = cm.viridis(np.clip((cloud[:, 1] - 1.5) / 3, 0, 1))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=10, color=hcol, alpha=0.5, edgecolor="none")
    ax.scatter([8.0], [4.0], s=150, color=TX, edgecolor=CORAL, lw=2.5, zorder=5)
    ax.annotate("這棟建築 = 這一點", xy=(8.0, 4.0), xytext=(6.3, 4.6),
                color=CORAL, fontsize=11.5, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CORAL, lw=1.6))
    ax.text(8.2, 0.5, "一條街幾千棟建築 = 高維空間裡幾千個點\n位置相近 = 形態相似", color=MUT, fontsize=11, ha="center")
    _save(fig, "m_point.png")


# -------------------------------------------------- VAE 潛空間影像網格(canonical latent grid)
def m_latentgrid():
    import matplotlib.cm as cm
    G = 6
    fig, ax = _fig(6.4, 6.4); ax.set_xlim(-0.6, G); ax.set_ylim(-0.6, G)
    rng = np.random.default_rng(7)
    for r in range(G):       # 高度 bottom→top
        for c in range(G):   # 密度 left→right
            d = c / (G - 1); h = r / (G - 1)
            ax.add_patch(FancyBboxPatch((c + 0.06, r + 0.06), 0.88, 0.88,
                         boxstyle="round,pad=0.0,rounding_size=0.05", fc="#0C1A2E", ec=GRID, lw=0.6))
            nb = int(2 + d * 12)
            for _ in range(nb):
                bx = c + 0.15 + rng.random() * 0.62
                by = r + 0.12 + rng.random() * 0.5
                bh = 0.08 + h * 0.3 * (0.6 + rng.random() * 0.6)
                col = cm.inferno(0.25 + h * 0.7)
                ax.add_patch(Rectangle((bx, by), 0.09, bh, fc=col, ec="none"))
    _arrow(ax, (0, -0.35), (G - 0.2, -0.35), c=GOLD, lw=2.5)
    _arrow(ax, (-0.35, 0), (-0.35, G - 0.2), c=TEAL, lw=2.5)
    ax.text(G / 2, -0.75, "潛空間軸 1  →  密度增加", color=GOLD, fontsize=12, ha="center", fontweight="bold")
    ax.text(-0.7, G / 2, "潛空間軸 2  →  高度增加", color=TEAL, fontsize=12, va="center", rotation=90, fontweight="bold")
    _save(fig, "m_latentgrid.png")


def main():
    m_point(); m_pca(); m_tsne(); m_isomap(); m_ae(); m_vae(); m_cnn(); m_latentgrid()
    print("\n[DONE]", OUT)


if __name__ == "__main__":
    main()
