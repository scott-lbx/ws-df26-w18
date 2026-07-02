# -*- coding: utf-8 -*-
"""extra_figs.py —— 論點框架 + GNN 概念 + 深度學習 urban 案例研究大圖(重畫)

深色房風(BG=12253C),供簡報「一頁一張大圖」用。案例為重畫的**概念示意** + 出處引用,非複製原論文圖。
存 out/figs/extra/。執行: python extra_figs.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn._font import set_cjk_font
set_cjk_font()

OUT = Path(__file__).resolve().parent / "out" / "figs" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#12253C"; TX = "#F2F5FA"; MUT = "#8B949E"; GRID = "#33404F"
BLUE = "#5AB0FF"; GOLD = "#F2C14E"; TEAL = "#4EC9B0"; CORAL = "#F96167"; PUR = "#C586C0"


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    return fig, ax


def _arrow(ax, p1, p2, c=TX, lw=2.2, style="-|>", ms=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, lw=lw, color=c))


def _save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", transparent=True); plt.close(fig); print("[extra]", name)


def _box(ax, x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec="none"))


# ---------------------------------------------------- 論點框架
def m_thesis():
    fig, ax = _fig(12.0, 4.4); ax.set_xlim(0, 12); ax.set_ylim(0, 4.4)
    cols = [("①", "編碼", "Encode", "把城市壓成資訊層\n(潛空間座標)", BLUE),
            ("②", "權力區分", "Differentiate", "不同權力體制\n在潛空間分開", GOLD),
            ("③", "重新生成", "Re-generate", "從潛空間採樣\n生成新的城市", TEAL)]
    for i, (n, zh, en, desc, c) in enumerate(cols):
        x = 0.4 + i * 3.95
        _box(ax, x, 0.5, 3.6, 3.4, "#1C3252")
        ax.add_patch(Circle((x + 0.55, 3.35), 0.32, fc=c))
        ax.text(x + 0.55, 3.35, n, color="#0C1A2E", fontsize=18, fontweight="bold", ha="center", va="center")
        ax.text(x + 1.1, 3.45, zh, color=c, fontsize=19, fontweight="bold", va="center")
        ax.text(x + 1.12, 2.95, en, color=MUT, fontsize=12, va="center")
        ax.text(x + 0.35, 2.2, desc, color=TX, fontsize=14, va="center")
        # mini icon
        if i == 0:
            for k in range(5): ax.add_patch(Rectangle((x+0.5+k*0.12, 0.8), 0.08, 0.5-abs(k-2)*0.12, fc=BLUE))
            _arrow(ax, (x+1.4, 1.0), (x+2.1, 1.0), c=BLUE, lw=2); ax.add_patch(Circle((x+2.5, 1.0), 0.12, fc=CORAL))
        if i == 1:
            for c2, dx in [(BLUE, 0.6), (GOLD, 1.6), (TEAL, 2.6)]:
                for _ in range(6):
                    ax.add_patch(Circle((x+dx+np.random.default_rng(int(dx*10)).uniform(-0.2,0.2), 1.0+np.random.default_rng(int(dx*7)+_).uniform(-0.2,0.2)), 0.07, fc=c2))
        if i == 2:
            ax.add_patch(Circle((x+0.7, 1.0), 0.14, fc=TEAL))
            _arrow(ax, (x+0.95, 1.0), (x+1.7, 1.0), c=TEAL, lw=2)
            for k in range(5): ax.add_patch(Rectangle((x+2.0+k*0.13, 0.8), 0.09, 0.3+0.15*(k%3), fc=GOLD))
        if i < 2:
            _arrow(ax, (x + 3.65, 2.0), (x + 4.0, 2.0), c=MUT, lw=2.5, ms=18)
    _save(fig, "m_thesis.png")


# ---------------------------------------------------- 城市是一張圖(graph)
def m_graph():
    fig, ax = _fig(11.0, 4.6); ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    # 左:表格(每棟獨立)
    ax.text(1.9, 4.6, "表格:每棟樓獨立一列", color=MUT, fontsize=13, ha="center")
    for i in range(5):
        y = 3.6 - i * 0.5
        ax.add_patch(Rectangle((0.7, y), 2.4, 0.36, fc="#1C3252", ec=GRID))
        for k in range(4):
            ax.add_patch(Rectangle((0.85 + k * 0.55, y + 0.07), 0.4, 0.22, fc=BLUE, alpha=0.4 + 0.15 * ((i + k) % 3)))
    ax.text(1.9, 0.6, "看不到「誰挨著誰」", color=CORAL, fontsize=12, ha="center")
    _arrow(ax, (3.4, 2.5), (4.3, 2.5), c=GOLD, lw=3, ms=20); ax.text(3.85, 2.85, "改成圖", color=GOLD, fontsize=12, fontweight="bold", ha="center")
    # 右:圖(節點=建築,邊=鄰接/街道)
    rng = np.random.default_rng(3)
    pos = rng.uniform([5.0, 1.0], [10.2, 4.2], size=(16, 2))
    # edges by proximity
    for i in range(len(pos)):
        d = np.linalg.norm(pos - pos[i], axis=1)
        for j in np.argsort(d)[1:3]:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color=TEAL, lw=1.1, alpha=0.5, zorder=1)
    ccol = [BLUE, GOLD, TEAL, CORAL]
    for i, p in enumerate(pos):
        ax.add_patch(Circle(p, 0.16, fc=ccol[i % 4], ec="#0C1A2E", lw=0.8, zorder=3))
    ax.text(7.6, 4.6, "圖 Graph:節點=建築 · 邊=鄰接/街道連通", color=TX, fontsize=13, ha="center")
    ax.text(7.6, 0.55, "GNN 讀的是「關係」——urban 尺度的自然結構", color=TEAL, fontsize=12.5, ha="center", fontweight="bold")
    _save(fig, "m_graph.png")


# ---------------------------------------------------- GNN 訊息傳遞
def m_gnn_mp():
    fig, ax = _fig(11.0, 4.4); ax.set_xlim(0, 11); ax.set_ylim(0, 4.4)
    center = np.array([3.0, 2.2])
    neigh = center + np.array([[-1.6, 1.1], [-1.8, -0.9], [0.2, 1.9], [-0.1, -1.8], [1.7, 0.9]])
    for n in neigh:
        ax.plot([center[0], n[0]], [center[1], n[1]], color=GRID, lw=1.4, zorder=1)
        ax.add_patch(Circle(n, 0.24, fc=BLUE, ec="#0C1A2E", lw=1, zorder=3))
        _arrow(ax, tuple(n + (center - n) * 0.28), tuple(center + (n - center) * 0.32), c=GOLD, lw=2, ms=13)
    ax.add_patch(Circle(center, 0.34, fc=CORAL, ec="white", lw=1.5, zorder=5))
    ax.text(center[0], center[1], "我", color="white", fontsize=15, fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(3.0, 4.15, "① 聚合鄰居的資訊", color=GOLD, fontsize=14, fontweight="bold", ha="center")
    _arrow(ax, (5.0, 2.2), (6.2, 2.2), c=MUT, lw=3, ms=20)
    ax.text(5.6, 2.55, "多層堆疊", color=MUT, fontsize=12, ha="center")
    # 右:更新後的節點嵌入
    ax.add_patch(Circle((8.4, 2.2), 0.4, fc=TEAL, ec="white", lw=1.5))
    ax.text(8.4, 2.2, "z", color="#0C1A2E", fontsize=18, fontweight="bold", ha="center", va="center")
    ax.text(8.4, 3.5, "② 更新成\n帶鄰域脈絡的嵌入", color=TX, fontsize=13, ha="center")
    ax.text(8.4, 0.7, "每棟建築的向量\n= 自己 + 街坊關係", color=TEAL, fontsize=12.5, ha="center", fontweight="bold")
    _save(fig, "m_gnn_mp.png")


# ==================================================== 案例研究(重畫概念 + 引用)
def _pipeline(ax, stages, y=2.2, x0=0.6, x1=10.4):
    n = len(stages); gap = (x1 - x0) / n
    cx = []
    for i, (label, c) in enumerate(stages):
        x = x0 + gap * i + 0.15
        _box(ax, x, y - 0.75, gap - 0.5, 1.5, "#1C3252")
        ax.text(x + (gap - 0.5) / 2, y, label, color=c, fontsize=13.5, fontweight="bold", ha="center", va="center")
        cx.append(x + (gap - 0.5) / 2)
        if i < n - 1:
            _arrow(ax, (x + gap - 0.5, y), (x + gap + 0.15, y), c=MUT, lw=2.5, ms=16)
    return cx


def cs_embeddings():
    fig, ax = _fig(11.5, 3.4); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4)
    _pipeline(ax, [("城市大數據\n(POI·街景·移動)", BLUE), ("神經嵌入\n(表徵學習)", GOLD),
                   ("2D 地圖:社區\n自動成結構", TEAL)], y=1.7)
    _save(fig, "cs_embeddings.png")


def cs_vae_street():
    fig, ax = _fig(11.5, 3.4); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4)
    _pipeline(ax, [("街道網路\n(圖/影像)", BLUE), ("VAE\nμ / σ 潛空間", GOLD),
                   ("低維向量抓住\n網路指標", TEAL), ("生成新的\n街道網路", CORAL)], y=1.7)
    _save(fig, "cs_vae_street.png")


def cs_gnn_building():
    fig, ax = _fig(11.5, 3.4); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4)
    _pipeline(ax, [("建築鄰接圖\n(節點+邊)", BLUE), ("GNN\n(GraphSAGE)", GOLD),
                   ("全城預測建築\n屬性/功能", TEAL)], y=1.7)
    _save(fig, "cs_gnn_building.png")


def cs_generative():
    fig, ax = _fig(11.5, 3.4); ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4)
    _pipeline(ax, [("設計草圖/\n語義遮罩", BLUE), ("GAN / 擴散\n生成模型", GOLD),
                   ("即時真實\n都市渲染", TEAL), ("人機互動\n迭代設計", CORAL)], y=1.7)
    _save(fig, "cs_generative.png")


def main():
    m_thesis(); m_graph(); m_gnn_mp()
    cs_embeddings(); cs_vae_street(); cs_gnn_building(); cs_generative()
    print("\n[DONE]", OUT)


if __name__ == "__main__":
    main()
