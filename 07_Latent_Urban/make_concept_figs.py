# -*- coding: utf-8 -*-
"""make_concept_figs.py —— 畫教學大綱用的概念示意圖(非資料結果,純圖解)

存到 out/figs/concept/。搭配 教學大綱.md 標「(示意)」處。
執行:  python make_concept_figs.py
"""
import sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nn._font import set_cjk_font
set_cjk_font()

OUT = Path(__file__).resolve().parent / "out" / "figs" / "concept"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED, GRAY = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#b8b8b8"


def _box(ax, x, y, w, h, text, fc, fs=11, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="none"))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold", wrap=True)


def _arrow(ax, x1, y1, x2, y2, c="k", lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=lw, color=c))


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[concept] {name}")


# ---------------------------------------------------- c0 動機
def c0_motivation():
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 5)
    ax.text(2.3, 4.7, "人腦:一次看 9 張分佈圖", ha="center", fontsize=12, fontweight="bold")
    rng = np.random.default_rng(0)
    for i in range(9):
        gx, gy = 0.5 + (i % 3)*1.3, 3.0 - (i//3)*1.3
        a = fig.add_axes([0.04 + (i % 3)*0.11, 0.24 + (2-i//3)*0.18, 0.09, 0.13])
        a.hist(rng.normal(size=60), bins=8, color=GRAY); a.set_xticks([]); a.set_yticks([])
    ax.text(2.3, 0.35, "眼花、看不出全貌", ha="center", fontsize=11, color=RED)
    _arrow(ax, 4.4, 2.5, 5.6, 2.5, c=ORANGE, lw=3)
    ax.text(5.0, 2.8, "NN 壓縮", ha="center", fontsize=11, color=ORANGE, fontweight="bold")
    # right: 3 clearly separated clusters
    for c, (cx, cy) in zip([BLUE, ORANGE, GREEN], [(-1.1, 0.9), (1.2, 1.1), (0.2, -1.2)]):
        p = rng.normal(size=(16, 2))*0.42 + [cx, cy]
        ax.scatter(8.4 + p[:, 0], 2.6 + p[:, 1], s=34, color=c, alpha=0.85, edgecolor="w", linewidth=0.4)
    ax.text(8.4, 4.7, "壓成一張 2D 潛空間圖", ha="center", fontsize=12, fontweight="bold")
    ax.text(8.4, 0.35, "一眼看出族群", ha="center", fontsize=11, color=GREEN)
    fig.suptitle("為什麼規劃要用 NN:把高維形態壓成看得懂的一張圖", fontsize=14)
    _save(fig, "c0_motivation.png")


# ---------------------------------------------------- c1 workflow 介入點
def c1_workflow():
    fig, ax = plt.subplots(figsize=(13, 4)); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 4)
    steps = ["多源\n拼接", "角色\n映射", "高度\n情景", "權力\n四體制", "3D\n量體"]
    for i, s in enumerate(steps):
        _box(ax, 0.3 + i*1.7, 2.2, 1.5, 1.2, s, BLUE, fs=10)
        if i < 4:
            _arrow(ax, 1.8 + i*1.7, 2.8, 2.1 + i*1.7, 2.8)
    ax.text(4.6, 3.7, "05 現有流程(不動一行)", ha="center", fontsize=12, color=BLUE, fontweight="bold")
    _arrow(ax, 8.85, 2.8, 9.7, 2.8, c=ORANGE, lw=3)
    ax.text(9.3, 3.2, "NN 從輸出接手", ha="center", fontsize=10, color=ORANGE, fontweight="bold")
    nn = ["壓縮\n(潛空間)", "分析\n(群聚)", "生成\n(新方案)"]
    for i, s in enumerate(nn):
        _box(ax, 9.9, 3.0 - i*1.15, 2.6, 0.95, s, ORANGE, fs=10)
    ax.text(6.3, 1.2, "資料粒度:每棟樓→向量 · 整城→高度影像 · 每城×體制→9 指標",
            ha="center", fontsize=10, color=GRAY)
    fig.suptitle("NN 在哪裡介入:接在 05 輸出後,不改資料來源", fontsize=14)
    _save(fig, "c1_workflow.png")


# ---------------------------------------------------- c2 三代方法漫畫
def c2_three_gen():
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for a in ax:
        a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 10)
    # 1 lookup 查字典
    ax[0].add_patch(Rectangle((2, 2), 6, 6, fc="#eef", ec=BLUE, lw=2))
    for i, (k, v) in enumerate([("商業", "開發商"), ("政府", "國家"), ("居住", "居民")]):
        ax[0].text(3, 6.5 - i*1.3, k, fontsize=12); ax[0].text(5, 6.5-i*1.3, "→", fontsize=12)
        ax[0].text(5.7, 6.5-i*1.3, v, fontsize=12, color=BLUE)
    ax[0].set_title("① 規則 Lookup:人訂對照\n(查分區規範)", fontsize=12)
    # 2 cluster 便利貼分堆
    rng = np.random.default_rng(1)
    for c, cx, cy in [(BLUE, 3, 6), (ORANGE, 7, 7), (GREEN, 6, 3)]:
        p = rng.normal(size=(9, 2))*0.7 + [cx, cy]
        ax[1].scatter(p[:, 0], p[:, 1], s=60, color=c, alpha=0.8)
    ax[1].set_title("② 非監督 ML:自己分堆\n(不給答案,找族群)", fontsize=12)
    # 3 generate 畫新城市
    rng = np.random.default_rng(3)
    for _ in range(14):
        x, y = rng.uniform(2, 8), rng.uniform(2, 6)
        h = rng.uniform(0.4, 2.2)
        ax[2].add_patch(Rectangle((x, y), 0.5, h, fc=ORANGE, ec="k", lw=0.4))
    ax[2].text(5, 8, "生成:沒看過\n但合理的城市", ha="center", fontsize=12, color=RED, fontweight="bold")
    ax[2].set_title("③ 神經網路:會創作\n(壓縮+重建+生成)", fontsize=12)
    fig.suptitle("三代方法:規則 → 非監督 ML → 神經網路", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "c2_three_generations.png")


# ---------------------------------------------------- c3 PCA 主軸
def c3_pca_axes():
    fig, ax = plt.subplots(figsize=(7, 5.5))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(180, 2)) @ [[2.2, 0], [0, 0.7]]
    ang = np.deg2rad(30); R = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    X = X @ np.array(R).T
    ax.scatter(X[:, 0], X[:, 1], s=16, color=GRAY, alpha=0.6)
    v1 = np.array([np.cos(ang), np.sin(ang)])*4
    v2 = np.array([-np.sin(ang), np.cos(ang)])*1.5
    _arrow(ax, 0, 0, v1[0], v1[1], c=RED, lw=3)
    _arrow(ax, 0, 0, v2[0], v2[1], c=BLUE, lw=3)
    ax.text(v1[0]*1.05, v1[1]*1.05, "PC1\n最長方向\n(強度/資本軸)", color=RED, fontsize=11, fontweight="bold")
    ax.text(v2[0]*1.1-1, v2[1]*1.1, "PC2\n次長", color=BLUE, fontsize=11, fontweight="bold")
    ax.set_xlim(-5, 6); ax.set_ylim(-4, 5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("PCA:找城市點雲「最長的方向」當新座標軸", fontsize=13)
    _save(fig, "c3_pca_axes.png")


# ---------------------------------------------------- c4 t-SNE 鄰居
def c4_tsne():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rng = np.random.default_rng(2)
    # 左:原始(散開)
    cents = [(2, 6), (7, 7), (5, 2), (8, 3)]
    cols = [BLUE, ORANGE, GREEN, RED]
    for c, (cx, cy) in zip(cols, cents):
        p = rng.normal(size=(12, 2))*0.9 + [cx, cy]
        ax[0].scatter(p[:, 0], p[:, 1], s=40, color=c, alpha=0.7)
    ax[0].set_title("原始高維(想像成散開)", fontsize=12)
    # 右:t-SNE(擠成緊團,團間亂放)
    tsne_c = [(2, 2), (2.5, 7.5), (8, 6), (7.5, 2)]
    for c, (cx, cy) in zip(cols, tsne_c):
        p = rng.normal(size=(12, 2))*0.35 + [cx, cy]
        ax[1].scatter(p[:, 0], p[:, 1], s=40, color=c, alpha=0.8)
    ax[1].set_title("t-SNE:把鄰居擠成緊團\n(團內對,團間距離無意義)", fontsize=12)
    for a in ax:
        a.set_xlim(0, 10); a.set_ylim(0, 10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("t-SNE:只保「誰跟誰是鄰居」,別讀團與團的距離", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "c4_tsne_neighbors.png")


# ---------------------------------------------------- c5 Isomap 瑞士捲
def c5_swissroll():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    t = np.linspace(1.2, 4.2*np.pi, 400)
    x, y = t*np.cos(t), t*np.sin(t)
    ax[0].scatter(x, y, c=t, cmap="viridis", s=12)
    # 穿牆的直線 vs 沿捲的路
    ax[0].plot([x[60], x[330]], [y[60], y[330]], "r--", lw=2)
    ax[0].text(0, 12, "紅虛線=直線距離『穿牆』\n(錯)", color=RED, fontsize=10, ha="center")
    ax[0].set_title("資料像捲起來的紙", fontsize=12); ax[0].set_aspect("equal")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    # 攤平
    ax[1].scatter(t, np.zeros_like(t)+np.sin(t*0)*0, c=t, cmap="viridis", s=14)
    ax[1].scatter(t, t*0, c=t, cmap="viridis", s=30)
    ax[1].set_ylim(-1, 1); ax[1].set_yticks([])
    ax[1].set_title("Isomap:沿紙面量距離、攤平\n(保全局結構)", fontsize=12)
    ax[1].set_xlabel("攤平後的位置")
    fig.suptitle("Isomap:把摺起來的資料沿流形攤平", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "c5_isomap_swissroll.png")


# ---------------------------------------------------- c6 AE 沙漏
def c6_ae_hourglass():
    fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
    ax.set_xlim(0, 11); ax.set_ylim(0, 6)
    # 編碼器梯形(寬→窄)
    ax.fill([1, 3, 3, 1], [1, 2.5, 3.5, 5], color=BLUE, alpha=0.85)
    ax.text(2, 3, "編碼器", ha="center", va="center", color="white", fontsize=12, fontweight="bold", rotation=90)
    # 瓶頸
    ax.add_patch(Rectangle((3, 2.5), 1.2, 1, fc=RED, ec="none"))
    ax.text(3.6, 4.0, "潛碼\n(2 個數字)", ha="center", fontsize=11, color=RED, fontweight="bold")
    ax.text(3.6, 1.7, "= 城市形態\nDNA", ha="center", fontsize=9, color=RED)
    # 解碼器梯形(窄→寬)
    ax.fill([4.2, 6.2, 6.2, 4.2], [2.5, 1, 5, 3.5], color=ORANGE, alpha=0.85)
    ax.text(5.2, 3, "解碼器", ha="center", va="center", color="white", fontsize=12, fontweight="bold", rotation=90)
    # 輸入輸出
    ax.text(0.5, 3, "原城市\n(9 維)", ha="center", fontsize=11)
    ax.text(6.9, 3, "還原\n(9 維)", ha="center", fontsize=11)
    _arrow(ax, 6.3, 3, 6.6, 3)
    ax.text(3.6, 5.5, "壓到最窄還能畫回來 = 抓住了精華", ha="center", fontsize=12, color="k", fontweight="bold")
    ax.text(9, 3.4, "重點:\n非線性壓縮\n→ 同 2 維\n重建遠比 PCA 準", ha="center", fontsize=11,
            color=GREEN, bbox=dict(boxstyle="round", fc="#eff7ef", ec=GREEN))
    fig.suptitle("AE 自編碼器:學徒「臨摹 → 只憑幾個數字默寫」", fontsize=14)
    _save(fig, "c6_ae_hourglass.png")


# ---------------------------------------------------- c7 VAE 連續潛空間
def c7_vae_continuity():
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    rng = np.random.default_rng(0)
    # AE: 有洞
    P = rng.normal(size=(40, 2))*1.6
    ax[0].scatter(P[:, 0], P[:, 1], s=30, color=BLUE, alpha=0.7)
    ax[0].scatter([0.3], [0.2], s=120, color=RED, marker="X")
    ax[0].text(0.5, 0.5, "踩進空洞\n→ 解出亂碼", color=RED, fontsize=10)
    ax[0].set_title("AE 潛空間:點與點之間是洞\n(不能亂採樣)", fontsize=12)
    # VAE: 填滿高斯
    th = np.linspace(0, 2*np.pi, 100)
    for r, a in [(1, 0.10), (2, 0.06), (3, 0.03)]:
        ax[1].fill(r*np.cos(th), r*np.sin(th), color=GREEN, alpha=a)
    P2 = rng.normal(size=(120, 2))
    ax[1].scatter(P2[:, 0], P2[:, 1], s=16, color=GREEN, alpha=0.5)
    ax[1].scatter([0.4], [-0.3], s=120, color=RED, marker="X")
    ax[1].text(0.6, -0.3, "隨處點一下\n→ 都生出合理城市", color=RED, fontsize=10)
    ax[1].set_title("VAE 潛空間:規範成連續高斯雲\n(處處可採樣 → 能生成)", fontsize=12)
    for a in ax:
        a.set_xlim(-4, 4); a.set_ylim(-4, 4); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.suptitle("VAE 比 AE 多會什麼:把潛空間填滿 → 能『造城』", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "c7_vae_continuity.png")


# ---------------------------------------------------- c8 方法光譜
def c8_spectrum():
    fig, ax = plt.subplots(figsize=(13, 4)); ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 4)
    # 漸層長條
    for i in range(130):
        ax.add_patch(Rectangle((0.5 + i*0.09, 1.6), 0.095, 0.6,
                     fc=plt.cm.RdYlGn_r(i/130), ec="none"))
    ax.text(0.5, 2.5, "可控 · 透明", fontsize=12, color=GREEN, fontweight="bold")
    ax.text(12.2, 2.5, "強大 · 能生成", fontsize=12, color=RED, fontweight="bold", ha="right")
    methods = [("Lookup\n規則", 1.2, BLUE), ("PCA", 3.4, BLUE), ("t-SNE/\nIsomap", 5.6, GRAY),
               ("AE", 8.0, ORANGE), ("VAE", 10.2, RED), ("Conv-VAE\n(CNN)", 12.0, RED)]
    for name, x, c in methods:
        ax.plot([x, x], [1.6, 2.2], color="k", lw=1)
        ax.add_patch(Circle((x, 1.9), 0.12, fc=c, ec="white", zorder=5))
        ax.text(x, 0.9, name, ha="center", fontsize=10, color=c, fontweight="bold")
    ax.text(6.5, 3.4, "沒有最好的方法,只有對應規劃問題的方法", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", xy=(12.3, 3.0), xytext=(0.5, 3.0),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=GRAY))
    ax.text(6.5, 0.2, "← 越左越可控可審    越右越能壓縮·重建·生成 →", ha="center", fontsize=10, color=GRAY)
    fig.suptitle("方法光譜:從『透明可控』到『強大能生成』", fontsize=14)
    _save(fig, "c8_method_spectrum.png")


def main():
    c0_motivation(); c1_workflow(); c2_three_gen(); c3_pca_axes(); c4_tsne()
    c5_swissroll(); c6_ae_hourglass(); c7_vae_continuity(); c8_spectrum()
    print(f"\n[DONE] 9 概念示意圖 → {OUT}")


if __name__ == "__main__":
    main()
