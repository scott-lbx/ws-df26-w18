# -*- coding: utf-8 -*-
"""interpret.py —— 群聚可解释性 + 4 体制解读

图4 04_cluster_interpret.png  PCA 轴载荷 + 三方法 silhouette(按家族/按体制)
图5 05_regime_signature.png   4 体制形态签名(Δ特征热图) + PCA 空间体制平移向量
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import encoders as L
from . import _paths

_paths.set_cjk_font()
FIGS = _paths.FIGS
FIGS.mkdir(parents=True, exist_ok=True)
CSV = _paths.CSV

REGIMES = ["baseline", "developer_led", "state_led", "resident_self_build", "shared"]
REG_ZH = {"baseline": "基线", "developer_led": "开发商主导", "state_led": "政府主导",
          "resident_self_build": "居民自建", "shared": "共享"}


def load():
    df = pd.read_csv(CSV)
    X = df[L.FEATURES].values.astype(float)
    std = L.Standardizer().fit(X)
    return df, std.transform(X), std


def silhouettes(df, Xz):
    from sklearn.metrics import silhouette_score
    from sklearn.manifold import TSNE
    from .analysis import manifold_proj
    fam = df["family"].str.split("/").str[0].values
    reg = df["regime"].values
    P_mani, mani_label = manifold_proj(Xz)
    mani_key = mani_label.split("(")[0]   # "UMAP" 或 "Isomap"
    embeds = {
        "原始9维": Xz,
        "PCA": L.PCAModel(k=2).fit(Xz).transform(Xz),
        "t-SNE": TSNE(2, perplexity=8, init="pca", random_state=0).fit_transform(Xz),
        mani_key: P_mani,
    }
    rows = []
    for name, E in embeds.items():
        rows.append({"space": name,
                     "按家族": silhouette_score(E, fam),
                     "按体制": silhouette_score(E, reg)})
    return pd.DataFrame(rows), embeds


def fig_cluster_interpret(df, Xz):
    sil, embeds = silhouettes(df, Xz)
    pca = L.PCAModel(k=3).fit(Xz)
    load = pca.components_

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6),
                           gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax[0].imshow(load, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax[0].set_xticks(range(len(L.FEATURES)))
    ax[0].set_xticklabels(L.FEATURES, rotation=40, fontsize=9)
    ax[0].set_yticks(range(3))
    ax[0].set_yticklabels([f"PC{i+1}\n{pca.explained_ratio_[i]*100:.0f}%"
                           for i in range(3)])
    for i in range(3):
        for j in range(len(L.FEATURES)):
            ax[0].text(j, i, f"{load[i,j]:+.2f}", ha="center", va="center",
                       fontsize=7, color="k")
    ax[0].set_title("(a) PCA 轴载荷:每根主轴 = 哪些特征的线性组合")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="载荷")

    x = np.arange(len(sil)); w = 0.36
    ax[1].bar(x - w/2, sil["按家族"], w, label="按家族标签", color="#4c72b0")
    ax[1].bar(x + w/2, sil["按体制"], w, label="按体制标签", color="#dd8452")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(sil["space"])
    ax[1].set_ylabel("silhouette(越高越成簇)")
    ax[1].set_title("(b) 谁在成簇:家族 vs 体制")
    ax[1].legend(fontsize=9)
    for i, r in sil.iterrows():
        ax[1].text(i - w/2, r["按家族"], f'{r["按家族"]:.2f}', ha="center",
                   va="bottom" if r["按家族"] >= 0 else "top", fontsize=7)
        ax[1].text(i + w/2, r["按体制"], f'{r["按体制"]:.2f}', ha="center",
                   va="bottom" if r["按体制"] >= 0 else "top", fontsize=7)

    cap = ("图说:(a) PC1 载荷**负**=高强度/资本(far/h_mean/h_cv/grain 密集高耸),"
           "**正**=瘦长细多(slender/n 自建细粒);PC2 见 h_mean/slender 的另一组权衡。"
           "(b) 三方法 silhouette 都是**按体制**远高于**按家族**(家族近 0/负)——"
           "群聚由**权力体制**主导:同一算子把各城推向一致方向,城市家族反被打散;UMAP/t-SNE 放大体制簇。")
    fig.text(0.5, 0.005, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("群聚可解释性:轴的含义 + 谁在成簇", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    p = FIGS / "04_cluster_interpret.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")
    print("\n=== silhouette ===\n", sil.to_string(index=False))
    return pca


def fig_regime_signature(df, Xz, std, pca):
    idx = {(r["slug"], r["regime"]): i for i, r in df.iterrows()}
    sig = {}
    for reg in REGIMES[1:]:
        deltas = []
        for slug in df["slug"].unique():
            b, a = idx.get((slug, "baseline")), idx.get((slug, reg))
            if b is not None and a is not None:
                deltas.append(Xz[a] - Xz[b])
        sig[reg] = np.mean(deltas, 0)
    S = np.array([sig[r] for r in REGIMES[1:]])

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.6),
                           gridspec_kw={"width_ratios": [1.3, 1]})
    vmax = np.abs(S).max()
    im = ax[0].imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax[0].set_xticks(range(len(L.FEATURES)))
    ax[0].set_xticklabels(L.FEATURES, rotation=40, fontsize=9)
    ax[0].set_yticks(range(4))
    ax[0].set_yticklabels([REG_ZH[r] for r in REGIMES[1:]], fontsize=11)
    for i in range(4):
        for j in range(len(L.FEATURES)):
            ax[0].text(j, i, f"{S[i,j]:+.1f}", ha="center", va="center",
                       fontsize=7, color="k")
    ax[0].set_title("(a) 体制形态签名:baseline→体制 的 Δ特征(标准化,12城均值)")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04, label="Δ(σ)")

    P = pca.transform(Xz)
    pidx = {(r["slug"], r["regime"]): i for i, r in df.iterrows()}
    colors = {"developer_led": "#c44e52", "state_led": "#4c72b0",
              "resident_self_build": "#55a868", "shared": "#8172b3"}
    Pb = P[df.regime == "baseline"]
    c0 = Pb.mean(0)
    ax[1].scatter(Pb[:, 0], Pb[:, 1], c="gray", s=40, alpha=0.5,
                  label="baseline", zorder=2)
    tips = [[Pb[:, 0].min(), Pb[:, 1].min()], [Pb[:, 0].max(), Pb[:, 1].max()]]
    for reg in REGIMES[1:]:
        dx = dy = 0; n = 0
        for slug in df["slug"].unique():
            b, a = pidx.get((slug, "baseline")), pidx.get((slug, reg))
            if b is not None and a is not None:
                dx += P[a, 0] - P[b, 0]; dy += P[a, 1] - P[b, 1]; n += 1
        dx /= n; dy /= n
        tip = (float(c0[0] + dx), float(c0[1] + dy))
        tips.append([tip[0], tip[1]])
        ax[1].annotate("", xy=tip, xytext=(c0[0], c0[1]),
                       arrowprops=dict(arrowstyle="-|>", color=colors[reg], lw=2.5))
        ax[1].text(tip[0], tip[1], REG_ZH[reg], color=colors[reg],
                   fontsize=11, ha="center", va="bottom", fontweight="bold")
    T = np.array(tips)
    mx = (T[:, 0].max() - T[:, 0].min()) * 0.15 + 0.5
    my = (T[:, 1].max() - T[:, 1].min()) * 0.15 + 0.5
    ax[1].set_xlim(T[:, 0].min() - mx, T[:, 0].max() + mx)
    ax[1].set_ylim(T[:, 1].min() - my, T[:, 1].max() + my)
    ax[1].axhline(0, color="k", lw=0.4); ax[1].axvline(0, color="k", lw=0.4)
    ax[1].set_xlabel(f"PC1 {pca.explained_ratio_[0]*100:.0f}%(强度/资本 →)")
    ax[1].set_ylabel(f"PC2 {pca.explained_ratio_[1]*100:.0f}%")
    ax[1].set_title("(b) PCA 空间:体制把城市推向哪个方向(均值平移)")
    ax[1].legend(fontsize=8, loc="upper right")

    cap = ("图说:(a) 红=该体制抬升该特征、蓝=压低。开发商→far/slender/h_max↑(塔化增密);"
           "政府→concentration↑(向公共重心收拢);居民自建→grain↓/n↑(细粒化);共享→coverage↓开放地面。"
           "(b) 同一起点,四体制平移方向不同=可解释的『权力如何重写形态』。")
    fig.text(0.5, 0.005, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("4 权力体制解读:各自的形态签名与位移方向", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    p = FIGS / "05_regime_signature.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")
    print("\n=== 体制签名 Δ(σ,12城均值)===")
    print(pd.DataFrame(S, columns=L.FEATURES,
                       index=[REG_ZH[r] for r in REGIMES[1:]]).round(2).to_string())


def main():
    df, Xz, std = load()
    pca = fig_cluster_interpret(df, Xz)
    fig_regime_signature(df, Xz, std, pca)
    print(f"\n[DONE] {FIGS}")


if __name__ == "__main__":
    main()
