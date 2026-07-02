# -*- coding: utf-8 -*-
"""analysis.py —— PCA/AE/VAE 训练 + 分析,出真实图说

图 1  01_method_compare.png   PCA vs UMAP vs t-SNE(降维方法差异)
图 2  02_ae_vae_loss.png      AE/VAE 训练损失深入(recon/KL 分解、train/val、β 退火)
图 3  03_recon_compare.png    重建损失对照(PCA-k vs AE vs VAE;整体 + 逐特征)
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

REGIME_MARK = {"baseline": "o", "developer_led": "s", "state_led": "^",
               "resident_self_build": "D", "shared": "P"}


def manifold_proj(Xz):
    """流形投影:优先 UMAP;没装 umap 就退回 sklearn Isomap(处处可跑)。
    回传 (embedding[N,2], 标签)。"""
    try:
        import umap
        E = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.3,
                      random_state=0).fit_transform(Xz)
        return E, "UMAP(流形+全局)"
    except Exception:
        from sklearn.manifold import Isomap
        E = Isomap(n_components=2, n_neighbors=10).fit_transform(Xz)
        return E, "Isomap(流形;未装 umap 时替代)"


def load():
    df = pd.read_csv(CSV)
    X = df[L.FEATURES].values.astype(float)
    return df, X


def color_map(labels):
    uniq = sorted(set(labels))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(uniq), 3)))
    return {u: cmap[i] for i, u in enumerate(uniq)}


# ---------------------------------------------------- 图1:降维方法差异
def fig_method_compare(df, Xz):
    from sklearn.manifold import TSNE

    pca = L.PCAModel(k=2).fit(Xz)
    P_pca = pca.transform(Xz)
    P_tsne = TSNE(n_components=2, perplexity=8, init="pca",
                  random_state=0).fit_transform(Xz)
    P_mani, mani_label = manifold_proj(Xz)

    fams = df["family"].str.split("/").str[0].values
    cmap = color_map(fams)
    panels = [("PCA(线性投影)", P_pca), ("t-SNE(局部邻域)", P_tsne),
              (mani_label, P_mani)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (title, P) in zip(axes, panels):
        for i in range(len(df)):
            ax.scatter(P[i, 0], P[i, 1], color=cmap[fams[i]],
                       marker=REGIME_MARK.get(df.regime.iloc[i], "o"),
                       s=70, edgecolor="k", linewidth=0.4, alpha=0.85)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_xlabel(f"PC1  {pca.explained_ratio_[0]*100:.0f}% 方差")
    axes[0].set_ylabel(f"PC2  {pca.explained_ratio_[1]*100:.0f}% 方差")

    fam_handles = [plt.Line2D([0], [0], marker='o', color='w', label=f,
                   markerfacecolor=cmap[f], markeredgecolor='k', markersize=9)
                   for f in cmap]
    reg_handles = [plt.Line2D([0], [0], marker=m, color='w', label=r,
                   markerfacecolor='gray', markeredgecolor='k', markersize=9)
                   for r, m in REGIME_MARK.items()]
    axes[2].legend(handles=fam_handles, title="家族(颜色)",
                   loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.legend(handles=reg_handles, title="体制(形状)", loc="lower center",
               bbox_to_anchor=(0.5, 0.07), ncol=5, fontsize=8)
    fig.suptitle("降维方法差异:同一 9 维形态语料,三种投影的结构", fontsize=15)
    cap = ("图说:PCA 保全局方差、轴可解释但线性;t-SNE 夸大局部簇、忽略全局距离;"
           "第三格(UMAP;未装则 Isomap)看流形/全局结构。看家族(颜色)是否成簇、体制(形状)如何在城内扰动。")
    fig.text(0.5, 0.005, cap, ha="center", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0.13, 1, 0.95])
    p = FIGS / "01_method_compare.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


# ---------------------------------------------------- 图2:AE/VAE 训练损失深入
def city_group_split(df, val_cities=("beiwaitan", "dapuqiao", "kongjiang")):
    val = df["slug"].isin(val_cities).values
    return ~val, val


def fig_loss(df, Xz):
    tr, va = city_group_split(df)
    Xtr, Xval = Xz[tr], Xz[va]
    d = Xz.shape[1]

    ae_a = L.build_ae(d, latent=2, hidden=(16, 8), seed=0)
    h_ae_a = L.train_ae(ae_a, Xtr, Xval, epochs=500, lr=3e-3,
                        weight_decay=1e-4, denoise=0.05, seed=0)
    ae_b = L.build_ae(d, latent=2, hidden=(16, 8), bn=False, seed=0)
    h_ae_b = L.train_ae(ae_b, Xtr, Xval, epochs=500, lr=3e-3,
                        weight_decay=0.0, denoise=0.0, seed=0)

    vae_a = L.build_vae(d, latent=2, hidden=(16, 8), seed=0)
    h_v_a = L.train_vae(vae_a, Xtr, Xval, epochs=700, lr=3e-3,
                        beta=1.0, anneal=0.4, seed=0)
    vae_b = L.build_vae(d, latent=2, hidden=(16, 8), seed=0)
    h_v_b = L.train_vae(vae_b, Xtr, Xval, epochs=700, lr=3e-3,
                        beta=4.0, anneal=0.0, seed=0)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    ax[0].plot(h_ae_a["train"], label="正则AE 训练", color="C0")
    ax[0].plot(h_ae_a["val"], "--", label="正则AE 验证", color="C0")
    ax[0].plot(h_ae_b["train"], label="裸AE 训练", color="C3")
    ax[0].plot(h_ae_b["val"], "--", label="裸AE 验证", color="C3")
    ax[0].set_title("(a) AE:正则 vs 裸训练"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Huber loss"); ax[0].legend(fontsize=8); ax[0].set_yscale("log")

    ax[1].plot(h_v_a["total"], label="total", color="k")
    ax[1].plot(h_v_a["recon"], label="recon", color="C0")
    ax[1].plot(h_v_a["kl"], label="KL", color="C1")
    ax[1].set_title("(b) VAE 损失分解(β退火)"); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss"); ax[1].legend(fontsize=8); ax[1].set_yscale("log")

    ax[2].plot(h_v_a["recon"], label="β退火 recon", color="C0")
    ax[2].plot(h_v_a["kl"], label="β退火 KL", color="C0", ls=":")
    ax[2].plot(h_v_b["recon"], label="β=4固定 recon", color="C3")
    ax[2].plot(h_v_b["kl"], label="β=4固定 KL", color="C3", ls=":")
    ax[2].set_title("(c) β退火 vs β=4(posterior collapse)")
    ax[2].set_xlabel("epoch"); ax[2].set_ylabel("loss"); ax[2].legend(fontsize=8)
    ax[2].set_yscale("log")

    cap = ("图说:(a) 裸AE 训练↓但验证↑=过拟合;正则AE(weight_decay+denoise)验证更稳。"
           "(b) VAE recon 与 KL 此消彼长,退火让 KL 慢慢介入。(c) 固定大 β 压垮 KL→"
           "posterior collapse(KL→0、潜码失效);退火避免。")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("AE/VAE 训练损失深入:过拟合、损失分解、β 退火", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = FIGS / "02_ae_vae_loss.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")
    return dict(ae_a=ae_a, vae_a=vae_a)


# ---------------------------------------------------- 图3:重建损失对照
def fig_recon(df, Xz, models=None):
    d = Xz.shape[1]
    results = {}
    for k in (2, 3, 4):
        m = L.PCAModel(k=k).fit(Xz)
        results[f"PCA k={k}"] = L.recon_mse(Xz, m.reconstruct(Xz))
    for lat in (2, 3):
        ae = L.build_ae(d, latent=lat, hidden=(16, 8), seed=0)
        L.train_ae(ae, Xz, epochs=800, lr=3e-3, weight_decay=1e-5,
                   denoise=0.0, seed=0, huber=False)
        results[f"AE latent={lat}"] = L.recon_mse(Xz, L.ae_reconstruct(ae, Xz))
    per_feat_vae = None
    for lat in (2, 3):
        vae = L.build_vae(d, latent=lat, hidden=(16, 8), seed=0)
        L.train_vae(vae, Xz, epochs=900, lr=3e-3, beta=0.3, anneal=0.3, seed=0)
        xh = L.vae_reconstruct(vae, Xz)
        results[f"VAE latent={lat}"] = L.recon_mse(Xz, xh)
        if lat == 2:
            per_feat_vae = L.recon_mse(Xz, xh, per_feature=True)

    pca2 = L.PCAModel(k=2).fit(Xz)
    pf_pca = L.recon_mse(Xz, pca2.reconstruct(Xz), per_feature=True)
    ae2 = L.build_ae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_ae(ae2, Xz, epochs=800, lr=3e-3, weight_decay=1e-5, huber=False, seed=0)
    pf_ae = L.recon_mse(Xz, L.ae_reconstruct(ae2, Xz), per_feature=True)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    names = list(results.keys())
    vals = [results[n] for n in names]
    colors = ["#4c72b0" if "PCA" in n else "#dd8452" if "AE" in n else "#55a868"
              for n in names]
    ax[0].bar(names, vals, color=colors)
    ax[0].set_ylabel("整体重建 MSE(标准化空间)")
    ax[0].set_title("(a) 重建损失:方法 × 潜维")
    ax[0].tick_params(axis="x", rotation=30)
    for i, v in enumerate(vals):
        ax[0].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    x = np.arange(len(L.FEATURES)); w = 0.27
    ax[1].bar(x - w, pf_pca, w, label="PCA k=2", color="#4c72b0")
    ax[1].bar(x, pf_ae, w, label="AE latent=2", color="#dd8452")
    ax[1].bar(x + w, per_feat_vae, w, label="VAE latent=2", color="#55a868")
    ax[1].set_xticks(x); ax[1].set_xticklabels(L.FEATURES, rotation=40, fontsize=8)
    ax[1].set_ylabel("逐特征重建 MSE"); ax[1].set_title("(b) 谁的哪一维还原得好")
    ax[1].legend(fontsize=8)

    cap = ("图说:(a) 同样压到 2 维,AE/VAE 非线性瓶颈重建 MSE 低于 PCA;加维后差距缩小。"
           "(b) PCA 在高方差主轴维(far/h_max)还原好,AE 更均衡;VAE 因 KL 正则略让步换连续潜空间。")
    fig.text(0.5, -0.03, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("重建损失对照:AE/VAE 能压到比 PCA 更低", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = FIGS / "03_recon_compare.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")

    print("\n=== 整体重建 MSE(越低越好)===")
    for n in names:
        print(f"  {n:16s} {results[n]:.4f}")
    return results


def main():
    df, X = load()
    std = L.Standardizer().fit(X)
    Xz = std.transform(X)
    print(f"语料 {Xz.shape[0]} 样本 × {Xz.shape[1]} 特征;标准化完成")
    fig_method_compare(df, Xz)
    fig_loss(df, Xz)
    fig_recon(df, Xz)
    print(f"\n[DONE] 图存于 {FIGS}")


if __name__ == "__main__":
    main()
