# -*- coding: utf-8 -*-
"""generate.py —— 潜空间「采样 + 生成」能力比较(PCA / t-SNE / Isomap / AE / VAE)

核心事实:能不能「生成」= 有没有 decoder(逆映射)。
    PCA    有 inverse_transform → 可生成(线性)
    t-SNE  无逆映射            → **不可生成**(纯可视化)
    Isomap 无逆映射            → **不可生成**
    AE     有 decoder          → 可生成,但潜空间无先验、有洞
    VAE    有 decoder + N(0,I) 先验 → 可从先验采样,原则性生成

图6 06_gen_capability.png  能力矩阵 + 采样品质(plausibility / validity)
图7 07_gen_cloud.png       真实 vs 生成点云(共享 PCA-2D)+ 插值平滑度

指标(都在标准化空间 Xz 上量,validity 换回真实尺度查物理约束):
  plausibility  生成点到最近真实样本的距离(σ,越低越像真实数据流形)
  validity      生成样本满足物理约束的比例(far>0 / 0≤coverage≤1 / h_max≥h_mean ...)
  interp        潜空间插值路径中点到流形的距离(越低=插值越连续可用)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import encoders as L
from . import _paths
from .analysis import load as _load

_paths.set_cjk_font()
FIGS = _paths.FIGS
FIGS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)
N_GEN = 400

# 能力矩阵:方法 × [编码, 解码/逆映射, 先验采样, 插值]
CAPABILITY = {
    "PCA":    [1, 1, 1, 1],   # 逆变换;可在主轴分布采样;线性插值
    "t-SNE":  [1, 0, 0, 0],   # 只有 forward、无逆映射
    "Isomap": [1, 0, 0, 0],   # 同上
    "AE":     [1, 1, 0, 1],   # 有 decoder,但无先验(采样点可能落在洞里)
    "VAE":    [1, 1, 1, 1],   # decoder + N(0,I) 先验,可原则性采样
}
CAP_COLS = ["编码", "解码(逆映射)", "先验采样", "插值生成"]


# ---------------------------------------------------- 指标
def nn_dist(A, B):
    """A 每行到 B 最近点的欧氏距离。"""
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return d.min(1)


def validity(real_feats):
    """真实尺度特征是否满足物理约束。real_feats: (N,9) 对应 L.FEATURES 顺序。"""
    f = {name: real_feats[:, i] for i, name in enumerate(L.FEATURES)}
    ok = ((f["far"] > 0) & (f["coverage"] >= 0) & (f["coverage"] <= 1.05) &
          (f["h_mean"] > 0) & (f["h_max"] > 0) & (f["h_max"] >= f["h_mean"] * 0.99) &
          (f["h_cv"] >= 0) & (f["grain"] > 0) & (f["slender"] > 0) &
          (f["concentration"] >= 0) & (f["concentration"] <= 1.05) & (f["n"] >= 1))
    return ok


# ---------------------------------------------------- 采样(只对可生成方法)
def sample_pca(pca, Xz, n):
    S = pca.transform(Xz)                       # 主轴分数
    mu, cov = S.mean(0), np.cov(S.T)
    Zs = RNG.multivariate_normal(mu, cov, size=n)
    return pca.inverse(Zs)


def sample_ae(model, Xz, n):
    Z = L.ae_encode(model, Xz)                  # 无先验 → 用经验分布近似
    mu, cov = Z.mean(0), np.cov(Z.T)
    Zs = RNG.multivariate_normal(mu, cov, size=n)
    return L.ae_decode(model, Zs)


def sample_vae(model, Xz, n):
    latent = model.mu.out_features                   # 潜维
    Zs = RNG.standard_normal((n, latent))            # 直接从先验 N(0,I) 采样
    return L.vae_decode(model, Zs)


def interp_path(encode_fn, decode_fn, Xz, real, pairs=12, steps=11):
    """随机取 pairs 对真实样本,潜空间线性插值 steps 步,量中点到流形距离。"""
    idx = RNG.integers(0, len(Xz), size=(pairs, 2))
    mids = []
    for a, b in idx:
        za, zb = encode_fn(Xz[a:a+1])[0], encode_fn(Xz[b:b+1])[0]
        ts = np.linspace(0, 1, steps)[1:-1]              # 去两端
        Z = np.array([(1-t)*za + t*zb for t in ts])
        mids.append(decode_fn(Z))
    mids = np.vstack(mids)
    return float(nn_dist(mids, real).mean())


# ---------------------------------------------------- 训练三个可生成模型
def _fit_models(Xz):
    d = Xz.shape[1]
    pca = L.PCAModel(k=2).fit(Xz)
    ae = L.build_ae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_ae(ae, Xz, epochs=800, lr=3e-3, weight_decay=1e-5, huber=False, seed=0)
    vae = L.build_vae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_vae(vae, Xz, epochs=900, lr=3e-3, beta=0.5, anneal=0.3, seed=0)
    return pca, ae, vae


def run(df, Xz, std):
    pca, ae, vae = _fit_models(Xz)
    gen = {
        "PCA": sample_pca(pca, Xz, N_GEN),
        "AE": sample_ae(ae, Xz, N_GEN),
        "VAE": sample_vae(vae, Xz, N_GEN),
    }
    metrics = {}
    for name, G in gen.items():
        plaus = float(nn_dist(G, Xz).mean())
        val = float(validity(std.inverse(G)).mean() * 100)
        metrics[name] = {"plaus": plaus, "valid": val}
    # 插值平滑度
    metrics["PCA"]["interp"] = interp_path(pca.transform, pca.inverse, Xz, Xz)
    metrics["AE"]["interp"] = interp_path(
        lambda X: L.ae_encode(ae, X), lambda Z: L.ae_decode(ae, Z), Xz, Xz)
    metrics["VAE"]["interp"] = interp_path(
        lambda X: L.vae_encode(vae, X), lambda Z: L.vae_decode(vae, Z), Xz, Xz)
    return pca, gen, metrics


# ---------------------------------------------------- 图6:能力矩阵 + 品质
def fig_capability(metrics):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5),
                           gridspec_kw={"width_ratios": [1.2, 1, 1]})
    # (a) 能力矩阵
    M = np.array([CAPABILITY[m] for m in CAPABILITY])
    ax[0].imshow(M, cmap="Greens", vmin=0, vmax=1.4, aspect="auto")
    ax[0].set_xticks(range(len(CAP_COLS))); ax[0].set_xticklabels(CAP_COLS, fontsize=10)
    ax[0].set_yticks(range(len(CAPABILITY))); ax[0].set_yticklabels(list(CAPABILITY), fontsize=11)
    for i, m in enumerate(CAPABILITY):
        for j in range(len(CAP_COLS)):
            ax[0].text(j, i, "有" if CAPABILITY[m][j] else "無",
                       ha="center", va="center", fontsize=13, fontweight="bold",
                       color="white" if CAPABILITY[m][j] else "#b02020")
    ax[0].set_title("(a) 生成能力矩阵:有 decoder 才能生成")

    gens = ["PCA", "AE", "VAE"]
    cols = ["#4c72b0", "#dd8452", "#55a868"]
    # (b) plausibility(越低越像真实)
    pv = [metrics[g]["plaus"] for g in gens]
    ax[1].bar(gens, pv, color=cols)
    for i, v in enumerate(pv):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel("生成点→最近真实样本距离(σ)")
    ax[1].set_title("(b) plausibility(越低越像真实)")
    # (c) validity(越高越合理)
    vv = [metrics[g]["valid"] for g in gens]
    ax[2].bar(gens, vv, color=cols)
    for i, v in enumerate(vv):
        ax[2].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax[2].set_ylabel("满足物理约束比例 %"); ax[2].set_ylim(0, 105)
    ax[2].set_title("(c) validity(越高越合理)")

    cap = ("图说:(a) t-SNE/Isomap 无逆映射→**不能生成**,只能可视化。(b) 从潜空间采样解码,"
           "VAE(先验 N(0,I))与 PCA 生成点更贴近真实流形;AE 无先验、采样易落洞→偏离。"
           "(c) 生成样本守物理约束的比例,VAE 通常最稳。")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("潜空间采样与生成能力:谁能生成、生成得多真", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = FIGS / "06_gen_capability.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


# ---------------------------------------------------- 图7:点云 + 插值
def fig_cloud(pca_ref, Xz, gen, metrics):
    Pr = pca_ref.transform(Xz)
    gens = ["PCA", "AE", "VAE"]; cols = ["#4c72b0", "#dd8452", "#55a868"]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.8),
                           gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    for k, g in enumerate(gens):
        Pg = pca_ref.transform(gen[g])
        ax[k].scatter(Pg[:, 0], Pg[:, 1], s=10, alpha=0.35, color=cols[k],
                      label="生成", zorder=1)
        ax[k].scatter(Pr[:, 0], Pr[:, 1], s=45, color="k", edgecolor="w",
                      linewidth=0.5, label="真实", zorder=3)
        ax[k].set_title(f"{g} 采样(共享 PCA-2D)")
        ax[k].set_xticks([]); ax[k].set_yticks([])
        ax[k].legend(fontsize=8, loc="upper right")
    # (d) 插值平滑度
    iv = [metrics[g]["interp"] for g in gens]
    ax[3].bar(gens, iv, color=cols)
    for i, v in enumerate(iv):
        ax[3].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax[3].set_ylabel("插值中点→流形距离(σ)")
    ax[3].set_title("(d) 插值连续性(越低越好)")

    cap = ("图说:黑=真实 60 点、彩=各法生成 400 点,投到同一 PCA-2D。生成点云应**盖住**真实、"
           "不外溢也不塌成一点。(d) 两真实样本潜空间插值,中间点离流形越近=潜空间越连续可插值(VAE 优势)。")
    fig.text(0.5, -0.03, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("真实 vs 生成点云 + 潜空间插值连续性", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = FIGS / "07_gen_cloud.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")


def main():
    df, X = _load()
    std = L.Standardizer().fit(X)
    Xz = std.transform(X)
    pca_ref, gen, metrics = run(df, Xz, std)
    fig_capability(metrics)
    fig_cloud(pca_ref, Xz, gen, metrics)
    print("\n=== 生成能力指标 ===")
    print(f"{'方法':6s} {'plausibility↓':>14s} {'validity%↑':>11s} {'interp↓':>9s}")
    for g in ("PCA", "AE", "VAE"):
        m = metrics[g]
        print(f"{g:6s} {m['plaus']:>14.3f} {m['valid']:>10.0f}% {m['interp']:>9.3f}")
    print("t-SNE / Isomap: 无 decoder,不可采样/生成/插值。")
    print(f"\n[DONE] {FIGS}")
    return metrics


if __name__ == "__main__":
    main()
