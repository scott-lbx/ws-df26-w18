# -*- coding: utf-8 -*-
"""cluster.py —— 無監督聚類 vs 手動 regimes(資料自己會分群嗎?)

手動 regimes = 人設定的 5 種權力配方。這裡反過來問:
**只看形態特徵、不告訴標籤,KMeans 會不會自己分出這 5 群?**
用 ARI(調整蘭德指數)量「無監督群 ↔ 手動 regime」的吻合度;
silhouette 量群本身的緊緻度。比不同潛空間(原始/PCA/AE/VAE)哪個最好分。

圖8 08_cluster_vs_regime.png  ARI/silhouette 條 + 最佳空間的 KMeans×regime 混淆熱圖
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

REGIMES = ["baseline", "developer_led", "state_led", "resident_self_build", "shared"]
REG_ZH = {"baseline": "基线", "developer_led": "开发商", "state_led": "政府",
          "resident_self_build": "居民自建", "shared": "共享"}


def latent_spaces(Xz):
    """回傳 {名称: 嵌入} 供聚类比较。"""
    d = Xz.shape[1]
    pca = L.PCAModel(k=2).fit(Xz)
    ae = L.build_ae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_ae(ae, Xz, epochs=800, lr=3e-3, weight_decay=1e-5, huber=False, seed=0)
    vae = L.build_vae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_vae(vae, Xz, epochs=900, lr=3e-3, beta=0.5, anneal=0.3, seed=0)
    return {
        "原始9维": Xz,
        "PCA-2": pca.transform(Xz),
        "AE-2": L.ae_encode(ae, Xz),
        "VAE-2": L.vae_encode(vae, Xz),
    }


def evaluate(df, Xz, k=5):
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    reg = df["regime"].values
    spaces = latent_spaces(Xz)
    rows, labels_best = [], {}
    for name, E in spaces.items():
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(E)
        ari = adjusted_rand_score(reg, km.labels_)
        sil = silhouette_score(E, km.labels_)
        rows.append({"space": name, "ARI": ari, "silhouette": sil})
        labels_best[name] = km.labels_
    import pandas as pd
    res = pd.DataFrame(rows)
    best = res.sort_values("ARI", ascending=False)["space"].iloc[0]
    return res, labels_best[best], best


def fig_cluster_vs_regime(df, Xz, k=5):
    import pandas as pd
    res, best_labels, best = evaluate(df, Xz, k)
    reg = df["regime"].values

    fig, ax = plt.subplots(1, 3, figsize=(16, 5),
                           gridspec_kw={"width_ratios": [1, 1, 1.2]})
    x = np.arange(len(res))
    ax[0].bar(x, res["ARI"], color="#4c72b0")
    ax[0].set_xticks(x); ax[0].set_xticklabels(res["space"], rotation=15)
    ax[0].set_ylabel("ARI(与手动 regime 吻合度)")
    ax[0].set_title("(a) KMeans 能否复原手动 regime")
    for i, v in enumerate(res["ARI"]):
        ax[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax[1].bar(x, res["silhouette"], color="#dd8452")
    ax[1].set_xticks(x); ax[1].set_xticklabels(res["space"], rotation=15)
    ax[1].set_ylabel("silhouette(群紧致度)")
    ax[1].set_title("(b) 无监督群本身好不好")
    for i, v in enumerate(res["silhouette"]):
        ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # (c) 最佳空间:KMeans 群 × 手动 regime 混淆
    ct = pd.crosstab(pd.Series(best_labels, name="KMeans 群"),
                     pd.Series(reg, name="手动 regime"))
    ct = ct.reindex(columns=[r for r in REGIMES if r in ct.columns])
    im = ax[2].imshow(ct.values, cmap="Blues", aspect="auto")
    ax[2].set_xticks(range(ct.shape[1]))
    ax[2].set_xticklabels([REG_ZH.get(c, c) for c in ct.columns], rotation=30, fontsize=9)
    ax[2].set_yticks(range(ct.shape[0]))
    ax[2].set_yticklabels([f"群{i}" for i in ct.index])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax[2].text(j, i, ct.values[i, j], ha="center", va="center", fontsize=9,
                       color="white" if ct.values[i, j] > ct.values.max()/2 else "k")
    ax[2].set_title(f"(c) 最佳空间「{best}」:群 × regime 混淆")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

    cap = ("图说:(a) ARI 高=无监督 KMeans 只看形态就复原了人设的 regime→regime 有数据依据;"
           "低=数据自分的群和人设不同(另有结构)。(b) 群紧致度。(c) 对角线越集中,"
           "自动群越对应单一 regime。→ 深度潜空间能否比手动分类更细/更客观地找到城市形态族群。")
    fig.text(0.5, -0.02, cap, ha="center", fontsize=9, wrap=True)
    fig.suptitle("无监督聚类 vs 手动 regimes:数据自己会分群吗", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = FIGS / "08_cluster_vs_regime.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p.name}")
    print("\n=== 聚类 vs regime ===\n", res.to_string(index=False))
    print(f"最佳空间: {best}")
    return res


def main():
    df, X = _load()
    std = L.Standardizer().fit(X)
    Xz = std.transform(X)
    fig_cluster_vs_regime(df, Xz)
    print(f"\n[DONE] {FIGS}")


if __name__ == "__main__":
    main()
