# -*- coding: utf-8 -*-
"""slides_figs.py —— 匯出「裸面板」給簡報:每個 subplot 拆成獨立 PNG,
圖內不放標題/圖例/說明文字(那些改由 PPTX 原生文字承載),透明背景、深底友善。

產出:out/figs/slides/  (p_*.png)
執行:  python slides_figs.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from regime_latent import encoders as L
from regime_latent import _paths
from regime_latent.analysis import load as _load, manifold_proj, city_group_split

OUT = _paths.OUT / "figs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

TICK = "C9D1D9"      # 淺灰刻度,深底可見
GRID = "#33404F"
BLUE, ORANGE, GREEN, RED, PURPLE = "#5AB0FF", "#F2C14E", "#4EC9B0", "#F96167", "#C586C0"
FAM_C = {"资本": "#5AB0FF", "单位": "#F2C14E", "居民": "#4EC9B0"}
REG_MARK = {"baseline": "o", "developer_led": "s", "state_led": "^",
            "resident_self_build": "D", "shared": "P"}

plt.rcParams.update({
    "text.color": "#" + TICK, "axes.labelcolor": "#" + TICK,
    "xtick.color": "#" + TICK, "ytick.color": "#" + TICK,
    "axes.edgecolor": GRID, "font.size": 12,
})


def _bare(ax):
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors="#" + TICK, labelsize=9)


def _save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("[slide-fig]", name)


# ---------------- compute once ----------------
def _setup():
    df, X = _load()
    std = L.Standardizer().fit(X)
    Xz = std.transform(X)
    d = Xz.shape[1]
    pca3 = L.PCAModel(k=3).fit(Xz)
    ae = L.build_ae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_ae(ae, Xz, epochs=800, lr=3e-3, weight_decay=1e-5, huber=False, seed=0)
    vae = L.build_vae(d, latent=2, hidden=(16, 8), seed=0)
    L.train_vae(vae, Xz, epochs=900, lr=3e-3, beta=0.5, anneal=0.3, seed=0)
    return df, X, Xz, std, pca3, ae, vae


def main():
    df, X, Xz, std, pca3, ae, vae = _setup()
    fams = df["family"].str.split("/").str[0].values
    regs = df["regime"].values

    # ===== 降維三面板(散點,無圖例) =====
    from sklearn.manifold import TSNE
    P_pca = pca3.transform(Xz)[:, :2]
    P_tsne = TSNE(2, perplexity=8, init="pca", random_state=0).fit_transform(Xz)
    P_iso, _ = manifold_proj(Xz)
    for nm, Pp in [("pca", P_pca), ("tsne", P_tsne), ("isomap", P_iso)]:
        fig, ax = plt.subplots(figsize=(4.3, 4.0))
        for i in range(len(df)):
            ax.scatter(Pp[i, 0], Pp[i, 1], color=FAM_C.get(fams[i], "#999"),
                       marker=REG_MARK.get(regs[i], "o"), s=70, edgecolor="white",
                       linewidth=0.4, alpha=0.9)
        ax.set_xticks([]); ax.set_yticks([]); _bare(ax)
        _save(fig, f"p_proj_{nm}.png")

    # ===== 訓練損失三面板 =====
    tr, va = city_group_split(df); Xtr, Xval = Xz[tr], Xz[va]
    d = Xz.shape[1]
    ae_a = L.build_ae(d, 2, (16, 8), seed=0)
    ha = L.train_ae(ae_a, Xtr, Xval, epochs=500, weight_decay=1e-4, denoise=0.05, seed=0)
    ae_b = L.build_ae(d, 2, (16, 8), bn=False, seed=0)
    hb = L.train_ae(ae_b, Xtr, Xval, epochs=500, weight_decay=0.0, denoise=0.0, seed=0)
    v_a = L.build_vae(d, 2, (16, 8), seed=0)
    hva = L.train_vae(v_a, Xtr, Xval, epochs=700, beta=1.0, anneal=0.4, seed=0)
    v_b = L.build_vae(d, 2, (16, 8), seed=0)
    hvb = L.train_vae(v_b, Xtr, Xval, epochs=700, beta=4.0, anneal=0.0, seed=0)

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(ha["train"], color=BLUE, label="_"); ax.plot(ha["val"], "--", color=BLUE)
    ax.plot(hb["train"], color=RED); ax.plot(hb["val"], "--", color=RED)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); _bare(ax)
    _save(fig, "p_loss_ae.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(hva["total"], color="#F2F5FA"); ax.plot(hva["recon"], color=BLUE); ax.plot(hva["kl"], color=ORANGE)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); _bare(ax)
    _save(fig, "p_loss_vae.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(hva["recon"], color=BLUE); ax.plot(hva["kl"], ":", color=BLUE)
    ax.plot(hvb["recon"], color=RED); ax.plot(hvb["kl"], ":", color=RED)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); _bare(ax)
    _save(fig, "p_loss_collapse.png")

    # ===== 重建:整體 bar + 逐特徵 =====
    res = {}
    for k in (2, 3, 4):
        m = L.PCAModel(k=k).fit(Xz); res[f"PCA k={k}"] = L.recon_mse(Xz, m.reconstruct(Xz))
    for lat in (2, 3):
        a = L.build_ae(d, lat, (16, 8), seed=0); L.train_ae(a, Xz, epochs=800, weight_decay=1e-5, huber=False, seed=0)
        res[f"AE {lat}"] = L.recon_mse(Xz, L.ae_reconstruct(a, Xz))
    for lat in (2, 3):
        vv = L.build_vae(d, lat, (16, 8), seed=0); L.train_vae(vv, Xz, epochs=900, beta=0.3, anneal=0.3, seed=0)
        res[f"VAE {lat}"] = L.recon_mse(Xz, L.vae_reconstruct(vv, Xz))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    names = list(res); vals = [res[n] for n in names]
    cols = [BLUE if "PCA" in n else ORANGE if "AE" in n else GREEN for n in names]
    ax.bar(range(len(names)), vals, color=cols)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, fontsize=8, ha="right")
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#F2F5FA")
    _bare(ax); _save(fig, "p_recon_bar.png")

    pca2 = L.PCAModel(k=2).fit(Xz)
    pf_pca = L.recon_mse(Xz, pca2.reconstruct(Xz), per_feature=True)
    a2 = L.build_ae(d, 2, (16, 8), seed=0); L.train_ae(a2, Xz, epochs=800, weight_decay=1e-5, huber=False, seed=0)
    pf_ae = L.recon_mse(Xz, L.ae_reconstruct(a2, Xz), per_feature=True)
    v2 = L.build_vae(d, 2, (16, 8), seed=0); L.train_vae(v2, Xz, epochs=900, beta=0.3, anneal=0.3, seed=0)
    pf_v = L.recon_mse(Xz, L.vae_reconstruct(v2, Xz), per_feature=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.2)); x = np.arange(len(L.FEATURES)); w = 0.27
    ax.bar(x - w, pf_pca, w, color=BLUE); ax.bar(x, pf_ae, w, color=ORANGE); ax.bar(x + w, pf_v, w, color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(L.FEATURES, rotation=40, fontsize=8, ha="right"); _bare(ax)
    _save(fig, "p_recon_feat.png")

    # ===== PCA 載荷熱圖 + silhouette =====
    load = pca3.components_
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    im = ax.imshow(load, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(len(L.FEATURES))); ax.set_xticklabels(L.FEATURES, rotation=40, fontsize=8, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(["PC1", "PC2", "PC3"])
    for i in range(3):
        for j in range(len(L.FEATURES)):
            ax.text(j, i, f"{load[i,j]:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="k" if abs(load[i, j]) < 0.45 else "white")
    _bare(ax); _save(fig, "p_loadings.png")

    from sklearn.metrics import silhouette_score
    sil_rows = []
    for nm, E in [("原始9维", Xz), ("PCA", P_pca), ("t-SNE", P_tsne), ("Isomap", P_iso)]:
        sil_rows.append((nm, silhouette_score(E, fams), silhouette_score(E, regs)))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    xs = np.arange(len(sil_rows)); w = 0.36
    ax.bar(xs - w/2, [r[1] for r in sil_rows], w, color=BLUE)
    ax.bar(xs + w/2, [r[2] for r in sil_rows], w, color=ORANGE)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in sil_rows], fontsize=8); _bare(ax)
    _save(fig, "p_silhouette.png")

    # ===== 體制簽名熱圖 + PCA 位移箭頭 =====
    REGS = ["baseline", "developer_led", "state_led", "resident_self_build", "shared"]
    idx = {(r["slug"], r["regime"]): i for i, r in df.iterrows()}
    S = []
    for reg in REGS[1:]:
        ds = [Xz[idx[(s, reg)]] - Xz[idx[(s, "baseline")]] for s in df["slug"].unique()
              if (s, reg) in idx and (s, "baseline") in idx]
        S.append(np.mean(ds, 0))
    S = np.array(S); vmax = np.abs(S).max()
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(L.FEATURES))); ax.set_xticklabels(L.FEATURES, rotation=40, fontsize=8, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(["开发商", "政府", "居民自建", "共享"], fontsize=9)
    for i in range(4):
        for j in range(len(L.FEATURES)):
            ax.text(j, i, f"{S[i,j]:+.1f}", ha="center", va="center", fontsize=6.5,
                    color="k" if abs(S[i, j]) < vmax*0.6 else "white")
    _bare(ax); _save(fig, "p_signature.png")

    Pp = pca3.transform(Xz)[:, :2]
    Pb = Pp[df.regime.values == "baseline"]; c0 = Pb.mean(0)
    colr = {"developer_led": RED, "state_led": BLUE, "resident_self_build": GREEN, "shared": PURPLE}
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.scatter(Pb[:, 0], Pb[:, 1], c="#8B949E", s=40, alpha=0.6)
    tips = [[Pb[:, 0].min(), Pb[:, 1].min()], [Pb[:, 0].max(), Pb[:, 1].max()]]
    for reg in REGS[1:]:
        dd = [Pp[idx[(s, reg)]] - Pp[idx[(s, "baseline")]] for s in df["slug"].unique()
              if (s, reg) in idx and (s, "baseline") in idx]
        dv = np.mean(dd, 0); tip = (float(c0[0]+dv[0]), float(c0[1]+dv[1])); tips.append(list(tip))
        ax.annotate("", xy=tip, xytext=(c0[0], c0[1]), arrowprops=dict(arrowstyle="-|>", color=colr[reg], lw=2.5))
    T = np.array(tips); mx = np.ptp(T[:, 0])*0.15+0.5; my = np.ptp(T[:, 1])*0.15+0.5
    ax.set_xlim(T[:, 0].min()-mx, T[:, 0].max()+mx); ax.set_ylim(T[:, 1].min()-my, T[:, 1].max()+my)
    ax.set_xticks([]); ax.set_yticks([]); _bare(ax); _save(fig, "p_regime_arrows.png")

    # ===== 生成:plaus/valid bars + 點雲 + interp =====
    from regime_latent import generate as G
    pca_ref, gen, gm = G.run(df, Xz, std)
    gens = ["PCA", "AE", "VAE"]; gc = [BLUE, ORANGE, GREEN]
    for key, fname in [("plaus", "p_gen_plaus.png"), ("valid", "p_gen_valid.png"), ("interp", "p_gen_interp.png")]:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        vals = [gm[g][key] for g in gens]
        ax.bar(range(3), vals, color=gc)
        ax.set_xticks(range(3)); ax.set_xticklabels(gens, fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}" if key != "valid" else f"{v:.0f}%", ha="center", va="bottom", fontsize=9, color="#F2F5FA")
        _bare(ax); _save(fig, fname)
    Pr = pca_ref.transform(Xz)
    for g, c, fn in zip(gens, gc, ["p_cloud_pca.png", "p_cloud_ae.png", "p_cloud_vae.png"]):
        Pg = pca_ref.transform(gen[g])
        fig, ax = plt.subplots(figsize=(3.6, 3.4))
        ax.scatter(Pg[:, 0], Pg[:, 1], s=10, alpha=0.35, color=c)
        ax.scatter(Pr[:, 0], Pr[:, 1], s=40, color="#F2F5FA", edgecolor="#12253C", linewidth=0.5)
        ax.set_xticks([]); ax.set_yticks([]); _bare(ax); _save(fig, fn)

    # ===== KMeans vs regime:ARI/sil bar + 混淆 =====
    from regime_latent.cluster import latent_spaces
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    spaces = latent_spaces(Xz)
    rows = []; best_lab = None; best_ari = -1; best_name = None
    for nm, E in spaces.items():
        km = KMeans(5, n_init=10, random_state=0).fit(E)
        ari = adjusted_rand_score(regs, km.labels_); sil = silhouette_score(E, km.labels_)
        rows.append((nm, ari, sil))
        if ari > best_ari: best_ari, best_lab, best_name = ari, km.labels_, nm
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(range(len(rows)), [r[1] for r in rows], color=BLUE)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=15, fontsize=8)
    for i, r in enumerate(rows): ax.text(i, r[1], f"{r[1]:.2f}", ha="center", va="bottom", fontsize=8, color="#F2F5FA")
    _bare(ax); _save(fig, "p_ari.png")
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(range(len(rows)), [r[2] for r in rows], color=ORANGE)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r[0] for r in rows], rotation=15, fontsize=8)
    for i, r in enumerate(rows): ax.text(i, r[2], f"{r[2]:.2f}", ha="center", va="bottom", fontsize=8, color="#F2F5FA")
    _bare(ax); _save(fig, "p_sil_cluster.png")
    ct = pd.crosstab(pd.Series(best_lab, name="k"), pd.Series(regs, name="r"))
    ct = ct.reindex(columns=[r for r in REGS if r in ct.columns])
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(["基线", "开发商", "政府", "居民自建", "共享"][:ct.shape[1]], rotation=25, fontsize=8, ha="right")
    ax.set_yticks(range(ct.shape[0])); ax.set_yticklabels([f"群{i}" for i in ct.index], fontsize=8)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=9,
                    color="white" if ct.values[i, j] > ct.values.max()/2 else "k")
    _bare(ax); _save(fig, "p_confusion.png")

    print("\n[DONE] 裸面板 →", OUT)


if __name__ == "__main__":
    main()
