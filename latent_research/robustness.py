"""
robustness.py — 穩健性補遺:把「有效獨立單位=6 配方」的誠實折進統計
=================================================================
回應完備性評審的 top gap:主分析(analyze_latent.py)的 LOSO 0.972、Cohen's d、
silhouette、ARI、反演準確率全是乾淨點估計,未給任何不確定範圍。本模組補上:

  1 城市層 bootstrap  : 以「城市」(非 72 偽樣本)為重抽單位,B 次重抽 → silhouette/ARI/
                        最近質心準確率的 95% 百分位 CI。這才是誠實的重抽單位。
  2 LOSO 逐城 fold    : 12 個留一城 fold 各自準確率的分佈(min/median/range)。
  3 leave-one-recipe  : 逐一拿掉 5 個非 current 體制,重算分離度 → 無單一配方主導。
  4 多 seed AE/VAE    : 5 個 seed 重訓,silhouette mean±sd → 潛在對初始化的穩健度。
  5 逐體制離散度      : 各體制 12 城點在效應空間的散度 → 驗「developer_scale_graded 變異較大」。
  6 參數洩漏量化      : 精確為 0 的 log 比值格數(算子不觸碰該自由度)+ h_max 釘在算子 cap 的數 →
                        把評審的「特徵在讀算子參數」疑慮給出實際數字。

輸出 out/latent_research/robustness.json + figures/robustness.png。可重現(seed 固定)。
"""
from __future__ import annotations
import sys
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut

import analyze_latent as A   # 重用 load / 空間 / AE / VAE / 常數

SEED = A.SEED
OUT = A.OUT
FIG = A.FIG
REGIME_ORDER = A.REGIME_ORDER
NONCUR = [r for r in REGIME_ORDER if r != "current"]
CAPS = {480.0, 600.0, 200.0}   # densify cap_m / concentrate cap_m / open_ground cap_m


def _pca2(X):
    return PCA(n_components=2).fit_transform(X)


def _sil_excur(Z, y):
    m = y != "current"
    try:
        return float(silhouette_score(Z[m], y[m]))
    except Exception:
        return float("nan")


def _ci(vals):
    a = np.array(vals, float); a = a[~np.isnan(a)]
    return {"lo": float(np.percentile(a, 2.5)), "med": float(np.median(a)),
            "hi": float(np.percentile(a, 97.5)), "mean": float(a.mean())}


# --------------------------------------------------------- 1 城市層 bootstrap
def bootstrap_cities(Xr, y, sites, B=600):
    uniq = sorted(set(sites))
    rng = np.random.default_rng(SEED)
    sil, ari, nc = [], [], []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(sites == c)[0] for c in pick])
        Xb, yb = Xr[idx], y[idx]
        Z = _pca2(Xb)
        sil.append(_sil_excur(Z, yb))
        try:
            km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Z)
            ari.append(adjusted_rand_score(yb, km.labels_))
        except Exception:
            ari.append(np.nan)
        nc.append(A.nearest_centroid_invert(Z, yb)[0])
    return {"B": B, "resample_unit": "city (12)",
            "silhouette_ex_current": _ci(sil), "kmeans_ari": _ci(ari),
            "nearest_centroid_acc": _ci(nc)}


# --------------------------------------------------------- 2 LOSO 逐城 fold
def loso_folds(Xr, y, sites):
    logo = LeaveOneGroupOut()
    accs = {}
    for tr, te in logo.split(Xr, y, sites):
        clf = LogisticRegression(max_iter=5000, C=1.0).fit(Xr[tr], y[tr])
        accs[str(sites[te][0])] = float(accuracy_score(y[te], clf.predict(Xr[te])))
    v = np.array(list(accs.values()))
    return {"per_city": accs, "mean": float(v.mean()), "min": float(v.min()),
            "n_cities_below_1": int((v < 1.0).sum())}


# --------------------------------------------------------- 3 leave-one-recipe
def leave_one_recipe(Xr, y):
    out = {}
    for drop in NONCUR:
        m = y != drop
        out[drop] = _sil_excur(_pca2(Xr[m]), y[m])
    v = np.array(list(out.values()))
    return {"per_dropped_recipe": out, "min": float(v.min()), "max": float(v.max()),
            "spread": float(v.max() - v.min())}


# --------------------------------------------------------- 4 多 seed AE/VAE
def multiseed_latents(Xr, y, seeds=(1, 7, 42, 123, 2026)):
    ae, vae = [], []
    for s in seeds:
        Zae, _ = A.train_ae(Xr, seed=s); ae.append(_sil_excur(Zae, y))
        Zva, _ = A.train_vae(Xr, seed=s); vae.append(_sil_excur(Zva, y))
    return {"seeds": list(seeds),
            "ae_sil_ex_current": {"mean": float(np.mean(ae)), "sd": float(np.std(ae)),
                                  "vals": [round(x, 3) for x in ae]},
            "vae_sil_ex_current": {"mean": float(np.mean(vae)), "sd": float(np.std(vae)),
                                   "vals": [round(x, 3) for x in vae]}}


# --------------------------------------------------------- 5 逐體制離散度
def regime_dispersion(Xr, y):
    Z = _pca2(Xr)
    out = {}
    for r in REGIME_ORDER:
        P = Z[y == r]
        c = P.mean(0)
        out[r] = float(np.sqrt(((P - c) ** 2).sum(1)).mean())   # 平均到質心距離
    return out


# --------------------------------------------------------- 6 參數洩漏量化
def parameter_leakage(df, Xr, y):
    non = y != "current"
    zero_cells = int(np.sum(np.abs(Xr[non]) < 1e-9))
    total_cells = int(Xr[non].size)
    hmax = df["h_max"].to_numpy(float)
    at_cap = int(sum(any(abs(h - c) < 0.5 for c in CAPS) for h in hmax))
    # 逐體制:非 current 列中,精確 0 的 log 比值特徵數(算子不觸碰的自由度)
    per_regime_zero = {}
    for r in NONCUR:
        m = y == r
        per_regime_zero[r] = int(np.sum(np.abs(Xr[m]) < 1e-9))
    return {"exact_zero_logratio_cells": zero_cells, "of_total_noncurrent_cells": total_cells,
            "exact_zero_fraction": round(zero_cells / total_cells, 3),
            "h_max_pinned_at_operator_cap": at_cap, "of_rows": int(len(hmax)),
            "per_regime_untouched_dof_count": per_regime_zero,
            "note": "精確 0 = 該算子不觸碰該特徵;h_max 釘 cap = 特徵在報參數而非形態。皆屬確定性洩漏。"}


# --------------------------------------------------------- 圖
def plot_robustness(boot, disp, ms, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # 左:bootstrap CI(森林圖)
    metrics = [("silhouette\n(ex-current)", boot["silhouette_ex_current"]),
               ("KMeans ARI", boot["kmeans_ari"]),
               ("nearest-centroid\nacc", boot["nearest_centroid_acc"])]
    ys = range(len(metrics))
    for i, (name, ci) in enumerate(metrics):
        ax[0].plot([ci["lo"], ci["hi"]], [i, i], color="#4a6fa5", lw=3, solid_capstyle="round")
        ax[0].plot(ci["med"], i, "o", color="#c0654a", ms=9)
    ax[0].set_yticks(list(ys)); ax[0].set_yticklabels([m[0] for m in metrics])
    ax[0].set_xlim(0, 1); ax[0].grid(alpha=0.2, axis="x")
    ax[0].set_title(f"city-level bootstrap 95% CI  (B={boot['B']}, unit=city)")
    # 右:逐體制效應空間離散度
    regs = [r for r in REGIME_ORDER]
    vals = [disp[r] for r in regs]
    cols = [A.REGIME_COLOR[r] for r in regs]
    ax[1].bar(range(len(regs)), vals, color=cols, alpha=0.9, edgecolor="white")
    ax[1].set_xticks(range(len(regs))); ax[1].set_xticklabels(regs, rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("mean dist to regime centroid")
    ax[1].set_title("per-regime dispersion in power-effect space")
    fig.suptitle(f"robustness — multi-seed AE sil={ms['ae_sil_ex_current']['mean']:.2f}"
                 f"±{ms['ae_sil_ex_current']['sd']:.2f} · VAE {ms['vae_sil_ex_current']['mean']:.2f}"
                 f"±{ms['vae_sil_ex_current']['sd']:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    df = A.load()
    y = df["regime"].astype(str).to_numpy()
    sites = df["site"].to_numpy()
    Xr = A.power_effect_space(df)

    print("== 穩健性補遺(誠實重抽單位 = 城市/配方)==")
    boot = bootstrap_cities(Xr, y, sites)
    folds = loso_folds(Xr, y, sites)
    lor = leave_one_recipe(Xr, y)
    ms = multiseed_latents(Xr, y)
    disp = regime_dispersion(Xr, y)
    leak = parameter_leakage(df, Xr, y)

    res = {"n_cities": int(df["site"].nunique()), "n_recipes": len(REGIME_ORDER),
           "effective_independent_units": len(REGIME_ORDER),
           "city_bootstrap": boot, "loso_folds": folds, "leave_one_recipe": lor,
           "multiseed_latent": ms, "regime_dispersion": disp, "parameter_leakage": leak}
    json.dump(res, open(OUT / "robustness.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    plot_robustness(boot, disp, ms, FIG / "robustness.png")

    b = boot
    print(f"  city bootstrap (B={b['B']}, unit=city):")
    print(f"    silhouette ex-current  95%CI [{b['silhouette_ex_current']['lo']:.2f}, {b['silhouette_ex_current']['hi']:.2f}]  med {b['silhouette_ex_current']['med']:.2f}")
    print(f"    KMeans ARI             95%CI [{b['kmeans_ari']['lo']:.2f}, {b['kmeans_ari']['hi']:.2f}]  med {b['kmeans_ari']['med']:.2f}")
    print(f"    nearest-centroid acc   95%CI [{b['nearest_centroid_acc']['lo']:.2f}, {b['nearest_centroid_acc']['hi']:.2f}]  med {b['nearest_centroid_acc']['med']:.2f}")
    print(f"  LOSO 逐城 fold: mean {folds['mean']:.3f}  min {folds['min']:.3f}  ({folds['n_cities_below_1']}/12 城 <1.0)")
    print(f"  leave-one-recipe silhouette spread: [{lor['min']:.2f}, {lor['max']:.2f}]  Δ={lor['spread']:.2f}(無單一配方主導)")
    print(f"  multi-seed: AE sil {ms['ae_sil_ex_current']['mean']:.2f}±{ms['ae_sil_ex_current']['sd']:.2f}  VAE {ms['vae_sil_ex_current']['mean']:.2f}±{ms['vae_sil_ex_current']['sd']:.2f}")
    print("  逐體制離散度:", {k: round(v, 2) for k, v in disp.items()})
    print(f"  參數洩漏:精確0比值格 {leak['exact_zero_logratio_cells']}/{leak['of_total_noncurrent_cells']}"
          f"({leak['exact_zero_fraction']*100:.0f}%);h_max 釘 cap {leak['h_max_pinned_at_operator_cap']}/{leak['of_rows']} 列")
    print(f"== 輸出 → {(OUT / 'robustness.json').relative_to(A.PROJECT)} + figures/robustness.png ==")


if __name__ == "__main__":
    main()
