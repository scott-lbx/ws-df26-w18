"""
analyze_latent.py — 潛在研究:利益相關者 ↔ 城市形態 的雙向映射
=================================================================
語料 = out/latent_research/features.csv(12 城 × 6 權力體制 = 72 樣本 × 9 形態特徵)。

研究問題(對應工作坊主題「雙向映射」):
  Q1 形態空間  : 72 種形態在低維潛在空間怎麼分佈?(PCA / AE / VAE 三種壓縮)
  Q2 power→form: 每種權力把形態往哪個方向推?(權力效應向量 + 效應量)
  Q3 form→power: 只看形態,能不能反推出是哪種權力造的?(跨城 LOSO 分類)
  Q4 可逆性     : 這個映射雙向一致嗎?(權力效應空間的分離度 + 反演準確率)

方法論核心 —— 兩個座標系:
  ‧ 形態空間 (absolute)    : 9 特徵 log+z-score。站點尺度(棟數/面積)會主導 → 城市各自成群。
  ‧ 權力效應空間 (relative): 每特徵對「該站現狀」取 log 比值。抵消站點尺度 → 只剩「權力做了什麼」。
    現狀 = 原點;每種權力 = 一個位移。→ 若權力在此空間跨城成群,就證明「權力留下站點無關的形態指紋」。

零 AI 推斷;所有結論都可由 features.csv 重算。seed 固定,結果可重現。
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (confusion_matrix, adjusted_rand_score,
                             silhouette_score, accuracy_score)
from sklearn.model_selection import LeaveOneGroupOut

import torch
import torch.nn as nn

SEED = 20260702
np.random.seed(SEED)
torch.manual_seed(SEED)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "out" / "latent_research"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FEATURES = ["n", "far", "coverage", "h_mean", "h_max", "h_cv", "grain", "slender", "concentration"]
LOG_FEATS = ["n", "far", "grain", "slender", "h_mean", "h_max"]   # 重尾/量級 → log1p
REGIME_ORDER = ["current", "developer_led", "state_led",
                "resident_self_build", "developer_scale_graded", "shared"]
REGIME_COLOR = {
    "current": "#7f7f7f", "developer_led": "#c0654a", "state_led": "#4a6fa5",
    "resident_self_build": "#5a9367", "developer_scale_graded": "#d1913c", "shared": "#8a6fb0",
}
EPS = 1e-9


# ----------------------------------------------------------------- 載入 + 特徵工程
def load() -> pd.DataFrame:
    df = pd.read_csv(OUT / "features.csv", encoding="utf-8-sig")
    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    return df.sort_values(["site", "regime"]).reset_index(drop=True)


def form_space(df: pd.DataFrame) -> np.ndarray:
    """形態空間:log1p 重尾特徵後全域 z-score。回傳 (72,9)。"""
    X = df[FEATURES].to_numpy(float).copy()
    for j, f in enumerate(FEATURES):
        if f in LOG_FEATS:
            X[:, j] = np.log1p(X[:, j])
    return StandardScaler().fit_transform(X)


def power_effect_space(df: pd.DataFrame) -> np.ndarray:
    """權力效應空間:每特徵對『該站現狀』取 log 比值 → 抵消站點尺度。current 列 ≈ 0 向量。回傳 (72,9)。"""
    R = np.zeros((len(df), len(FEATURES)), float)
    cur = {s: g[g["regime"] == "current"].iloc[0] for s, g in df.groupby("site")}
    for i, row in df.iterrows():
        base = cur[row["site"]]
        for j, f in enumerate(FEATURES):
            R[i, j] = np.log((row[f] + EPS) / (base[f] + EPS))
    return R


# ----------------------------------------------------------------- 潛在壓縮:AE / VAE
class AE(nn.Module):
    def __init__(self, d_in=9, d_lat=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU(), nn.Linear(16, d_lat))
        self.dec = nn.Sequential(nn.Linear(d_lat, 16), nn.ReLU(), nn.Linear(16, d_in))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


class VAE(nn.Module):
    def __init__(self, d_in=9, d_lat=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 16), nn.ReLU())
        self.mu = nn.Linear(16, d_lat)
        self.lv = nn.Linear(16, d_lat)
        self.dec = nn.Sequential(nn.Linear(d_lat, 16), nn.ReLU(), nn.Linear(16, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv


def train_ae(X, epochs=4000, lr=5e-3, seed=SEED):
    torch.manual_seed(seed)
    m = AE(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        xh, _ = m(xt)
        loss = ((xh - xt) ** 2).mean()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xh, z = m(xt)
        recon = float(((xh - xt) ** 2).mean())
    return z.detach().numpy(), recon


def train_vae(X, epochs=4000, lr=5e-3, beta=0.5, seed=SEED):
    torch.manual_seed(seed)
    m = VAE(); opt = torch.optim.Adam(m.parameters(), lr=lr)
    xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        xh, mu, lv = m(xt)
        rec = ((xh - xt) ** 2).mean()
        kld = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
        (rec + beta * kld).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        xh, mu, lv = m(xt)
        recon = float(((xh - xt) ** 2).mean())
    return mu.detach().numpy(), recon


# ----------------------------------------------------------------- Q2 power → form 效應表
def regime_effects(df: pd.DataFrame) -> pd.DataFrame:
    """每 (regime, feature) 的跨城平均 log 比值 + 效應量(Cohen's d,regime vs current 配對於同站)。"""
    rows = []
    piv = {(r["site"], str(r["regime"])): r for _, r in df.iterrows()}
    sites = sorted(df["site"].unique())
    for reg in REGIME_ORDER:
        if reg == "current":
            continue
        for f in FEATURES:
            ratios, deltas = [], []
            for s in sites:
                cur = piv[(s, "current")][f]
                val = piv[(s, reg)][f]
                ratios.append(np.log((val + EPS) / (cur + EPS)))
                deltas.append(val - cur)
            ratios = np.array(ratios); deltas = np.array(deltas)
            d = float(deltas.mean() / (deltas.std() + EPS))          # 配對效應量
            rows.append({"regime": reg, "feature": f,
                         "mean_log_ratio": float(ratios.mean()),
                         "mean_pct_change": float((np.exp(ratios.mean()) - 1) * 100),
                         "cohens_d": d, "abs_d": abs(d)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- Q3 form → power 分類(跨城 LOSO)
def loso_classify(R: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """留一城交叉驗證:train 11 城、test 第 12 城。回傳 (acc, y_true, y_pred)。
    特徵 = 權力效應空間(站點無關)→ 測『權力形態指紋是否跨城通用』。"""
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    for tr, te in logo.split(R, y, groups):
        clf = LogisticRegression(max_iter=5000, C=1.0)
        clf.fit(R[tr], y[tr])
        y_pred.extend(clf.predict(R[te])); y_true.extend(y[te])
    return accuracy_score(y_true, y_pred), np.array(y_true), np.array(y_pred)


def permutation_test(R, y, groups, observed, n_perm=300):
    """置換檢定:洗牌體制標籤 n_perm 次,各自重算 LOSO 準確率 → 虛無分佈 + p 值。
    測 observed 準確率是否顯著高於『標籤與形態無關』的洗牌基線。"""
    rng = np.random.default_rng(SEED)
    null = np.empty(n_perm)
    for k in range(n_perm):
        yp = rng.permutation(y)
        acc, _, _ = loso_classify(R, yp, groups)
        null[k] = acc
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {"observed": float(observed), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)), "p_value": p,
            "n_perm": n_perm}, null


def plot_permutation(null, observed, p, path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(null, bins=24, color="#b8b8b8", edgecolor="white", label="shuffled-label null")
    ax.axvline(observed, color="#c0654a", lw=2.5, label=f"observed acc={observed:.2f}")
    ax.axvline(1 / 6, color="#4a6fa5", lw=1.5, ls="--", label="chance=0.17")
    ax.set_xlabel("LOSO accuracy"); ax.set_ylabel("count")
    ax.set_title(f"permutation test — form→power recoverability  (p={p:.4f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def nearest_centroid_invert(Z: np.ndarray, y: np.ndarray):
    """反演:在潛在空間用『最近體制質心』反推 power。回傳 acc(可逆性代理)。"""
    cent = {r: Z[y == r].mean(0) for r in REGIME_ORDER}
    order = list(cent.keys()); C = np.stack([cent[r] for r in order])
    pred = [order[int(np.argmin(((z - C) ** 2).sum(1)))] for z in Z]
    return accuracy_score(y, pred), pred


# ----------------------------------------------------------------- 圖
def scatter_2d(Z, y, sites, title, path, xlabel="dim 1", ylabel="dim 2"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for reg in REGIME_ORDER:
        m = y == reg
        ax.scatter(Z[m, 0], Z[m, 1], c=REGIME_COLOR[reg], label=reg,
                   s=70, edgecolors="white", linewidths=0.6, alpha=0.9)
        cx, cy = Z[m, 0].mean(), Z[m, 1].mean()
        ax.scatter([cx], [cy], c=REGIME_COLOR[reg], s=260, marker="*",
                   edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_title(title, fontsize=12); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_explained_var(pca, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ev = pca.explained_variance_ratio_
    ax.bar(range(1, len(ev) + 1), ev, color="#4a6fa5", alpha=0.85)
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), "o-", color="#c0654a", label="cumulative")
    ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    ax.set_title("PCA explained variance (form space)"); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_effect_heatmap(eff: pd.DataFrame, path):
    regs = [r for r in REGIME_ORDER if r != "current"]
    M = eff.pivot(index="regime", columns="feature", values="mean_log_ratio").reindex(index=regs, columns=FEATURES)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vmax = np.nanmax(np.abs(M.to_numpy()))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES, rotation=40, ha="right")
    ax.set_yticks(range(len(regs))); ax.set_yticklabels(regs)
    for i in range(len(regs)):
        for j in range(len(FEATURES)):
            v = M.to_numpy()[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                    color="black" if abs(v) < vmax * 0.6 else "white")
    ax.set_title("power → form : mean log-ratio vs current (by regime × feature)")
    fig.colorbar(im, ax=ax, label="log-ratio (Δ vs current)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_confusion(y_true, y_pred, acc, path):
    cm = confusion_matrix(y_true, y_pred, labels=REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(REGIME_ORDER, rotation=40, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(REGIME_ORDER)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xlabel("predicted power"); ax.set_ylabel("true power")
    ax.set_title(f"form → power (leave-one-city-out)  acc={acc:.2f}  (chance=0.17)")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_methods_compare(Zs: dict, y, path):
    fig, axes = plt.subplots(1, len(Zs), figsize=(5 * len(Zs), 4.6))
    if len(Zs) == 1:
        axes = [axes]
    for ax, (name, Z) in zip(axes, Zs.items()):
        for reg in REGIME_ORDER:
            m = y == reg
            ax.scatter(Z[m, 0], Z[m, 1], c=REGIME_COLOR[reg], s=45,
                       edgecolors="white", linewidths=0.5, alpha=0.9, label=reg)
        ax.set_title(name); ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0.15)
    axes[-1].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.suptitle("power-effect space — three latent embeddings (regimes should cluster across cities)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------- 主流程
def main():
    df = load()
    y = df["regime"].astype(str).to_numpy()
    sites = df["site"].to_numpy()
    print(f"== 載入 {df.shape[0]} 樣本 × {len(FEATURES)} 特徵;{df['site'].nunique()} 城 × {len(REGIME_ORDER)} 體制 ==")

    Xf = form_space(df)            # 形態空間(絕對)
    Xr = power_effect_space(df)    # 權力效應空間(站點相對)

    results = {"n_samples": int(df.shape[0]), "n_sites": int(df["site"].nunique()),
               "n_regimes": len(REGIME_ORDER), "features": FEATURES, "seed": SEED}

    # ---- Q1 形態空間 PCA(絕對)
    pca_f = PCA().fit(Xf)
    Zf = pca_f.transform(Xf)[:, :2]
    results["form_space_pca_evr"] = [float(v) for v in pca_f.explained_variance_ratio_]
    plot_explained_var(pca_f, FIG / "explained_variance.png")
    scatter_2d(Zf, y, sites, "form space (absolute) — PCA  ·  cities separate by scale",
               FIG / "pca_form_space.png",
               xlabel=f"PC1 ({pca_f.explained_variance_ratio_[0]*100:.0f}%)",
               ylabel=f"PC2 ({pca_f.explained_variance_ratio_[1]*100:.0f}%)")

    # ---- Q1/Q4 權力效應空間:PCA + AE + VAE(站點相對)
    pca_r = PCA().fit(Xr)
    Zr_pca = pca_r.transform(Xr)[:, :2]
    Zr_ae, rec_ae = train_ae(Xr)
    Zr_vae, rec_vae = train_vae(Xr)
    results["power_effect_pca_evr"] = [float(v) for v in pca_r.explained_variance_ratio_]
    results["ae_recon_mse"] = rec_ae
    results["vae_recon_mse"] = rec_vae

    # 三法潛在的體制分離度(silhouette;current 在原點會壓低,故也算「排除 current」版)
    def sil(Z):
        try:
            s_all = float(silhouette_score(Z, y))
        except Exception:
            s_all = float("nan")
        mask = y != "current"
        try:
            s_noc = float(silhouette_score(Z[mask], y[mask]))
        except Exception:
            s_noc = float("nan")
        return s_all, s_noc
    results["silhouette"] = {
        "pca": sil(Zr_pca), "ae": sil(Zr_ae), "vae": sil(Zr_vae),
        "note": "(all_regimes, excluding_current)"}

    scatter_2d(Zr_pca, y, sites, "power-effect space — PCA  ·  regimes cluster across cities",
               FIG / "pca_power_effect.png",
               xlabel=f"PC1 ({pca_r.explained_variance_ratio_[0]*100:.0f}%)",
               ylabel=f"PC2 ({pca_r.explained_variance_ratio_[1]*100:.0f}%)")
    scatter_2d(Zr_vae, y, sites, "power-effect space — VAE latent", FIG / "vae_power_effect.png",
               xlabel="z1", ylabel="z2")
    plot_methods_compare({"PCA": Zr_pca, "AE": Zr_ae, "VAE": Zr_vae}, y, FIG / "latent_methods_compare.png")

    # ---- Q2 power → form 效應表
    eff = regime_effects(df)
    eff.to_csv(OUT / "regime_effects.csv", index=False, encoding="utf-8-sig")
    plot_effect_heatmap(eff, FIG / "regime_effects_heatmap.png")
    signatures = {}
    for reg in [r for r in REGIME_ORDER if r != "current"]:
        sub = eff[eff["regime"] == reg].sort_values("abs_d", ascending=False)
        top = sub.iloc[0]
        signatures[reg] = {"feature": top["feature"], "cohens_d": round(float(top["cohens_d"]), 2),
                           "pct_change": round(float(top["mean_pct_change"]), 1)}
    results["regime_signatures"] = signatures

    # ---- Q3 form → power 分類(跨城 LOSO,權力效應空間)
    acc, yt, yp = loso_classify(Xr, y, sites)
    plot_confusion(yt, yp, acc, FIG / "confusion_loso.png")
    per_regime = {}
    for reg in REGIME_ORDER:
        m = yt == reg
        per_regime[reg] = round(float((yp[m] == reg).mean()), 3) if m.any() else None
    perm, null = permutation_test(Xr, y, sites, acc, n_perm=300)
    plot_permutation(null, acc, perm["p_value"], FIG / "permutation_test.png")
    results["form_to_power"] = {"loso_accuracy": round(float(acc), 3),
                                "chance": round(1 / 6, 3), "per_regime_recall": per_regime,
                                "permutation": {k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in perm.items()}}

    # ---- Q4 可逆性:效應空間分離(KMeans ARI)+ 最近質心反演
    km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit(Zr_pca)
    ari = float(adjusted_rand_score(y, km.labels_))
    inv_acc, _ = nearest_centroid_invert(Zr_pca, y)
    results["invertibility"] = {"kmeans_ari_vs_regime": round(ari, 3),
                                "nearest_centroid_accuracy": round(float(inv_acc), 3)}

    # ---- 落盤潛在座標 + 結果
    pd.DataFrame({"site": sites, "regime": y,
                  "pca_form_1": Zf[:, 0], "pca_form_2": Zf[:, 1],
                  "pca_pe_1": Zr_pca[:, 0], "pca_pe_2": Zr_pca[:, 1],
                  "ae_1": Zr_ae[:, 0], "ae_2": Zr_ae[:, 1],
                  "vae_1": Zr_vae[:, 0], "vae_2": Zr_vae[:, 1]}
                 ).to_csv(OUT / "latent_coords.csv", index=False, encoding="utf-8-sig")
    json.dump(results, open(OUT / "results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # ---- 摘要列印
    print("\n-- Q1 潛在壓縮(權力效應空間 silhouette,越高越分離)--")
    for k, v in results["silhouette"].items():
        if k != "note":
            a, b = v
            print(f"   {k.upper():4s} all={a:+.3f}  ex-current={b:+.3f}")
    print(f"   PCA 前二主成分累積解釋 {sum(results['power_effect_pca_evr'][:2])*100:.0f}%  ·  VAE recon MSE={rec_vae:.3f}")
    print("\n-- Q2 power→form 每體制最強形態指紋 --")
    for reg, s in signatures.items():
        print(f"   {reg:22s} {s['feature']:14s} d={s['cohens_d']:+.2f}  ({s['pct_change']:+.0f}%)")
    print("\n-- Q3 form→power(跨城 LOSO 分類)--")
    print(f"   accuracy = {acc:.3f}  (chance 0.167)   per-regime recall: "
          + ", ".join(f"{k[:9]}={v}" for k, v in per_regime.items()))
    print(f"   置換檢定:虛無均值={perm['null_mean']:.3f}  p95={perm['null_p95']:.3f}  p={perm['p_value']:.4f}")
    print("\n-- Q4 可逆性 --")
    print(f"   KMeans ARI = {ari:.3f}   nearest-centroid inversion acc = {inv_acc:.3f}")
    print(f"\n== 輸出 → {OUT.relative_to(PROJECT)}  (results.json, regime_effects.csv, latent_coords.csv, figures/*.png) ==")


if __name__ == "__main__":
    main()
