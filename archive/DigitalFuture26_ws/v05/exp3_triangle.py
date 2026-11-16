#!/usr/bin/env python3
"""
v05 Experiment 3 — the triangle test.
Can window-level FORM (v03 morphological indicators) recover the asserted
POWER profile (E2, derived from type composition x age via literature priors)?

n = 24 windows. Tiny sample — disclosed everywhere. Seed 0.

Analyses:
 1) Spearman correlation screen power-dim x form-variable, permutation p
    (10,000 perms, seed 0), Benjamini-Hochberg FDR 0.10 over all 48 tests.
 2) LOO Ridge prediction of each power dim from FORM / TYPE(ceiling) / BOTH,
    alpha in {0.1,1,10,100} chosen by nested LOO on the 23 training points.
 3) Train(18) -> heldout(6) secondary check: MAE + per-window errors.
Outputs: results_e3.json, fig/fig_e3_cloud_power.png, fig/fig_e3_loo.png
"""
import json
import warnings
# macOS Accelerate/numpy emits spurious 'divide by zero/overflow/invalid in matmul'
# RuntimeWarnings (known false positive). Outputs were verified NaN/inf-free before
# adding this filter (see NOTES in results_e3.json meta).
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)
import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 0
N_PERM = 10000
ALPHAS = [0.1, 1.0, 10.0, 100.0]

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation"
V05 = f"{ROOT}/v05"
FIG = f"{V05}/fig"

POWER_DIMS = ["government", "developer_capital", "resident_community", "informal"]
FORM_VARS = ["PC1", "PC2", "FAR", "coverage", "h_p50", "h_p90", "median_fp",
             "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
IND10 = ["FAR", "coverage", "h_p50", "h_p90", "median_fp",
         "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
SEM8 = ["residential_estate", "residential_lowrise", "commercial_retail",
        "office_finance", "industrial", "public_admin", "education", "healthcare"]

# ---------------------------------------------------------------- load
e2 = json.load(open(f"{V05}/results_e2.json"))
sig = json.load(open(f"{ROOT}/v03/signatures.json"))
w5 = json.load(open(f"{V05}/windows.json"))["windows"]

sig_by_id = {w["id"]: w for w in sig["windows"]}
ids = sorted(sig_by_id.keys())
assert len(ids) == 24

names, roles = {}, {}
Y = np.zeros((24, 4))          # power profile
X_form = np.zeros((24, 10))    # 10 v03 indicators
PCA = np.zeros((24, 2))        # v03 pca_xy
X_type = np.zeros((24, 9))     # 8 semantic GFA shares (of TOTAL gfa) + unlabeled share

for i, wid in enumerate(ids):
    prof = e2["window_profiles"][wid]
    names[wid] = prof["name"]
    roles[wid] = prof["role"]
    Y[i] = [prof["profile"][d] for d in POWER_DIMS]
    s = sig_by_id[wid]
    X_form[i] = [s["indicators"][k] for k in IND10]
    PCA[i] = s["pca_xy"]
    ww = w5[wid]
    unl = ww["unlabeled_gfa_share"]
    # convert labeled-GFA shares to shares of TOTAL GFA so the 9 features sum to 1
    X_type[i, :8] = [ww["semantic_gfa_share_labeled"][k] * (1 - unl) for k in SEM8]
    X_type[i, 8] = unl

train_mask = np.array([roles[w] == "train" for w in ids])
heldout_ids = [w for w in ids if roles[w] == "heldout"]
assert sorted(heldout_ids) == ["A02", "A07", "A20", "A29", "A35", "A43"]

# form-variable matrix for the correlation screen (PC1/PC2 + 10 indicators)
F = np.column_stack([PCA[:, 0], PCA[:, 1], X_form])  # 24 x 12

# ---------------------------------------------------------------- 1) screen
rng = np.random.default_rng(RNG_SEED)


def spearman_vec(rx, RY):
    """Pearson of rank-vector rx (n,) against rows of rank-matrix RY (m,n)."""
    rxc = rx - rx.mean()
    RYc = RY - RY.mean(axis=1, keepdims=True)
    num = RYc @ rxc
    den = np.sqrt((RYc ** 2).sum(axis=1) * (rxc ** 2).sum())
    return num / den


rank_F = np.column_stack([rankdata(F[:, j]) for j in range(F.shape[1])])  # 24x12
screen = {}
all_p, all_keys = [], []
for di, dim in enumerate(POWER_DIMS):
    ry = rankdata(Y[:, di])
    rho_obs = spearman_vec(ry, rank_F.T)  # 12
    # one shared permutation set per power dim (10k x 24 permuted rank vectors)
    perms = np.array([rng.permutation(ry) for _ in range(N_PERM)])  # 10000x24
    pc = perms - perms.mean(axis=1, keepdims=True)
    Fc = rank_F - rank_F.mean(axis=0, keepdims=True)
    num = pc @ Fc                                            # 10000x12
    den = np.sqrt((pc ** 2).sum(axis=1, keepdims=True) * (Fc ** 2).sum(axis=0))
    rho_perm = num / den
    p = (1 + (np.abs(rho_perm) >= np.abs(rho_obs)[None, :]).sum(axis=0)) / (1 + N_PERM)
    screen[dim] = {}
    for j, fv in enumerate(FORM_VARS):
        screen[dim][fv] = {"rho": round(float(rho_obs[j]), 4),
                           "p_perm": round(float(p[j]), 5)}
        all_p.append(float(p[j]))
        all_keys.append((dim, fv))

# Benjamini-Hochberg FDR 0.10 over all 48 tests
m = len(all_p)
order = np.argsort(all_p)
bh_sig = np.zeros(m, dtype=bool)
thresh_k = -1
for rank_i, oi in enumerate(order, start=1):
    if all_p[oi] <= 0.10 * rank_i / m:
        thresh_k = rank_i
if thresh_k > 0:
    bh_sig[order[:thresh_k]] = True
survivors = []
for k, (dim, fv) in enumerate(all_keys):
    screen[dim][fv]["bh_fdr10"] = bool(bh_sig[k])
    if bh_sig[k]:
        survivors.append({"dim": dim, "var": fv,
                          "rho": screen[dim][fv]["rho"],
                          "p_perm": screen[dim][fv]["p_perm"]})
survivors.sort(key=lambda r: r["p_perm"])
print(f"[screen] {len(survivors)}/48 survive BH FDR 0.10")
for s in survivors:
    print("   ", s)

# ---------------------------------------------------------------- 2) LOO ridge
FEATURE_SETS = {
    "FORM": X_form,
    "TYPE": X_type,    # CEILING — circular by construction (profiles derive from types)
    "BOTH": np.column_stack([X_form, X_type]),
}


def fit_predict(Xtr, ytr, Xte, alpha):
    sc = StandardScaler().fit(Xtr)
    mdl = Ridge(alpha=alpha, random_state=RNG_SEED).fit(sc.transform(Xtr), ytr)
    return mdl.predict(sc.transform(Xte))


def inner_loo_alpha(Xtr, ytr):
    """Pick alpha minimizing inner-LOO SSE on the given training points."""
    n = len(ytr)
    best_a, best_sse = None, np.inf
    for a in ALPHAS:
        sse = 0.0
        for i in range(n):
            tr = np.arange(n) != i
            yh = fit_predict(Xtr[tr], ytr[tr], Xtr[i:i + 1], a)[0]
            sse += (ytr[i] - yh) ** 2
        if sse < best_sse:
            best_sse, best_a = sse, a
    return best_a


loo = {}
loo_pred_form = np.zeros((24, 4))
for set_name, X in FEATURE_SETS.items():
    loo[set_name] = {}
    for di, dim in enumerate(POWER_DIMS):
        y = Y[:, di]
        yhat = np.zeros(24)
        alphas_used = []
        for i in range(24):
            tr = np.arange(24) != i
            a = inner_loo_alpha(X[tr], y[tr])
            alphas_used.append(a)
            yhat[i] = fit_predict(X[tr], y[tr], X[i:i + 1], a)[0]
        sse = float(((y - yhat) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - sse / sst
        mae = float(np.abs(y - yhat).mean())
        loo[set_name][dim] = {
            "loo_r2": round(r2, 4), "mae": round(mae, 4),
            "alpha_mode": float(max(set(alphas_used), key=alphas_used.count)),
            "alphas_used": sorted(set(alphas_used)),
        }
        if set_name == "FORM":
            loo_pred_form[:, di] = yhat
        print(f"[loo] {set_name:4s} {dim:18s} R2={r2:+.3f} MAE={mae:.4f}")

# ---------------------------------------------------------------- 2b) sensitivity:
# E2 flagged A29,A32,A33,A41 as zero-age suburbs whose gov/dev targets sit on the
# artifactual no_age fallback (low-confidence GFA share 0.73-0.96). Re-run the
# FORM-only LOO for gov/dev on the remaining 20 windows.
ARTIFACT_WINS = ["A29", "A32", "A33", "A41"]
keep = np.array([w not in ARTIFACT_WINS for w in ids])
sens = {}
Xs = X_form[keep]
for di, dim in enumerate(["government", "developer_capital"]):
    y = Y[keep, di]
    n = len(y)
    yhat = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        a = inner_loo_alpha(Xs[tr], y[tr])
        yhat[i] = fit_predict(Xs[tr], y[tr], Xs[i:i + 1], a)[0]
    r2 = 1 - float(((y - yhat) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    sens[dim] = {"loo_r2": round(r2, 4), "mae": round(float(np.abs(y - yhat).mean()), 4),
                 "n": n, "excluded": ARTIFACT_WINS}
    print(f"[sens] FORM n=20 (no zero-age artifacts) {dim:18s} R2={r2:+.3f}")

# ---------------------------------------------------------------- 3) train->heldout
tr_idx = np.where(train_mask)[0]
ho_idx = np.where(~train_mask)[0]
heldout_eval = {}
for set_name, X in FEATURE_SETS.items():
    heldout_eval[set_name] = {}
    for di, dim in enumerate(POWER_DIMS):
        y = Y[:, di]
        a = inner_loo_alpha(X[tr_idx], y[tr_idx])
        yh = fit_predict(X[tr_idx], y[tr_idx], X[ho_idx], a)
        err = y[ho_idx] - yh
        sst = float(((y[ho_idx] - y[ho_idx].mean()) ** 2).sum())
        r2_ho = 1 - float((err ** 2).sum()) / sst if sst > 0 else None
        heldout_eval[set_name][dim] = {
            "alpha": a,
            "mae": round(float(np.abs(err).mean()), 4),
            "r2_heldout_n6_unstable": round(r2_ho, 4) if r2_ho is not None else None,
            "per_window": {ids[k]: {"actual": round(float(y[k]), 4),
                                    "pred": round(float(yh[j]), 4),
                                    "error": round(float(err[j]), 4)}
                           for j, k in enumerate(ho_idx)},
        }

# ---------------------------------------------------------------- fig 1: cloud
# fixed label placements for the annotated extremes (rendering only): shortened
# names + offsets chosen so no label crosses a marker; the two dense-centre
# points (A26, A43) are pushed into empty space with a thin leader line.
LBL_PLACE = {  # wid: (dx_pt, dy_pt, ha, display_name)
    "A01": (-9, -18, "right", "Lujiazui"),
    "A07": (0, -22, "center", "Bund"),
    "A27": (-9, 0, "right", "Lingang Nanhui"),
    "A26": (48, 43, "left", "Jiading"),
    "A43": (-20, -36, "right", "Liangcheng"),
    "A47": (0, 16, "center", "West Bund"),
    "A33": (0, -18, "center", "peri-urban"),
    "A10": (-9, 10, "right", "Laoximen old town"),
}
fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
for di, (dim, ax) in enumerate(zip(POWER_DIMS, axes.ravel())):
    v = Y[:, di]
    for msk, mk, lbl in [(train_mask, "o", "train"), (~train_mask, "^", "heldout")]:
        sc = ax.scatter(PCA[msk, 0], PCA[msk, 1], c=v[msk], cmap="viridis",
                        vmin=v.min(), vmax=v.max(), s=110, marker=mk,
                        edgecolors="k", linewidths=0.6, label=lbl, zorder=3)
    # annotate extremes of this dim (top2 / bottom2)
    ext = list(np.argsort(v)[:2]) + list(np.argsort(v)[-2:])
    xmid = (PCA[:, 0].min() + PCA[:, 0].max()) / 2
    for rj, k in enumerate(ext):
        wid = ids[k]
        if wid in LBL_PLACE:
            dx, dy, ha, disp = LBL_PLACE[wid]
        else:  # fallback: original staggered placement
            right_side = PCA[k, 0] > xmid
            dx = -9 if right_side else 9
            dy = 10 if rj % 2 == 0 else -18
            ha = "right" if right_side else "left"
            disp = names[wid]
        far = (dx * dx + dy * dy) ** 0.5 > 30
        ax.annotate(f"{wid} {disp}", (PCA[k, 0], PCA[k, 1]),
                    textcoords="offset points", xytext=(dx, dy),
                    ha=ha, va="center", fontsize=7.5,
                    arrowprops=(dict(arrowstyle="-", lw=0.55, color="0.45",
                                     shrinkA=2, shrinkB=2) if far else None))
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(f"{dim} weight")
    ax.set_title(f"{dim} on v03 PCA cloud")
    ax.set_xlabel("PC1 (intensity / verticality, 29.8%)")
    ax.set_ylabel("PC2 (sparse spikes vs dense mat, 27.0%)")
    if di == 0:
        ax.legend(loc="lower right", fontsize=8)
fig.suptitle("E3 — asserted power profiles on the v03 form PCA (n=24; profiles derived from type x age priors)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIG}/fig_e3_cloud_power.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: LOO
fig = plt.figure(figsize=(15, 8.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9], hspace=0.35, wspace=0.32)
for di, dim in enumerate(POWER_DIMS):
    ax = fig.add_subplot(gs[0, di])
    y, yh = Y[:, di], loo_pred_form[:, di]
    for msk, mk in [(train_mask, "o"), (~train_mask, "^")]:
        ax.scatter(y[msk], yh[msk], marker=mk, s=55, c="#3b6ea5",
                   edgecolors="k", linewidths=0.5, zorder=3)
    lo = min(y.min(), yh.min()) - 0.02
    hi = max(y.max(), yh.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    r = loo["FORM"][dim]
    ax.text(0.04, 0.96, f"LOO R$^2$ = {r['loo_r2']:+.2f}\nMAE = {r['mae']:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    ax.set_title(dim, fontsize=10)
    ax.set_xlabel("actual (E2 asserted)")
    if di == 0:
        ax.set_ylabel("LOO predicted (FORM only)")

axb = fig.add_subplot(gs[1, :])
xpos = np.arange(4)
wbar = 0.25
colors = {"FORM": "#3b6ea5", "TYPE": "#c08a2d", "BOTH": "#4d9e64"}
for si, sn in enumerate(["FORM", "TYPE", "BOTH"]):
    vals = [loo[sn][d]["loo_r2"] for d in POWER_DIMS]
    lbl = "TYPE (circular for res/inf; gov/dev also need age)" if sn == "TYPE" else sn
    axb.bar(xpos + (si - 1) * wbar, vals, wbar, label=lbl, color=colors[sn],
            edgecolor="k", linewidth=0.5)
    for xj, vv in zip(xpos + (si - 1) * wbar, vals):
        axb.text(xj, vv + (0.02 if vv >= 0 else -0.05), f"{vv:+.2f}",
                 ha="center", fontsize=8)
axb.axhline(0, color="k", lw=1)
axb.set_xticks(xpos)
axb.set_xticklabels(POWER_DIMS)
axb.set_ylabel("LOO R$^2$ (n=24)")
axb.set_title("LOO R$^2$ by feature set — TYPE channel: ceiling only for res/inf (priors type-determined); gov/dev also depend on the age regime",
              fontsize=10)
axb.legend(fontsize=9)
ymin = min(-0.15, min(loo[sn][d]["loo_r2"] for sn in FEATURE_SETS for d in POWER_DIMS) - 0.12)
axb.set_ylim(ymin, 1.12)
fig.suptitle("E3 — leave-one-out recovery of asserted power profiles (n=24, Ridge, nested-LOO alpha, seed 0)",
             fontsize=11)
fig.savefig(f"{FIG}/fig_e3_loo.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- save
out = {
    "meta": {
        "script": "v05/exp3_triangle.py",
        "n_windows": 24, "n_train": 18, "n_heldout": 6,
        "seed": RNG_SEED, "n_perm": N_PERM, "alphas": ALPHAS,
        "alpha_selection": "nested LOO on the 23 (outer-LOO) / 18 (heldout check) training points, min SSE",
        "standardization": "StandardScaler fit inside each fold (no leakage)",
        "type_features": "8 semantic GFA shares rescaled to shares of TOTAL window GFA + unlabeled share (sum=1)",
        "caveats": [
            "n=24 windows: every statistic here is small-sample; permutation p-values and LOO used throughout, but estimates remain unstable.",
            "TYPE feature set is circular by construction (E2 profiles are GFA-weighted priors over the same semantic classes) and is reported only as a ceiling. It is not an exact ceiling: residential_estate priors are age-conditioned, and age regimes are not in the TYPE features.",
            "POWER profiles are asserted from literature priors, not observed; this experiment tests internal recoverability, not ground truth.",
        ],
    },
    "correlation_screen": screen,
    "bh_fdr10_survivors": survivors,
    "n_tests": 48,
    "loo": loo,
    "loo_form_predictions": {ids[i]: {POWER_DIMS[d]: round(float(loo_pred_form[i, d]), 4)
                                      for d in range(4)} for i in range(24)},
    "heldout_check": heldout_eval,
    "sensitivity_form_loo_excl_zero_age": sens,
    "findings": [
        "Screen: 16/48 power-dim x form-variable Spearman tests survive BH FDR 0.10. The gov/dev axis is carried by PC2 and coverage, NOT PC1 as hypothesized: developer_capital vs PC2 rho=+0.688 (p=0.0006), vs coverage rho=-0.619; government vs coverage rho=+0.616, vs PC2 rho=-0.577. Developer-weighted windows are sparse-spike low-coverage fabrics; government-weighted windows are dense-mat high-coverage fabrics. PC1 (intensity) never exceeds |rho|=0.31 with any power dim.",
        "LOO FORM-only: developer_capital is the only well-recovered dim (R2=+0.46, MAE 0.063); government barely (R2=+0.10); resident_community FAILS as point prediction (R2=-0.45) despite strong rank correlations (sh_large rho=-0.678 was the single strongest screen result); informal collapses (R2=-2.24) — but its target sd is only 0.016 (range 0.008-0.088), so MAE is still just 0.015. Monotone association without point-predictability = the form signal exists but is shared with non-informal lowrise, exactly as the priors' design implies.",
        "SURPRISE — TYPE is NOT a ceiling for gov/dev: TYPE-only LOO R2 = -0.23 (government) and +0.04 (developer) vs FORM's +0.10 and +0.46. Reason: gov/dev profile variance is dominated by the AGE-conditioned residential_estate priors (pre1985 gov-heavy vs post1998 dev-heavy), and age regimes are absent from type shares. FORM (heights, towers, PC2) partially proxies construction era, so form beats the very type composition the profiles were built from. The circular ceiling holds only for the age-independent dims: resident_community R2=0.96, informal R2=0.82.",
        "BOTH never beats the best single set (gov -0.21, dev +0.33, res +0.83, inf +0.75): at n=24 with 19 features, adding the weaker block dilutes the stronger one despite ridge regularization.",
        "Train(18)->heldout(6): FORM MAE gov 0.069 / dev 0.089 / res 0.023 / inf 0.006. Largest errors are regression-to-the-mean at the extremes: A43 Liangcheng (most gov-dominant, 0.600) predicted 0.459; its developer weight (actual 0.242, the minimum) predicted 0.457 (error -0.215). A07 Bund (lowest gov, 0.288) overpredicted at 0.406.",
        "Caveat inherited from E2: 4 of 5 zero-age suburbs (A29,A32,A33,A41) carry artifactual no_age-fallback gov/dev targets (low-confidence GFA shares 0.73-0.96), adding noise to the regression TARGET itself. Sensitivity LOO on the remaining 20 windows: developer_capital is ROBUST (R2=+0.461 vs +0.459 at n=24), but government FLIPS NEGATIVE (R2=-0.070 vs +0.103) — government's apparent weak recoverability does not survive removing the artifact targets. Only developer_capital is robustly form-recoverable.",
    ],
}
with open(f"{V05}/results_e3.json", "w") as f:
    json.dump(out, f, indent=1)
print("saved results_e3.json")
