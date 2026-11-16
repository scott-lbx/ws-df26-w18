#!/usr/bin/env python3
"""
v05 Experiment 1 — building-level semantic recognition from form (and form+context).

Question: can a RandomForest recognize the fusion-derived semantic type of a building
from FORM alone / FORM+CTX, and does the skill TRANSFER across windows?
Two CV designs: StratifiedKFold(5) (comparable to old C report, leaks spatial context)
vs GroupKFold by window = leave-one-window-out (the honest spatial-transfer test).

Deliberately EXCLUDED features: anything POI/EULUC/AOI-derived — those sources produced
the fused labels (circularity). token_v03/v04 also excluded (tokens were clustered on
form+family composition of cells, partially label-adjacent via family).

Outputs: results_e1.json, fig/fig_e1_confusion.png, fig/fig_e1_transfer.png.
Seed 0 everywhere. Rerunnable.
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             classification_report, confusion_matrix)

V05 = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation/v05"
SEED = 0
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- data
df = gpd.read_parquet(f"{V05}/dataset.parquet")
df = df[df["semantic"] != "unlabeled"].copy().reset_index(drop=True)

CLASSES = ["residential_estate", "residential_lowrise", "commercial_retail",
           "office_finance", "public_admin", "education", "industrial", "healthcare"]
SHORT = {"residential_estate": "res_estate", "residential_lowrise": "res_lowrise",
         "commercial_retail": "commercial", "office_finance": "office",
         "public_admin": "public_admin", "education": "education",
         "industrial": "industrial", "healthcare": "healthcare"}
POWER = {"public_admin", "education", "healthcare"}

FORM_COLS = ["fp_area", "height", "floors", "elong"]
CTX_RAW = ["density_100m", "nbr_mean_h", "dist_road_major", "dist_road_all",
           "dist_water", "dist_green"]
DIST_COLS = ["dist_road_major", "dist_road_all", "dist_water", "dist_green"]

# NaN handling (DISCLOSED choice): median-impute per column, and for the dist_*
# features add binary <col>_isnan flags so the model can distinguish "no water in
# window" (imputed) from a real distance. dist_water is NaN in ~52% of rows.
nan_info = {}
for c in CTX_RAW:
    nshare = float(df[c].isna().mean())
    med = float(df[c].median())
    if c in DIST_COLS:
        df[c + "_isnan"] = df[c].isna().astype(int)
    if nshare > 0:
        df[c] = df[c].fillna(med)
    nan_info[c] = {"nan_share": round(nshare, 5), "imputed_median": round(med, 2)}

CTX_COLS = CTX_RAW + [c + "_isnan" for c in DIST_COLS]
FEATSETS = {"FORM": FORM_COLS, "FORM+CTX": FORM_COLS + CTX_COLS}

y = df["semantic"].values
groups = df["window"].values
maj_class = pd.Series(y).value_counts().idxmax()
baseline_acc = float((y == maj_class).mean())
print(f"n={len(df)}  majority={maj_class}  baseline acc={baseline_acc:.4f}")

def make_rf():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)

def metrics_block(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=CLASSES,
                                output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_dict = {t: {p: int(cm[i, j]) for j, p in enumerate(CLASSES)}
               for i, t in enumerate(CLASSES)}
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro",
                                         zero_division=0)), 4),
        "per_class": {c: {k: round(float(v), 4) for k, v in rep[c].items()}
                      for c in CLASSES},
        "confusion": cm_dict,
    }, cm

# ---------------------------------------------------------------- CV runs
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
gkf = GroupKFold(n_splits=24)   # 24 windows -> leave-one-window-out

runs, preds = {}, {}
for split_name, cv, grp in [("strat5", skf, None), ("group24", gkf, groups)]:
    for fs_name, cols in FEATSETS.items():
        X = df[cols].values
        yp = cross_val_predict(make_rf(), X, y, cv=cv, groups=grp, n_jobs=1)
        key = f"{split_name}_{fs_name}"
        preds[key] = yp
        runs[key], _ = metrics_block(y, yp)
        r = runs[key]
        print(f"{key:22s} acc={r['accuracy']:.3f} bal={r['balanced_accuracy']:.3f} "
              f"mF1={r['macro_f1']:.3f}")

# power-semantic supergroup: pooled recall under GroupKFold
power_mask = np.isin(y, list(POWER))
power_stats = {}
for fs_name in FEATSETS:
    for split_name in ["strat5", "group24"]:
        yp = preds[f"{split_name}_{fs_name}"]
        pooled = float((yp[power_mask] == y[power_mask]).mean())
        power_stats[f"{split_name}_{fs_name}"] = {
            "pooled_recall": round(pooled, 4),
            "n_power_buildings": int(power_mask.sum()),
            "per_class_recall": {c: runs[f"{split_name}_{fs_name}"]["per_class"][c]["recall"]
                                 for c in POWER},
        }

# per-window accuracy under GroupKFold (each window scored when held out)
role_map = df.groupby("window")["role"].first().to_dict()
per_window = {}
for w in sorted(role_map):
    m = groups == w
    per_window[w] = {
        "n": int(m.sum()), "role": role_map[w],
        "acc_FORM": round(float((preds["group24_FORM"][m] == y[m]).mean()), 4),
        "acc_FORM+CTX": round(float((preds["group24_FORM+CTX"][m] == y[m]).mean()), 4),
    }
accs_ctx = np.array([per_window[w]["acc_FORM+CTX"] for w in sorted(role_map)])
print(f"per-window FORM+CTX acc: min={accs_ctx.min():.3f} med={np.median(accs_ctx):.3f} "
      f"max={accs_ctx.max():.3f}")

# feature importances: FORM+CTX RF fit on ALL data (descriptive, not a transfer claim)
rf_all = make_rf().fit(df[FEATSETS["FORM+CTX"]].values, y)
fi = {c: round(float(v), 4)
      for c, v in zip(FEATSETS["FORM+CTX"], rf_all.feature_importances_)}

# ---------------------------------------------------------------- save results
results = {
    "experiment": "e1_building_level_recognition",
    "n_buildings_labeled": int(len(df)),
    "classes": CLASSES,
    "feature_sets": {k: v for k, v in FEATSETS.items()},
    "excluded_features_reason": ("POI/EULUC/AOI-derived features excluded: these sources "
                                 "produced the fused labels (circularity). token_v03/v04 "
                                 "excluded as label-adjacent (family enters token stats)."),
    "nan_handling": nan_info,
    "baseline": {"majority_class": maj_class, "accuracy": round(baseline_acc, 4)},
    "runs": runs,
    "power_supergroup": power_stats,
    "per_window_group24": per_window,
    "feature_importances_full_fit": fi,
}
with open(f"{V05}/results_e1.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- fig 1: confusion
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2))
for ax, fs_name in zip(axes, FEATSETS):
    cm = confusion_matrix(y, preds[f"group24_{fs_name}"], labels=CLASSES)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), [SHORT[c] for c in CLASSES], rotation=45, ha="right",
                  fontsize=8)
    ax.set_yticks(range(8), [SHORT[c] for c in CLASSES], fontsize=8)
    for i in range(8):
        for j in range(8):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=6.5, color="white" if cmn[i, j] > 0.5 else "black")
    r = runs[f"group24_{fs_name}"]
    ax.set_title(f"{fs_name} — GroupKFold by window\nacc={r['accuracy']:.3f}  "
                 f"bal-acc={r['balanced_accuracy']:.3f}  macro-F1={r['macro_f1']:.3f}",
                 fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
axes[0].set_ylabel("True (fused label)", fontsize=9)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="row-normalized share")
fig.suptitle("E1: confusion matrices under spatial transfer (leave-one-window-out), "
             "row-normalized, counts in parentheses", fontsize=11, y=0.99)
fig.savefig(f"{V05}/fig/fig_e1_confusion.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- fig 2: transfer
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2),
                         gridspec_kw={"width_ratios": [1.1, 1.5, 1.0]})

# (a) metric bars: 2 splits x 2 feature sets
ax = axes[0]
metric_keys = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced acc."),
               ("macro_f1", "Macro-F1")]
combos = [("strat5_FORM", "Random 5-fold, FORM", "#9ecae1"),
          ("strat5_FORM+CTX", "Random 5-fold, FORM+CTX", "#3182bd"),
          ("group24_FORM", "Window-grouped, FORM", "#fdae6b"),
          ("group24_FORM+CTX", "Window-grouped, FORM+CTX", "#e6550d")]
xpos = np.arange(len(metric_keys))
bw = 0.2
for k, (key, lab, col) in enumerate(combos):
    vals = [runs[key][mk] for mk, _ in metric_keys]
    bars = ax.bar(xpos + (k - 1.5) * bw, vals, bw, label=lab, color=col)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=7)
ax.axhline(baseline_acc, color="k", ls="--", lw=1)
ax.text(2.45, baseline_acc + 0.012, f"majority baseline {baseline_acc:.2f}",
        fontsize=7.5, ha="right")
ax.set_xticks(xpos, [lab for _, lab in metric_keys], fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("(a) Random CV vs spatial transfer", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right")

# (b) per-window GroupKFold accuracy sorted
ax = axes[1]
order = sorted(role_map, key=lambda w: per_window[w]["acc_FORM+CTX"])
vals = [per_window[w]["acc_FORM+CTX"] for w in order]
cols = ["#d62728" if role_map[w] == "heldout" else "#7f7f7f" for w in order]
ax.bar(range(len(order)), vals, color=cols)
ax.axhline(runs["group24_FORM+CTX"]["accuracy"], color="k", ls="--", lw=1,
           label=f"pooled acc {runs['group24_FORM+CTX']['accuracy']:.2f}")
ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7.5)
ax.set_ylim(0, 1.0)
ax.set_title("(b) Per-window accuracy when held out (FORM+CTX)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#7f7f7f", label="v03-train window"),
                   Patch(color="#d62728", label="v03-heldout window"),
                   plt.Line2D([0], [0], color="k", ls="--",
                              label=f"pooled acc {runs['group24_FORM+CTX']['accuracy']:.2f}")],
          fontsize=7.5, loc="upper left")
ax.set_ylabel("Accuracy")

# (c) feature importances
ax = axes[2]
fi_sorted = sorted(fi.items(), key=lambda kv: kv[1])
names = [k for k, _ in fi_sorted]
vals = [v for _, v in fi_sorted]
bar_cols = ["#2ca02c" if n.startswith("dist_") or n.endswith("_isnan")
            or n in ("density_100m", "nbr_mean_h") else "#1f77b4" for n in names]
ax.barh(range(len(names)), vals, color=bar_cols)
ax.set_yticks(range(len(names)), names, fontsize=8)
ax.set_title("(c) RF feature importance (FORM+CTX, full fit)", fontsize=10)
ax.legend(handles=[Patch(color="#1f77b4", label="form"),
                   Patch(color="#2ca02c", label="context (v04-derived + density)")],
          fontsize=7.5, loc="lower right")
fig.suptitle("E1: does form-based recognition transfer across windows?", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f"{V05}/fig/fig_e1_transfer.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("done; results_e1.json + 2 figures written")
