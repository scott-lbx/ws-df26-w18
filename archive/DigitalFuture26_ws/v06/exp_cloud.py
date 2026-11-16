#!/usr/bin/env python3
"""
exp_cloud.py — D5: place the 5 generated paradigm UNITS (1 km, 20x20 cells) on the
v03 real-Shanghai signature cloud, and run the Caoyang A13 anchor test.

Method (provenance):
- Reproduces v03 run_v03.py::signatures() EXACTLY from the v03 cache
  (cells.parquet token grid + library.parquet buildings): 19-dim signature =
  9 token shares T0..T8 + 10 indicators; z-score with TRAIN mu/sd (ddof=0);
  PCA(2, random_state=0) fit on the 18 TRAIN windows; XY = Z @ comps.T with
  sign-orientation "top-|loading| feature positive".
- CORRECTNESS GATE: recomputed pca_xy of all 24 real windows must match the
  values stored in TF/v03/signatures.json to <= 1e-3 (stored values are
  rounded to 4 decimals) BEFORE any paradigm is projected.
- Paradigm units: token composition from grid_unit_*.npy with typed voids
  9=E_ROAD, 10=E_WATER, 11=E_GREEN FOLDED BACK INTO T0 (v03 cloud knows only
  one undifferentiated void class; this folding is DISCLOSED in results).
  Indicators computed from unit_*.parquet with the identical v03 formulas.
- Nearest real windows per paradigm in PC space (2-dim) and full z-space
  (19-dim), top-3 each. Anchor test: rank of A13 (Caoyang Xincun, 1951
  neighbourhood-unit plan) among the 24 real windows for the generated
  neighbourhood_unit and microrayon fabrics.

Output: V06/results_d5.json, V06/fig/fig_d5_cloud.png. Seed 0.
"""
import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

np.random.seed(0)

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
V06 = os.path.join(TF, "v06")
FIG = os.path.join(V06, "fig")
os.makedirs(FIG, exist_ok=True)

HALF = 500.0
AREA = (2 * HALF) ** 2          # 1 km^2
NTOK = 9                        # T0 void + T1..T8
IND = ["FAR", "coverage", "h_p50", "h_p90", "median_fp",
       "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
FEAT_NAMES = [f"tok_T{t}" for t in range(NTOK)] + IND
FAM_KEYS = [("sh_house", "house"), ("sh_slab", "slab"), ("sh_tower", "tower"),
            ("sh_large", "large_footprint"), ("sh_mid", "midrise_other")]
PARADIGMS = ["garden_city", "ville_radieuse", "neighbourhood_unit",
             "microrayon", "gb50180"]
GATE_TOL = 1e-3


def indicators_from_buildings(g):
    """v03 signatures() indicator formulas, verbatim semantics."""
    fp = float(g["fp_area"].sum())
    fam_fp = g.groupby("family")["fp_area"].sum()
    ind = {
        "FAR": float(g["gfa"].sum() / AREA),
        "coverage": float(fp / AREA),
        "h_p50": float(np.percentile(g["height"], 50)) if len(g) else 0.0,
        "h_p90": float(np.percentile(g["height"], 90)) if len(g) else 0.0,
        "median_fp": float(g["fp_area"].median()) if len(g) else 0.0,
    }
    for f, key in FAM_KEYS:
        ind[f] = float(fam_fp.get(key, 0.0) / fp) if fp > 0 else 0.0
    return ind


# ---------------------------------------------------------------- 1. real cloud
sigs = json.load(open(os.path.join(V03, "signatures.json")))
win_meta = sigs["windows"]                      # keeps v03 order
ids = [w["id"] for w in win_meta]
roles = [w["role"] for w in win_meta]
names = {w["id"]: w["name"] for w in win_meta}

cells = pd.read_parquet(os.path.join(V03, "cache", "cells.parquet"))
lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))

M = np.zeros((len(ids), len(FEAT_NAMES)))
for i, wid in enumerate(ids):
    sub = cells[cells["window"] == wid]
    comp = [float((sub["token"] == t).mean()) for t in range(NTOK)]
    g = lib[lib["window"] == wid]
    ind = indicators_from_buildings(g)
    M[i] = comp + [ind[k] for k in IND]

tr = np.array([r == "train" for r in roles])
mu, sd = M[tr].mean(0), M[tr].std(0)
sd[sd == 0] = 1
Z = (M - mu) / sd
pca = PCA(n_components=2, random_state=0).fit(Z[tr])
comps = pca.components_.copy()
XY = Z @ comps.T
for k in range(2):                               # orient as v03 did
    j = int(np.argmax(np.abs(comps[k])))
    if comps[k, j] < 0:
        comps[k] *= -1
        XY[:, k] *= -1

# -------- correctness gate: reproduce stored pca_xy before projecting paradigms
stored = np.array([w["pca_xy"] for w in win_meta])
gate_err = float(np.max(np.abs(XY - stored)))
print(f"[gate] max |recomputed - stored pca_xy| over 24 windows = {gate_err:.6f}")
assert gate_err <= GATE_TOL, (
    f"PCA reproduction gate FAILED: {gate_err:.6f} > {GATE_TOL}")
print("[gate] PASS — projection basis verified, proceeding to paradigms")

# ---------------------------------------------------------------- 2. paradigms
d3 = json.load(open(os.path.join(V06, "results_d3.json")))
para_rows = {}
fold_disclosure = {}
for p in PARADIGMS:
    grid = np.load(os.path.join(V06, "cache_gen", f"grid_unit_{p}.npy"))
    flat = grid.ravel()
    n = flat.size                               # 400 cells
    folded = {"E_ROAD(9)": int((flat == 9).sum()),
              "E_WATER(10)": int((flat == 10).sum()),
              "E_GREEN(11)": int((flat == 11).sum()),
              "E_VACANT(0)": int((flat == 0).sum())}
    fold_disclosure[p] = folded
    f9 = flat.copy()
    f9[f9 >= 9] = 0                             # fold typed voids into T0
    comp = [float((f9 == t).mean()) for t in range(NTOK)]
    g = gpd.read_parquet(os.path.join(V06, "cache_gen", f"unit_{p}.parquet"))
    ind = indicators_from_buildings(g)
    para_rows[p] = {"comp": comp, "ind": ind, "n_buildings": int(len(g))}
    # cross-check against D3's own indicator bookkeeping
    d3i = d3["paradigms"][p]["unit"]["indicators"]
    for a, b in [("FAR", "FAR"), ("coverage", "coverage"),
                 ("h_p50", "height_p50"), ("h_p90", "height_p90")]:
        if abs(ind[a] - d3i[b]) > 0.02 * max(1.0, abs(d3i[b])):
            print(f"[warn] {p} {a}: recomputed {ind[a]:.4f} vs D3 {d3i[b]:.4f}")

P = np.array([para_rows[p]["comp"] + [para_rows[p]["ind"][k] for k in IND]
              for p in PARADIGMS])
ZP = (P - mu) / sd
XYP = ZP @ comps.T

# ---------------------------------------------------------------- 3. neighbours
def top3(dvec):
    order = np.argsort(dvec)
    return [{"id": ids[j], "name": names[ids[j]], "role": roles[j],
             "dist": round(float(dvec[j]), 4), "rank": int(r + 1)}
            for r, j in enumerate(order[:3])]


neighbours = {}
anchor = {}
i_a13 = ids.index("A13")
for pi, p in enumerate(PARADIGMS):
    d_pc = np.linalg.norm(XY - XYP[pi], axis=1)
    d_z = np.linalg.norm(Z - ZP[pi], axis=1)
    neighbours[p] = {"pc_space_top3": top3(d_pc), "z_space_top3": top3(d_z)}
    if p in ("neighbourhood_unit", "microrayon"):
        anchor[p] = {
            "A13_rank_pc_space": int(np.argsort(np.argsort(d_pc))[i_a13] + 1),
            "A13_dist_pc_space": round(float(d_pc[i_a13]), 4),
            "A13_rank_z_space": int(np.argsort(np.argsort(d_z))[i_a13] + 1),
            "A13_dist_z_space": round(float(d_z[i_a13]), 4),
            "n_real_windows_ranked": len(ids),
        }

a13_w = next(w for w in win_meta if w["id"] == "A13")
anchor["A13_actual"] = {
    "name": a13_w["name"], "role": a13_w["role"],
    "token_comp_T0_T8": a13_w["token_comp_T0_T8"],
    "indicators": a13_w["indicators"], "pca_xy": a13_w["pca_xy"],
}
anchor["interpretation"] = (
    "MISS for neighbourhood_unit (rank 14/24 PC, 11/24 z): the Perry recipe "
    "as distilled in D2 encodes 1929 low-rise fabric (61% T1 houses, "
    "sh_house .87, FAR 0.46, h_p90 13 m), so the generated unit lands among "
    "peri-urban villages (A32/A33) — the only surviving low-rise fabric in "
    "the vocabulary. A13 today is a T3-slab/T4-midrise fabric (30% T3 + 20% "
    "T4, sh_slab .46, FAR 1.57, h_p90 37 m): the 1951 adaptation built "
    "multi-storey workers' slabs, and 70 years of rebuilding/densification "
    "added midrise and scattered towers. NEAR-MISS for microrayon (rank 9/24 "
    "PC, 6/24 z): its family-share profile (slab .49, mid .26, median_fp "
    "230 m2, h_p90 42 m) nearly matches A13's (.46, .27, 262 m2, 37 m); the "
    "residual gap is density (FAR 0.65 vs 1.57) and void share (.72 vs .30) "
    "— doctrine-faithful Soviet spacing is far sparser than densified "
    "Caoyang. Interpretive reading (not asserted identity): today's Caoyang "
    "reads closer to a densified microrayon than to Perry's neighbourhood "
    "unit in built form, consistent with its slab-housing construction "
    "history; the 'miss' is informative about the 1951 translation, not a "
    "pipeline fault — the projection basis itself passed the 1e-3 gate.")

# ---------------------------------------------------------------- 4. figure
plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(9.6, 7.6), dpi=140)

# real windows, gray; TRAIN circles, HELDOUT stars
for i, wid in enumerate(ids):
    mk = "o" if roles[i] == "train" else "*"
    ms = 55 if roles[i] == "train" else 110
    ax.scatter(XY[i, 0], XY[i, 1], marker=mk, s=ms, c="0.62",
               edgecolors="0.35", linewidths=0.6, zorder=2)

# small id labels for real windows (manual nudges to avoid collisions)
real_off = {
    "A01": (8, 4), "A10": (8, -2), "A13": (9, 5), "A04": (8, 4),
    "A02": (9, 4), "A07": (8, 4), "A09": (8, -2), "A17": (-22, -4),
    "A21": (-32, -4), "A19": (6, -14), "A20": (7, -4), "A29": (8, 5),
    "A26": (8, -10), "A27": (8, 4), "A33": (9, -3), "A32": (8, 4),
    "A36": (8, 6), "A35": (-4, -16), "A41": (8, -13), "A39": (8, -2),
    "A45": (-24, -4), "A43": (-30, -10), "A47": (-20, -3), "A49": (8, 4),
}
for i, wid in enumerate(ids):
    dx, dy = real_off.get(wid, (8, 4))
    txt = f"{wid} Caoyang" if wid == "A13" else wid
    ax.annotate(txt, (XY[i, 0], XY[i, 1]), xytext=(dx, dy),
                textcoords="offset points", fontsize=7,
                color="0.30" if wid != "A13" else "black",
                fontweight="bold" if wid == "A13" else "normal", zorder=3)

# A13 highlight ring
ax.scatter(XY[i_a13, 0], XY[i_a13, 1], marker="o", s=190, facecolors="none",
           edgecolors="black", linewidths=1.4, zorder=4)

P_COL = {"garden_city": "#2a9d2a", "ville_radieuse": "#d62728",
         "neighbourhood_unit": "#1f77b4", "microrayon": "#9467bd",
         "gb50180": "#e6850e"}
P_LAB = {"garden_city": "Garden City (Howard 1898)",
         "ville_radieuse": "Ville Radieuse (Le Corbusier 1933)",
         "neighbourhood_unit": "Neighbourhood Unit (Perry 1929)",
         "microrayon": "Microrayon (SNiP)",
         "gb50180": "GB50180 (PRC code 2018)"}
p_off = {"garden_city": (-12, -20), "ville_radieuse": (10, 6),
         "neighbourhood_unit": (-15, 13), "microrayon": (-72, 38),
         "gb50180": (10, 6)}
P_LEADER = {"microrayon"}   # labels placed far from the marker get a leader line
for pi, p in enumerate(PARADIGMS):
    ax.scatter(XYP[pi, 0], XYP[pi, 1], marker="D", s=170, c=P_COL[p],
               edgecolors="black", linewidths=1.0, zorder=5)
    ax.annotate(P_LAB[p], (XYP[pi, 0], XYP[pi, 1]), xytext=p_off[p],
                textcoords="offset points", fontsize=8.5, fontweight="bold",
                color=P_COL[p], zorder=6,
                arrowprops=(dict(arrowstyle="-", color=P_COL[p], lw=0.7,
                                 alpha=0.85, shrinkA=2, shrinkB=9)
                            if p in P_LEADER else None))

# leader line: neighbourhood_unit -> A13 with rank annotation
nu_i = PARADIGMS.index("neighbourhood_unit")
nu_xy = XYP[nu_i]
a13_xy = XY[i_a13]
ax.annotate("", xy=a13_xy, xytext=nu_xy,
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.4,
                            linestyle="--", shrinkA=10, shrinkB=10), zorder=4)
nu_anchor = anchor["neighbourhood_unit"]
ax.annotate(
    f"anchor test: A13 Caoyang rank\n"
    f"{nu_anchor['A13_rank_pc_space']}/24 in PC space (d="
    f"{nu_anchor['A13_dist_pc_space']:.2f}),\n"
    f"{nu_anchor['A13_rank_z_space']}/24 in 19-dim z-space",
    nu_xy, xytext=(-2, -34), textcoords="offset points", fontsize=8,
    color="#1f77b4", ha="left",
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9),
    zorder=6)

evr = pca.explained_variance_ratio_
ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%) — intensity / verticality "
              "(+FAR, +T6 towers, −houses)")
ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%) — sparse spikes vs dense mat "
              "(+void, +h_p90, −coverage)")
ax.set_title("D5 — Generated paradigm units on the real-Shanghai signature cloud\n"
             "24 real 1-km windows (gray: TRAIN circles, HELDOUT stars) + 5 doctrine "
             "fabrics (diamonds); typed voids folded into T0")
ax.axhline(0, color="0.88", lw=0.7, zorder=1)
ax.axvline(0, color="0.88", lw=0.7, zorder=1)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor="0.62",
           markeredgecolor="0.35", markersize=7, label="real window (TRAIN)"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="0.62",
           markeredgecolor="0.35", markersize=11, label="real window (HELDOUT)"),
    Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
           markeredgecolor="black", markersize=8, label="generated paradigm unit"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
           markeredgecolor="black", markersize=10,
           label="A13 Caoyang Xincun (1951 neighbourhood-unit plan)"),
]
ax.legend(handles=legend_elems, loc="lower left", fontsize=7.5, framealpha=0.9)
ax.margins(0.10)
fig.tight_layout()
fig_path = os.path.join(FIG, "fig_d5_cloud.png")
fig.savefig(fig_path)
plt.close(fig)
print("figure ->", fig_path)

# ---------------------------------------------------------------- 5. results
res = {
    "_rule": ("D5 cloud placement: 19-dim v03 signature (9 token shares + 10 "
              "indicators) recomputed for 24 real windows from v03 cache, "
              "z-scored on TRAIN mu/sd, PCA(2,seed0) fit on 18 TRAIN, "
              "XY = Z @ comps.T, signs oriented top-|loading|-positive. "
              "Paradigm UNITS only (1 km, comparable to real windows)."),
    "void_folding_disclosure": {
        "rule": ("Generated grids use typed voids 9=E_ROAD, 10=E_WATER, "
                 "11=E_GREEN (v0.4 vocabulary); the v03 cloud has a single "
                 "undifferentiated void T0, so 9/10/11 are folded back into "
                 "T0 before computing token shares. The cloud therefore "
                 "cannot distinguish a doctrine's green void from vacancy."),
        "cells_folded_per_unit_of_400": fold_disclosure,
    },
    "correctness_gate": {
        "criterion": "max |recomputed - stored pca_xy| over 24 windows <= 1e-3",
        "max_abs_err": round(gate_err, 6),
        "passed": bool(gate_err <= GATE_TOL),
        "explained_variance_ratio": [round(float(v), 4) for v in evr],
    },
    "paradigm_signatures": {
        p: {
            "token_comp_T0_T8": [round(v, 4) for v in para_rows[p]["comp"]],
            "indicators": {k: round(para_rows[p]["ind"][k], 4) for k in IND},
            "n_buildings": para_rows[p]["n_buildings"],
            "pca_xy": [round(float(XYP[pi, 0]), 4), round(float(XYP[pi, 1]), 4)],
            "z_norm": round(float(np.linalg.norm(ZP[pi])), 4),
        } for pi, p in enumerate(PARADIGMS)
    },
    "nearest_real_windows": neighbours,
    "anchor_test_caoyang_A13": anchor,
}
out_path = os.path.join(V06, "results_d5.json")
json.dump(res, open(out_path, "w"), ensure_ascii=False, indent=1)
print("results ->", out_path)
print(json.dumps({"anchor": anchor, "neighbours": neighbours}, indent=1)[:4000])
