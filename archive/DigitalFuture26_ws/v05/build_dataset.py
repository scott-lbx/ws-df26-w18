#!/usr/bin/env python3
"""
v05 Phase A — assemble the analysis dataset.

Outputs (all under typology_foundation/v05/):
  dataset.parquet   one row per library building + grid/token/distance/density columns
  windows.json      per-window composition + provenance summary
  results_e0.json   dataset summary (counts, NaN shares)
  fig/fig_e0_map.png, fig/fig_e0_dataset.png

Conventions (inherited from v03/v04, see NOTES_AGENT1.md / NOTES_V04.md):
  - grid: 20x20 cells of 50 m, origin (cx-500, cy-500); row = (y-oy)//50, col = (x-ox)//50,
    clipped to 0..19; building rep point = polygon centroid (same as v03 cells.parquet,
    so token_v03 here is exactly the token of the cell the building was pooled into).
  - masks are clipped to the 1 km window -> distances are UPPER BOUNDS w.r.t. features
    just outside the window (same edge bias for every window).
  - seed 0 everywhere; EPSG:32651 throughout.
"""
import json
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
V04 = os.path.join(TF, "v04")
V05 = os.path.join(TF, "v05")
FIG = os.path.join(V05, "fig")
os.makedirs(FIG, exist_ok=True)

CELL, NCELL, HALF = 50.0, 20, 500.0
MASK_CLASSES = ["road_major", "road", "water", "green"]
DIST_COL = {"road_major": "dist_road_major", "road": "dist_road_all",
            "water": "dist_water", "green": "dist_green"}
SEM_CLASSES = ["residential_estate", "residential_lowrise", "commercial_retail",
               "office_finance", "industrial", "public_admin", "education", "healthcare"]
FAM_CLASSES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
SEM_COLORS = {  # consistent across all v05 figures
    "residential_estate": "#d95f02", "residential_lowrise": "#fdbf6f",
    "commercial_retail": "#e7298a", "office_finance": "#7570b3",
    "industrial": "#666666", "public_admin": "#1b9e77",
    "education": "#66a61e", "healthcare": "#e6ab02", "unlabeled": "#cccccc",
}


def load_split():
    return json.load(open(os.path.join(V03, "split.json")))


# ================================================================ dataset
def build_dataset(split):
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))
    cells = pd.read_parquet(os.path.join(V03, "cache", "cells.parquet"))
    wmeta = {w["id"]: w for w in split["windows"]}

    # rep point = centroid (matches v03 cells.parquet cell assignment exactly)
    cent = lib.geometry.centroid
    lib["_px"], lib["_py"] = cent.x.values, cent.y.values

    out = []
    for wid, w in wmeta.items():
        g = lib[lib["window"] == wid].copy()
        ox, oy = w["cx"] - HALF, w["cy"] - HALF
        g["col"] = np.clip(((g["_px"] - ox) // CELL).astype(int), 0, NCELL - 1)
        g["row"] = np.clip(((g["_py"] - oy) // CELL).astype(int), 0, NCELL - 1)

        # tokens
        ctok = cells[cells["window"] == wid].set_index(["row", "col"])["token"]
        g["token_v03"] = ctok.loc[list(zip(g["row"], g["col"]))].values.astype(int)
        g4 = np.load(os.path.join(V04, f"token_grid_v04_{wid}.npy"))
        g["token_v04"] = g4[g["row"].values, g["col"].values].astype(int)

        # distances to mask classes (window coords; masks already clipped to window)
        mask = gpd.read_parquet(os.path.join(V04, "cache", f"{wid}_mask.parquet"))
        mgeom = {r["cls"]: r.geometry for _, r in mask.iterrows()
                 if r.geometry is not None and not r.geometry.is_empty}
        pts = shapely.points(g["_px"].values, g["_py"].values)
        for cls in MASK_CLASSES:
            col = DIST_COL[cls]
            if cls in mgeom:
                g[col] = shapely.distance(pts, mgeom[cls])
                g[f"has_{cls}_in_window"] = True
            else:
                g[col] = np.nan
                g[f"has_{cls}_in_window"] = False

        # local density (same window only -> undercount near window edges)
        xy = g[["_px", "_py"]].values
        tree = cKDTree(xy)
        nbrs = tree.query_ball_point(xy, r=100.0)
        h = g["height"].values
        dens, nmh = np.empty(len(g), int), np.full(len(g), np.nan)
        for i, idx in enumerate(nbrs):
            others = [j for j in idx if j != i]
            dens[i] = len(others)
            if others:
                nmh[i] = h[others].mean()
        g["density_100m"], g["nbr_mean_h"] = dens, nmh
        g["role"], g["name"] = w["role"], w["name"]
        out.append(g)

    ds = pd.concat(out).sort_values("bidx").drop(columns=["_px", "_py"])
    ds = gpd.GeoDataFrame(ds, geometry="geometry", crs=lib.crs)
    ds.to_parquet(os.path.join(V05, "dataset.parquet"))
    print(f"[dataset] {len(ds)} rows -> dataset.parquet")
    return ds


# ================================================================ windows.json
def build_windows(split, ds):
    sigs = json.load(open(os.path.join(V03, "signatures.json")))
    sigmap = {s["id"]: s for s in sigs["windows"]}
    rec = {}
    for w in split["windows"]:
        wid = w["id"]
        g = ds[ds["window"] == wid]
        gfa_tot = float(g["gfa"].sum())
        lab = g[g["semantic"] != "unlabeled"]
        lab_gfa = float(lab["gfa"].sum())
        sem_share = {s: round(float(lab.loc[lab["semantic"] == s, "gfa"].sum() / lab_gfa), 4)
                     if lab_gfa > 0 else 0.0 for s in SEM_CLASSES}
        fam_share = {f: round(float(g.loc[g["family"] == f, "gfa"].sum() / gfa_tot), 4)
                     if gfa_tot > 0 else 0.0 for f in FAM_CLASSES}

        # residential_estate age regimes (GFA split, matches power_priors_v01 cells)
        re_ = g[g["semantic"] == "residential_estate"]
        re_gfa = float(re_["gfa"].sum())
        if re_gfa > 0:
            a = re_["age"]
            regimes = {
                "pre1985": float(re_.loc[a <= 1985, "gfa"].sum() / re_gfa),
                "1985_1997": float(re_.loc[(a > 1985) & (a < 1998), "gfa"].sum() / re_gfa),
                "post1998": float(re_.loc[a >= 1998, "gfa"].sum() / re_gfa),
                "no_age": float(re_.loc[a.isna(), "gfa"].sum() / re_gfa),
            }
            regimes = {k: round(v, 4) for k, v in regimes.items()}
        else:
            regimes = {"pre1985": 0.0, "1985_1997": 0.0, "post1998": 0.0, "no_age": 0.0}

        g4 = np.load(os.path.join(V04, f"token_grid_v04_{wid}.npy"))
        voids = {"E_VAC": int((g4 == 0).sum()), "E_ROAD": int((g4 == 9).sum()),
                 "E_WATER": int((g4 == 10).sum()), "E_GREEN": int((g4 == 11).sum()),
                 "built_cells": int(((g4 >= 1) & (g4 <= 8)).sum())}

        s = sigmap[wid]
        rec[wid] = {
            "id": wid, "name": w["name"], "role": w["role"], "type_group": w["type_group"],
            "n_buildings": int(len(g)),
            "labeled_pct": round(float((g["semantic"] != "unlabeled").mean() * 100), 2),
            "gfa_total": round(gfa_tot, 1),
            "unlabeled_gfa_share": round(float((gfa_tot - lab_gfa) / gfa_tot), 4) if gfa_tot > 0 else 0.0,
            "semantic_gfa_share_labeled": sem_share,
            "family_gfa_share": fam_share,
            "age_filled_pct": round(float(g["age"].notna().mean() * 100), 2),
            "res_estate_age_regime_gfa_share": regimes,
            "void_cells": voids,
            "v03_indicators": s["indicators"],
            "v03_pca_xy": s["pca_xy"],
            "v03_token_comp_T0_T8": s["token_comp_T0_T8"],
        }
    out = {
        "meta": {
            "built_by": "v05/build_dataset.py (Phase A)",
            "labeled_pct": "share of BUILDINGS (count) with semantic != unlabeled",
            "semantic_gfa_share_labeled": "GFA shares over labeled GFA only; see unlabeled_gfa_share",
            "res_estate_age_regimes": "pre1985: age<=1985; 1985_1997: 1986-1997; post1998: age>=1998; GFA shares of residential_estate GFA",
            "void_cells": "counts of 50m cells in v04 typed grid (400 per window)",
        },
        "windows": rec,
    }
    json.dump(out, open(os.path.join(V05, "windows.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[windows] {len(rec)} windows -> windows.json")
    return out


# ================================================================ results
NEW_FEATS = ["row", "col", "token_v03", "token_v04", "dist_road_major", "dist_road_all",
             "dist_water", "dist_green", "density_100m", "nbr_mean_h"]


def build_results(ds):
    feats = NEW_FEATS + ["age", "height", "fp_area"]
    nan_share = {f: round(float(ds[f].isna().mean()), 4) for f in feats}
    res = {
        "n_rows": int(len(ds)),
        "new_columns": NEW_FEATS + [f"has_{c}_in_window" for c in MASK_CLASSES] + ["role", "name"],
        "nan_share": nan_share,
        "per_window_n": ds.groupby("window").size().to_dict(),
        "semantic_counts": ds["semantic"].value_counts().to_dict(),
        "family_counts": ds["family"].value_counts().to_dict(),
        "has_class_in_window_false": {
            c: sorted(ds.loc[~ds[f"has_{c}_in_window"], "window"].unique().tolist())
            for c in MASK_CLASSES},
        "buildings_in_T0_cells_v03": int((ds["token_v03"] == 0).sum()),
        "buildings_in_void_cells_v04": int(ds["token_v04"].isin([0, 9, 10, 11]).sum()),
        "density_100m_median": float(ds["density_100m"].median()),
        "notes": [
            "distances are upper bounds: masks clipped to the 1 km window (edge bias, identical treatment all windows)",
            "density_100m / nbr_mean_h undercount near window edges (no neighbours outside window)",
        ],
    }
    json.dump(res, open(os.path.join(V05, "results_e0.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["n_rows", "nan_share",
                                          "buildings_in_T0_cells_v03",
                                          "buildings_in_void_cells_v04"]}, indent=1))
    return res


# ================================================================ figures
def _rect_label(ax, fig, x, y, text, boxes, fontsize=7.0, fontweight="normal"):
    """Greedy label placement, rectangle collision in DISPLAY space (px).

    Improves on the v03 fig1 point-based heuristic: TRUE rendered text extents
    (via the canvas renderer) are checked for overlap; candidates spiral out
    from the marker; far placements get a thin leader line.
    """
    renderer = fig.canvas.get_renderer()
    cands = []
    for r in (8, 16, 26, 38, 52, 70, 90, 112, 136):
        for ang in (25, -35, 90, 155, 215, -90, 60, -60, 0, 180,
                    120, -120, 30, -30, 150, -150):
            a = np.deg2rad(ang)
            cands.append((r * np.cos(a), r * np.sin(a)))
    pad = 2.0
    for dx, dy in cands:
        ha = "left" if dx >= 0 else "right"
        far = (dx ** 2 + dy ** 2) ** 0.5 > 14
        ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                          ha=ha, va="center", fontsize=fontsize, fontweight=fontweight,
                          zorder=5,
                          arrowprops=(dict(arrowstyle="-", lw=0.55, color="0.45",
                                           shrinkA=1, shrinkB=3) if far else None))
        # TEXT-only extent: Annotation.get_window_extent unions in the leader
        # line, whose long bbox would veto every far candidate in dense areas.
        # update_positions first (otherwise the offset transform is stale).
        ann.update_positions(renderer)
        e = matplotlib.text.Text.get_window_extent(ann, renderer)
        bb = (e.x0 - pad, e.y0 - pad, e.x1 + pad, e.y1 + pad)
        if any(bb[0] < b[2] and bb[2] > b[0] and bb[1] < b[3] and bb[3] > b[1]
               for b in boxes):
            ann.remove()
            continue
        boxes.append(bb)
        return
    ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(8, 4),
                      fontsize=fontsize, fontweight=fontweight, zorder=5)
    ann.update_positions(renderer)
    e = matplotlib.text.Text.get_window_extent(ann, renderer)
    boxes.append((e.x0, e.y0, e.x1, e.y1))
    print(f"[fig_map] WARNING: no free slot for label '{text}' (fallback used)")


def fig_map(split):
    ws = split["windows"]
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    lons = [w["lon"] for w in ws]; lats = [w["lat"] for w in ws]
    xr, yr = max(lons) - min(lons), max(lats) - min(lats)
    ax.set_xlim(min(lons) - 0.16 * xr, max(lons) + 0.22 * xr)
    ax.set_ylim(min(lats) - 0.09 * yr, max(lats) + 0.09 * yr)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Study area: 24 one-km windows, Shanghai")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()  # freeze axes geometry BEFORE display-space label placement
    fig.canvas.draw()   # ensure a renderer exists for true text extents
    boxes = []
    # markers first; reserve a small box around each marker so labels avoid them
    for w in ws:
        held = w["role"] == "heldout"
        ax.scatter(w["lon"], w["lat"], marker="*" if held else "o",
                   s=210 if held else 75, color="#e76f51" if held else "#2a9d8f",
                   edgecolor="black", linewidth=0.6, zorder=3)
        mx, my = ax.transData.transform((w["lon"], w["lat"]))
        boxes.append((mx - 8, my - 8, mx + 8, my + 8))
    # heldout labelled first (best slots), then train densest-area first
    order = sorted(ws, key=lambda w: (w["role"] != "heldout", w["id"]))
    for w in order:
        held = w["role"] == "heldout"
        _rect_label(ax, fig, w["lon"], w["lat"], f'{w["id"]} {w["name"]}', boxes,
                    fontsize=7.6 if held else 7.0,
                    fontweight="bold" if held else "normal")
    ax.scatter([], [], marker="o", s=75, color="#2a9d8f", edgecolor="black",
               linewidth=0.6, label="train (18)")
    ax.scatter([], [], marker="*", s=210, color="#e76f51", edgecolor="black",
               linewidth=0.6, label="heldout (6)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.savefig(os.path.join(FIG, "fig_e0_map.png"), dpi=140)
    plt.close(fig)
    print("[fig] fig_e0_map.png")


def _short_name(name, maxlen=22):
    """Shorten window names at a word boundary (no mid-word truncation)."""
    if len(name) <= maxlen:
        return name
    cut = name[:maxlen]
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut.rstrip(" /-")


def fig_dataset(wins):
    W = sorted(wins["windows"].values(), key=lambda r: r["labeled_pct"])
    labels = [f'{r["id"]} {_short_name(r["name"])}' + (" *" if r["role"] == "heldout" else "")
              for r in W]
    y = np.arange(len(W))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.5, 8.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) semantic GFA composition over TOTAL gfa: labeled shares scaled + unlabeled
    left = np.zeros(len(W))
    for s in SEM_CLASSES + ["unlabeled"]:
        if s == "unlabeled":
            vals = np.array([r["unlabeled_gfa_share"] for r in W])
        else:
            vals = np.array([r["semantic_gfa_share_labeled"][s] *
                             (1 - r["unlabeled_gfa_share"]) for r in W])
        axA.barh(y, vals, left=left, color=SEM_COLORS[s], height=0.78,
                 label=s.replace("_", " "))
        left += vals
    axA.set_yticks(y); axA.set_yticklabels(labels, fontsize=7.6)
    axA.set_xlim(0, 1.0); axA.set_xlabel("Share of window GFA")
    axA.set_title("(a) Semantic GFA composition per window\n(sorted by labeled %; * = heldout)",
                  fontsize=10.5)
    axA.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=7.4,
               framealpha=0.95, title="semantic", title_fontsize=8)

    # (b) n_buildings bars + labeled% line (same order)
    nb = [r["n_buildings"] for r in W]
    lp = [r["labeled_pct"] for r in W]
    axB.barh(y, nb, color="#5b8db8", height=0.78, label="n buildings")
    axB.set_yticks(y); axB.set_yticklabels([r["id"] for r in W], fontsize=7.6)
    axB.set_xlabel("Number of buildings", color="#3a6a96")
    axB2 = axB.twiny()
    axB2.plot(lp, y, "o-", color="#c1440e", markersize=4, linewidth=1.3,
              label="labeled %")
    axB2.set_xlim(0, 102); axB2.set_xlabel("Labeled buildings (%)", color="#c1440e")
    axB.set_title("(b) Dataset size and label coverage per window", fontsize=10.5)
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axB2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_e0_dataset.png"), dpi=140)
    plt.close(fig)
    print("[fig] fig_e0_dataset.png")


# ================================================================ main
if __name__ == "__main__":
    np.random.seed(0)
    split = load_split()
    ds = build_dataset(split)
    wins = build_windows(split, ds)
    build_results(ds)
    fig_map(split)
    fig_dataset(wins)
    print("[done]")
