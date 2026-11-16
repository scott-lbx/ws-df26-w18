#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v04.py — v0.4 Learn side: typed voids + no-build masks.

OWNER DIRECTIVE (2026-06-12): the generator treats every blank cell as buildable
— it "covers the blank as if it were road". v0.4 makes the void legible:
(1) build a per-window NO-BUILD mask from layers already on disk
    (04-交通数据 road network buffered by fclass; ESA10m water; OSM-landuse +
    EULUC green), and
(2) split T0 EMPTY into four observable void types
    E_VACANT(0) / E_ROAD(9) / E_WATER(10) / E_GREEN(11),
then recount the TRAIN adjacency over the 12-token vocabulary.
Built tokens T1..T8 are UNCHANGED (verbatim v0.3 vocabulary, pools untouched).

Steps (cached, idempotent, seed-free up to here):
    python3 run_v04.py prep                # one citywide read per layer -> cache/{water,roads,green}.parquet
    python3 run_v04.py masks A01 A02 ...   # per-window mask -> cache/<id>_mask.parquet
    python3 run_v04.py calib               # real-building violation stats (mask honesty)
    python3 run_v04.py voids               # typed grids token_grid_v04_<id>.npy + decomposition
    python3 run_v04.py adjacency           # 12-token TRAIN adjacency -> adjacency_v04.json
    python3 run_v04.py fig1                # typed-void maps + T0 decomposition
    python3 run_v04.py fig2                # 12-token adjacency vs v0.3 baseline

Vectors as FlatGeobuf via /tmp (NEVER GPKG on this mount). Generation side
(keep-out collage, exam v2, transplant) lives in run_generate_v04.py.
"""
import json, os, sys, shutil, time
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
V04 = os.path.join(TF, "v04")
CACHE = os.path.join(V04, "cache")
os.makedirs(CACHE, exist_ok=True)

DC = os.path.join(ROOT, "data_collection", "上海城市数据集")
SRC = {
    "road":  os.path.join(DC, "04-交通数据", "1-交通路网", "1-合集", "上海市_路网合集.shp"),
    "esa":   os.path.join(DC, "09-开源土地利用", "09-开源土地利用", "开源土地利用",
                          "10m", "SHP", "上海市_开源土地利用-ESA10m.shp"),
    "lu":    os.path.join(DC, "09-开源土地利用", "09-开源土地利用", "开源建设用地分类",
                          "Data", "其它-补充图层", "landuse-webmap", "上海市_Landuse.shp"),
    "euluc": os.path.join(DC, "09-开源土地利用", "09-开源土地利用", "开源建设用地分类",
                          "Data", "上海市-开源建设用地.shp"),
}
PAD = 0.0078
HALF = 500.0
CRS_M = "EPSG:32651"
CELL = 50.0
N = 20

# ---- road full widths (m) by OSM fclass; buffer = width/2.
# Footway/path/cycleway/steps/track/bridleway are EXCLUDED from the no-build
# mask: lilong lanes and compound paths are legitimately abutted (and overhung)
# by real buildings; including them would mass-flag real fabric (see calib).
ROAD_W = {
    "motorway": 30, "motorway_link": 18, "trunk": 30, "trunk_link": 18,
    "primary": 24, "primary_link": 14, "secondary": 18, "secondary_link": 12,
    "tertiary": 14, "tertiary_link": 10,
    "residential": 10, "unclassified": 10, "living_street": 8,
    "service": 6, "pedestrian": 6,
}
# Keep-out (sub-cell hard mask) uses only the MAJOR classes + water: the
# arterial network is public structural skeleton; minor streets/lanes are part
# of the fabric texture that pasted cell arrangements already carry. Calib
# showed the full mask flags 40% of real Laoximen buildings (>20% footprint
# in 10 m-buffered 'residential' lanes) — keep-out with it would shred old-city
# fabric. Cell-level void TYPING still uses ALL classes.
KEEPOUT_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link",
                   "primary", "primary_link", "secondary", "secondary_link",
                   "tertiary", "tertiary_link"}
ESA_WATER_CODE = 80   # ESA WorldCover standard: 80 = permanent water bodies.
                      # The shipped 类别 translation column is misaligned (A report
                      # gap #1 footnote); verified geometrically: Code-80 polygons
                      # cover the Huangpu at A01/A07 — see results 'water_check'.
GREEN_FCLASS = {"park", "grass", "forest", "meadow", "recreation_ground", "scrub",
                "heath", "nature_reserve", "orchard", "vineyard", "cemetery",
                "allotments"}        # farmland/farmyard stay E_VACANT (developable)
EULUC_GREEN = {"公园绿地"}

VOID_FRAC = 0.25     # cell void-typing threshold on mask-area fraction
VOID_PRIORITY = ["water", "road", "green"]   # checked in this order
TOKENS = ["E_VAC", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8",
          "E_ROAD", "E_WATER", "E_GREEN"]
NTOK = 12            # 0=E_VACANT, 1..8 built (v0.3 verbatim), 9/10/11 typed voids


def load_split():
    return json.load(open(os.path.join(V03, "split.json"), encoding="utf-8"))


def results_io(update=None):
    p = os.path.join(V04, "results_v04.json")
    res = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    if update:
        res.update(update)
        json.dump(res, open(p, "w"), ensure_ascii=False, indent=2)
    return res


# ================================================================ PREP (one citywide read per layer)
def prep():
    """The ESA layer (1.37M polygons) has no usable spatial index — per-window
    bbox reads take minutes each. Read each citywide layer ONCE, filter to the
    classes we need, reproject, cache as GeoParquet. Window masking then clips
    the caches (seconds)."""
    wp = os.path.join(CACHE, "water.parquet")
    if not os.path.exists(wp):
        t0 = time.time()
        e = gpd.read_file(SRC["esa"], columns=["Code"], where=f"Code = {ESA_WATER_CODE}")
        e = e.to_crs(CRS_M)
        e["geometry"] = e.geometry.buffer(0)
        e.to_parquet(wp)
        print(f"[prep] water: {len(e)} polys {time.time()-t0:.0f}s")
    rp = os.path.join(CACHE, "roads.parquet")
    if not os.path.exists(rp):
        t0 = time.time()
        r = gpd.read_file(SRC["road"], columns=["fclass"])
        r = r[r["fclass"].isin(ROAD_W)].to_crs(CRS_M)
        r["w"] = r["fclass"].map(ROAD_W)
        r.to_parquet(rp)
        print(f"[prep] roads: {len(r)} lines {time.time()-t0:.0f}s")
    gp = os.path.join(CACHE, "green.parquet")
    if not os.path.exists(gp):
        t0 = time.time()
        lu = gpd.read_file(SRC["lu"], columns=["fclass"])
        lu = lu[lu["fclass"].isin(GREEN_FCLASS)].to_crs(CRS_M)
        eu = gpd.read_file(SRC["euluc"], columns=["class2"])
        eu = eu[eu["class2"].isin(EULUC_GREEN)].to_crs(CRS_M)
        g = pd.concat([lu[["geometry"]], eu[["geometry"]]], ignore_index=True)
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=CRS_M)
        g["geometry"] = g.geometry.buffer(0)
        g.to_parquet(gp)
        print(f"[prep] green: {len(g)} polys {time.time()-t0:.0f}s")


# ================================================================ MASKS
def mask_one(w, layers):
    wid = w["id"]
    out = os.path.join(CACHE, f"{wid}_mask.parquet")
    if os.path.exists(out):
        print(f"[mask:{wid}] cached, skip")
        return
    t0 = time.time()
    sq = box(w["cx"] - HALF, w["cy"] - HALF, w["cx"] + HALF, w["cy"] + HALF)
    sq_pad = box(w["cx"] - HALF - 50, w["cy"] - HALF - 50,
                 w["cx"] + HALF + 50, w["cy"] + HALF + 50)
    parts = []

    r = layers["roads"]
    r = r.iloc[list(r.sindex.query(sq_pad, predicate="intersects"))]
    if len(r):
        geom = shapely.unary_union([g.buffer(hw / 2, cap_style="flat")
                                    for g, hw in zip(r.geometry, r["w"])])
        geom = geom.intersection(sq)
        if not geom.is_empty:
            parts.append(("road", geom))
        rm = r[r["fclass"].isin(KEEPOUT_CLASSES)]
        if len(rm):
            geom = shapely.unary_union([g.buffer(hw / 2, cap_style="flat")
                                        for g, hw in zip(rm.geometry, rm["w"])])
            geom = geom.intersection(sq)
            if not geom.is_empty:
                parts.append(("road_major", geom))

    for cls in ("water", "green"):
        g = layers[cls]
        g = g.iloc[list(g.sindex.query(sq, predicate="intersects"))]
        if len(g):
            geom = shapely.unary_union(g.geometry.values).intersection(sq)
            if not geom.is_empty:
                parts.append((cls, geom))

    gdf = gpd.GeoDataFrame({"cls": [p[0] for p in parts]},
                           geometry=[p[1] for p in parts], crs=CRS_M)
    gdf["area_m2"] = gdf.geometry.area
    gdf.to_parquet(out)
    fr = {c: round(float(a) / (2 * HALF) ** 2, 3) for c, a in zip(gdf["cls"], gdf["area_m2"])}
    print(f"[mask:{wid}] {time.time()-t0:.1f}s  window-area fractions {fr}")


def masks(ids):
    prep()
    split = load_split()
    wmap = {w["id"]: w for w in split["windows"]}
    layers = {"roads": gpd.read_parquet(os.path.join(CACHE, "roads.parquet")),
              "water": gpd.read_parquet(os.path.join(CACHE, "water.parquet")),
              "green": gpd.read_parquet(os.path.join(CACHE, "green.parquet"))}
    for wid in (ids or [w["id"] for w in split["windows"]]):
        mask_one(wmap[wid], layers)


def load_mask(wid, local=False, w=None):
    """Return dict cls->geometry (window coords; local=True shifts to 0..1000)."""
    g = gpd.read_parquet(os.path.join(CACHE, f"{wid}_mask.parquet"))
    out = dict(zip(g["cls"], g.geometry))
    if local:
        ox, oy = w["cx"] - HALF, w["cy"] - HALF
        out = {c: shapely.affinity.translate(geom, xoff=-ox, yoff=-oy)
               for c, geom in out.items()}
    return out


# ================================================================ CALIB
def calib():
    """Honesty check: how much REAL building footprint violates the mask?
    If real fabric 'violates' heavily, the mask is too aggressive for keep-out."""
    split = load_split()
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))
    rows = {}
    for w in split["windows"]:
        m = load_mask(w["id"])
        g = lib[lib["window"] == w["id"]]
        if not len(g):
            continue
        rec = {}
        hard_all = [m[c] for c in ("road", "water") if c in m]
        hard_all = shapely.unary_union(hard_all) if hard_all else None
        hard = [m[c] for c in ("road_major", "water") if c in m]
        hard = shapely.unary_union(hard) if hard else None
        for cls in ("road", "road_major", "water", "green", "HARD_allroads", "HARD"):
            geom = (hard_all if cls == "HARD_allroads"
                    else hard if cls == "HARD" else m.get(cls))
            if geom is None:
                rec[cls] = {"fp_share": 0.0, "bld_share_gt20": 0.0}
                continue
            inter = g.geometry.intersection(geom).area
            rec[cls] = {
                "fp_share": round(float(inter.sum() / g["fp_area"].sum()), 4),
                "bld_share_gt20": round(float((inter / g["fp_area"] > 0.2).mean()), 4),
            }
        rows[w["id"]] = {"name": w["name"], **rec}
        print(w["id"], w["name"],
              "KEEPOUT(major∪water) fp_share", rec["HARD"]["fp_share"],
              "bld>20%", rec["HARD"]["bld_share_gt20"],
              "| allroads bld>20%", rec["HARD_allroads"]["bld_share_gt20"])
    results_io({"mask_calibration": {
        "note": ("fp_share = share of real footprint area inside the mask class; "
                 "bld_share_gt20 = share of real buildings with >20% of their own "
                 "footprint inside it. HARD = road_major ∪ water = the keep-out "
                 "mask (calibrated: all-classes mask flags 40% of real Laoximen "
                 "buildings — minor lanes are fabric texture, not keep-out). "
                 "Green is typed at cell level only, never used for keep-out "
                 "(parks legitimately contain buildings)."),
        "road_widths_m": ROAD_W,
        "keepout_classes": sorted(KEEPOUT_CLASSES),
        "windows": rows,
    }})


# ================================================================ VOIDS
def voids():
    split = load_split()
    cells = pd.read_parquet(os.path.join(V03, "cache", "cells.parquet"))
    decomp, checks = {}, {}
    for w in split["windows"]:
        wid = w["id"]
        m = load_mask(wid)
        ox, oy = w["cx"] - HALF, w["cy"] - HALF
        g3 = np.load(os.path.join(V03, f"token_grid_{wid}.npy"))
        g4 = g3.copy()
        sub = cells[cells["window"] == wid].set_index(["row", "col"])
        n = {"E_VAC": 0, "E_ROAD": 0, "E_WATER": 0, "E_GREEN": 0}
        for r in range(N):
            for c in range(N):
                if g3[r, c] != 0:
                    continue
                cellbox = box(ox + c * CELL, oy + r * CELL,
                              ox + (c + 1) * CELL, oy + (r + 1) * CELL)
                fr = {cls: (m[cls].intersection(cellbox).area / CELL ** 2
                            if cls in m else 0.0)
                      for cls in ("water", "road", "green")}
                tok = 0
                for cls, tk in zip(VOID_PRIORITY, (10, 9, 11)):
                    if fr[cls] >= VOID_FRAC:
                        tok = tk
                        break
                g4[r, c] = tok
                n[{0: "E_VAC", 9: "E_ROAD", 10: "E_WATER", 11: "E_GREEN"}[tok]] += 1
        np.save(os.path.join(V04, f"token_grid_v04_{wid}.npy"), g4)
        n["T0_total"] = int((g3 == 0).sum())
        decomp[wid] = {"name": w["name"], "role": w["role"], **n}
        print(f"[voids:{wid}] T0 {n['T0_total']} -> vac {n['E_VAC']} road {n['E_ROAD']} "
              f"water {n['E_WATER']} green {n['E_GREEN']}")
    results_io({"void_typing": {
        "rule": (f"only T0 cells retyped; per-cell mask-area fraction; priority "
                 f"{VOID_PRIORITY} at threshold >= {VOID_FRAC}; else E_VACANT. "
                 "Built cells T1..T8 verbatim v0.3."),
        "tokens": TOKENS,
        "decomposition": decomp,
    }})


# ================================================================ ADJACENCY (TRAIN only)
def adjacency():
    split = load_split()
    A = np.zeros((NTOK, NTOK))
    for w in split["windows"]:
        if w["role"] != "train":
            continue
        g = np.load(os.path.join(V04, f"token_grid_v04_{w['id']}.npy"))
        for r in range(N):
            for c in range(N):
                for dr, dc in ((0, 1), (1, 0)):
                    rr, cc = r + dr, c + dc
                    if rr < N and cc < N:
                        a_, b_ = g[r, c], g[rr, cc]
                        A[a_, b_] += 1
                        A[b_, a_] += 1
    A_norm = A / A.sum(axis=1, keepdims=True).clip(min=1)
    json.dump({"tokens": TOKENS,
               "source": "TRAIN windows only (18 typed grids), 4-neighbour, pooled",
               "counts_pooled_4neighbour": A.astype(int).tolist(),
               "row_normalized": np.round(A_norm, 4).tolist()},
              open(os.path.join(V04, "adjacency_v04.json"), "w"), indent=2)
    # what the typed voids reveal: per built token, distribution over void types
    void_idx = {"E_VAC": 0, "E_ROAD": 9, "E_WATER": 10, "E_GREEN": 11}
    profile = {}
    for t in range(1, 9):
        tot = A[t].sum()
        profile[f"T{t}"] = {k: round(float(A[t, i] / tot), 4) if tot else 0.0
                            for k, i in void_idx.items()}
    results_io({"adjacency_v04": {
        "void_neighbour_profile_per_built_token": profile,
        "note": "row share of each built token's 4-neighbour contacts going to each void type",
    }})
    print(json.dumps(profile, indent=1))


# ================================================================ FIG 1
VOID_COLORS = {0: "#f5f0e6", 9: "#9e9e9e", 10: "#a8c8e8", 11: "#bcd9a0"}
TOKEN_COLORS = ["#f5f0e6", "#e9c46a", "#f4a261", "#e76f51", "#d62828",
                "#2a9d8f", "#457b9d", "#6a4c93", "#1d3557",
                "#9e9e9e", "#a8c8e8", "#bcd9a0"]


def fig1():
    split = load_split()
    res = results_io()
    decomp = res["void_typing"]["decomposition"]
    reps = ["A01", "A07", "A10", "A20", "A27", "A41"]   # river CBD / Bund / lilong /
    wmap = {w["id"]: w for w in split["windows"]}        # slab mat / new town / watertown
    fig = plt.figure(figsize=(17.5, 8.6))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.3, wspace=0.18)
    cmap = ListedColormap(TOKEN_COLORS)
    for k, wid in enumerate(reps):
        w = wmap[wid]
        ax = fig.add_subplot(gs[0, k])
        g4 = np.load(os.path.join(V04, f"token_grid_v04_{wid}.npy"))
        ax.imshow(g4, cmap=cmap, vmin=0, vmax=NTOK - 1, origin="lower",
                  extent=[0, 1000, 0, 1000])
        m = load_mask(wid, local=True, w=w)
        for cls, col in (("road", "#555555"), ("water", "#2c6fad"), ("green", "#4d7c3a")):
            if cls in m:
                geom = m[cls]
                geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                for p in geoms:
                    if p.geom_type != "Polygon" or p.area < 50:
                        continue
                    xs, ys = p.exterior.xy
                    ax.plot(xs, ys, color=col, lw=0.5, alpha=0.85)
        ax.set_xticks([]); ax.set_yticks([])
        d = decomp[wid]
        ax.set_title(f"{w['name']}\nT0 {d['T0_total']} -> R{d['E_ROAD']} W{d['E_WATER']} "
                     f"G{d['E_GREEN']} V{d['E_VAC']}", fontsize=8.6)
    # bottom: stacked decomposition bars, all 24
    ax = fig.add_subplot(gs[1, :])
    ids = [w["id"] for w in split["windows"]]
    names = [f"{wmap[i]['name']}{'*' if wmap[i]['role']=='heldout' else ''}" for i in ids]
    bottoms = np.zeros(len(ids))
    for key, col, lab in (("E_ROAD", "#9e9e9e", "E_ROAD"), ("E_WATER", "#a8c8e8", "E_WATER"),
                          ("E_GREEN", "#bcd9a0", "E_GREEN"), ("E_VAC", "#f5f0e6", "E_VACANT")):
        vals = np.array([decomp[i][key] for i in ids], dtype=float)
        ax.bar(range(len(ids)), vals, bottom=bottoms, color=col,
               edgecolor="gray", linewidth=0.3, label=lab)
        bottoms += vals
    ax.set_xticks(range(len(ids)), names, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("former T0 EMPTY cells (of 400)")
    ax.legend(fontsize=8.5, ncol=4)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("T0 decomposition per window (* = heldout) — what the blank actually was",
                 fontsize=10.5)
    fig.suptitle("v0.4 FIG 1 — Typed voids: the blank is not buildable by default "
                 "(mask outlines: road gray / water blue / green green)", fontsize=13)
    fig.savefig(os.path.join(V04, "fig1_typed_voids.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig1_typed_voids.png written")


# ================================================================ FIG 2
def fig2():
    adj4 = json.load(open(os.path.join(V04, "adjacency_v04.json")))
    adj3 = json.load(open(os.path.join(V03, "adjacency.json")))
    A4 = np.array(adj4["row_normalized"])
    A3 = np.array(adj3["row_normalized"])
    res = results_io()
    profile = res["adjacency_v04"]["void_neighbour_profile_per_built_token"]

    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(A3, cmap="YlGnBu", vmin=0, vmax=max(A3.max(), A4.max()))
    ax.set_xticks(range(9), [f"T{t}" for t in range(9)], fontsize=8)
    ax.set_yticks(range(9), [f"T{t}" for t in range(9)], fontsize=8)
    ax.set_title("v0.3 adjacency (9 tokens)\nall voids = one T0", fontsize=10)
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(A4, cmap="YlGnBu", vmin=0, vmax=max(A3.max(), A4.max()))
    ax.set_xticks(range(NTOK), adj4["tokens"], fontsize=7.5, rotation=45, ha="right")
    ax.set_yticks(range(NTOK), adj4["tokens"], fontsize=7.5)
    for i in range(NTOK):
        for j in range(NTOK):
            v = A4[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center",
                        fontsize=5.8, color="white" if v > 0.5 * A4.max() else "black")
    ax.set_title("v0.4 adjacency (12 tokens)\nTRAIN only, row-normalized", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax = fig.add_subplot(gs[2])
    bt = [f"T{t}" for t in range(1, 9)]
    x = np.arange(len(bt))
    bottoms = np.zeros(len(bt))
    for key, col in (("E_ROAD", "#9e9e9e"), ("E_WATER", "#a8c8e8"),
                     ("E_GREEN", "#bcd9a0"), ("E_VAC", "#f5f0e6")):
        vals = np.array([profile[t][key] for t in bt])
        ax.bar(x, vals, bottom=bottoms, color=col, edgecolor="gray", lw=0.4,
               label=key)
        bottoms += vals
    ax.set_xticks(x, bt, fontsize=9)
    ax.set_ylabel("share of the built token's neighbour contacts", fontsize=9)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Which void does each built token face?\n(per-row contact share)",
                 fontsize=10)
    fig.suptitle("v0.4 FIG 2 — Splitting T0 reveals structure one EMPTY token averaged away",
                 fontsize=13)
    fig.savefig(os.path.join(V04, "fig2_adjacency.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig2_adjacency.png written")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "prep":
        prep()
    elif cmd == "masks":
        masks(sys.argv[2:])
    elif cmd == "calib":
        calib()
    elif cmd == "voids":
        voids()
    elif cmd == "adjacency":
        adjacency()
    elif cmd == "fig1":
        fig1()
    elif cmd == "fig2":
        fig2()
    else:
        raise SystemExit(f"unknown cmd {cmd}")
