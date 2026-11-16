#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generate_v02.py — G2 generator for the v0.2 token grammar (Agent 2).

Pipeline (each subcommand caches intermediates so every bash call stays short):
  grow       : per patch, real token composition -> 20x20 grid, exact-multiset random
               init, then 5 random-order Gibbs sweeps. P(token | 4 neighbours) ∝
               adjacency pseudo-likelihood x composition prior. The composition prior
               carries a feedback term target/current so the sweeps cannot drift the
               composition away (plain Gibbs bled T5/T6 into T0 — the exact failure
               mode that sank v0.1). RANDOM cell order kills raster-scan bias.
  collage    : instantiation = CELL COLLAGE. Each non-EMPTY cell samples a real
               source cell of the same token (pools restricted to the same patch =
               the patch's "signature") and copies the whole arrangement (cell-local
               footprints + heights). OVERHANG DECISION: footprints that overhang
               the 50 m cell are kept as-is (clipping at cell lines would shred row
               housing); we clip only at the 1 km canvas edge and drop slivers <1 m².
  compare    : probe-style 10-indicator z-scored distance matrix (3 gen x 3 real),
               hit rate vs v0.1's 1/3 -> results_gen_v02.json.
  fig3       : real vs self-generated 3D axons + distance matrix + z-bars.
  transplant : signature A (composition + A-only pools) on site B (B's real grid;
               B's EMPTY cells stay empty; B's built cells ordered by real GFA
               receive A's token demand ordered by token intensity = pool-mean GFA).
               If A is sparser than B, surplus built cells of B go EMPTY (sparse-tall
               regime keeps its voids); if A is denser, demand is compressed
               proportionally to B's skeleton.
  fig4       : 3x3 transplant axons (rows = signature, cols = site).
  temporal   : Laoximen skeleton; composition interpolated in 5 steps from
               Laoximen's own signature to a developer-regime PROXY (= Lujiazui's
               signature); pools blended by the interpolation weight.
  fig5       : 5 axons + FAR / h_p90 / tower-share curves. Illustrative only.

seed 0 everywhere (each subcommand reseeds default_rng(0)). Vectors are written as
FlatGeobuf via /tmp (never GPKG — sqlite breaks on this mount). Figures dpi 130,
English-only labels.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.affinity import translate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = "/sessions/epic-peaceful-feynman/mnt/Urban-Stakeholder-Workshop"
if not os.path.isdir(ROOT):
    ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
V02 = os.path.join(ROOT, "typology_foundation", "v02")
CACHE = os.path.join(V02, "cache")
os.makedirs(CACHE, exist_ok=True)

PATCHES = ["lujiazui", "laoximen", "caoyang"]
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
SEED = 0
SWEEPS = 5
EPS = 0.01          # smoothing for zero adjacency entries (e.g. T1-T6 never observed)
BETA = 8.0          # composition-feedback exponent; beta=1 let counts drift badly
                    # (laoximen T0 33->4, T3 133->205) because the 4-neighbour
                    # likelihood product dominates a flat share prior; beta=8 keeps
                    # counts within ~12% of target while preserving clustering
CELL = 50.0
CANVAS = 1000.0
AREA = 1_000_000.0
N = 20
HCAP = 150.0        # shared viridis cap (p99 of heights = 126 m); zlim 0-350 (SWFC 338 m)
ZMAX = 350.0
ELEV, AZIM = 38, -60  # one camera for every axon


# ---------------------------------------------------------------- assets
def load_tokens():
    return json.load(open(os.path.join(V02, "tokens.json"), encoding="utf-8"))


def load_adjacency():
    adj = json.load(open(os.path.join(V02, "adjacency.json"), encoding="utf-8"))
    return np.array(adj["row_normalized"], dtype=float)


def load_grids():
    return {p: np.load(os.path.join(V02, f"token_grid_{p}.npy")) for p in PATCHES}


def token_intensity():
    """Token 'intensity' for rank-matching = mean cell GFA of the token (tokens.json)."""
    tk = load_tokens()
    inten = {0: 0.0}
    for t in range(1, 7):
        inten[t] = float(tk["stats"][str(t)]["gfa"])
    return inten


def load_pools_geo():
    g = gpd.read_file(os.path.join(V02, "token_pools.fgb"))
    g["key"] = g["patch"] + "_r" + g["row"].astype(str) + "_c" + g["col"].astype(str)
    return g


def cell_gfa_table():
    tp = pd.read_parquet(os.path.join(V02, "token_pools.parquet"))
    tp["gfa"] = tp["fp_area"] * tp["floors"]
    return tp.groupby(["patch", "row", "col"])["gfa"].sum().to_dict()


# ---------------------------------------------------------------- G2a growth
def grow_grid(target_counts, A, rng, sweeps=SWEEPS):
    """Exact-multiset random init, then `sweeps` random-order Gibbs sweeps.
    P(t | nbrs) ∝ prod_j A[t, nbr_j] * share_t * (target_t / current_t).
    The feedback factor is the 'composition prior' made operational: without it,
    rare tokens (T5/T6, prior ~0.02) are eaten by T0's strong diagonal."""
    target = np.asarray(target_counts, dtype=float)
    seq = np.repeat(np.arange(7), target_counts)
    rng.shuffle(seq)
    grid = seq.reshape(N, N).copy()
    Ae = A + EPS
    share = target / target.sum()
    counts = np.bincount(grid.ravel(), minlength=7).astype(float)
    for _ in range(sweeps):
        for idx in rng.permutation(N * N):   # random order: no raster-scan bias
            r, c = divmod(idx, N)
            like = np.ones(7)
            if r > 0:
                like *= Ae[:, grid[r - 1, c]]
            if r < N - 1:
                like *= Ae[:, grid[r + 1, c]]
            if c > 0:
                like *= Ae[:, grid[r, c - 1]]
            if c < N - 1:
                like *= Ae[:, grid[r, c + 1]]
            old = grid[r, c]
            counts[old] -= 1
            prior = share * (target / np.maximum(counts, 0.5)) ** BETA
            p = like * prior
            p /= p.sum()
            new = int(rng.choice(7, p=p))
            grid[r, c] = new
            counts[new] += 1
    return grid


def cmd_grow():
    rng = np.random.default_rng(SEED)
    A = load_adjacency()
    grids = load_grids()
    log = {}
    for p in PATCHES:
        target = np.bincount(grids[p].ravel(), minlength=7)
        g = grow_grid(target, A, rng)
        np.save(os.path.join(CACHE, f"gen_grid_{p}.npy"), g)
        ach = np.bincount(g.ravel(), minlength=7)
        log[p] = {"target": target.tolist(), "achieved": ach.tolist()}
        print(p, "target", target.tolist(), "achieved", ach.tolist())
    json.dump(log, open(os.path.join(CACHE, "grow_log.json"), "w"), indent=2)


# ---------------------------------------------------------------- G2b collage
def build_cell_index(pools):
    """(patch, token) -> list of source-cell keys; key -> row indices."""
    cells = pools.drop_duplicates(subset="key")
    by_pt = cells.groupby(["patch", "token"])["key"].apply(list).to_dict()
    by_t = cells.groupby("token")["key"].apply(list).to_dict()
    members = {k: v for k, v in pools.groupby("key").groups.items()}
    return by_pt, by_t, members


def collage(grid, pools, idx, rng, patch_chooser):
    """Instantiate a token grid. patch_chooser(rng) -> source patch name."""
    by_pt, by_t, members = idx
    rows = []
    clipbox = box(0, 0, CANVAS, CANVAS)
    for r in range(N):
        for c in range(N):
            t = int(grid[r, c])
            if t == 0:
                continue
            p = patch_chooser(rng)
            keys = by_pt.get((p, t)) or by_t[t]   # fallback: global pool
            key = keys[int(rng.integers(len(keys)))]
            sub = pools.loc[members[key]]
            for _, b in sub.iterrows():
                geom = translate(b.geometry, xoff=c * CELL, yoff=r * CELL)
                geom = geom.intersection(clipbox)   # clip at canvas edge only
                if geom.is_empty or geom.area < 1.0:
                    continue
                rows.append(
                    dict(geometry=geom, height=float(b.height), floors=int(b.floors),
                         family=b.family, semantic=b.semantic, token=t,
                         src_patch=b.patch, src_key=key, tgt_row=r, tgt_col=c))
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32651")
    gdf["fp_area"] = gdf.geometry.area
    gdf["gfa"] = gdf["fp_area"] * gdf["floors"]
    return gdf


def fixed(p):
    return lambda rng: p


def save_canvas(gdf, name, fgb=False):
    gdf.to_parquet(os.path.join(CACHE, f"{name}.parquet"))
    if fgb:
        tmp = os.path.join("/tmp", f"{name}.fgb")
        gdf.to_file(tmp, driver="FlatGeobuf")
        shutil.copy2(tmp, os.path.join(V02, f"{name}.fgb"))


def real_canvas(p):
    """Real patch in local 0-1000 coords (cached)."""
    fp = os.path.join(CACHE, f"real_canvas_{p}.parquet")
    if os.path.exists(fp):
        return gpd.read_parquet(fp)
    g = gpd.read_file(os.path.join(V02, "fused_v02.fgb"), where=f"patch='{p}'")
    minx, miny = g.total_bounds[0], g.total_bounds[1]
    g["geometry"] = g.geometry.apply(lambda gm: translate(gm, xoff=-minx, yoff=-miny))
    g.to_parquet(fp)
    return g


def cmd_collage():
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    for p in PATCHES:
        grid = np.load(os.path.join(CACHE, f"gen_grid_{p}.npy"))
        gdf = collage(grid, pools, idx, rng, fixed(p))
        save_canvas(gdf, f"generated_self_{p}", fgb=True)
        print(p, len(gdf), "buildings, FAR", round(gdf.gfa.sum() / AREA, 2))
        real_canvas(p)  # warm the cache for figures


# ---------------------------------------------------------------- indicators
def indicators(df):
    row = {
        "FAR": float((df["fp_area"] * df["floors"]).sum() / AREA),
        "coverage": float(df["fp_area"].sum() / AREA),
        "height_p50": float(np.percentile(df["height"], 50)),
        "height_p90": float(np.percentile(df["height"], 90)),
        "median_fp_area": float(df["fp_area"].median()),
    }
    sh = df["family"].value_counts(normalize=True)
    for f in FAMILIES:
        row[f"share_{f}"] = float(sh.get(f, 0.0))
    return row


def cmd_compare():
    fused = pd.read_parquet(os.path.join(V02, "fused_v02.parquet"))
    rows = {}
    for p in PATCHES:
        gen = gpd.read_parquet(os.path.join(CACHE, f"generated_self_{p}.parquet"))
        rows[f"gen_{p}"] = indicators(gen)
        rows[f"real_{p}"] = indicators(fused[fused["patch"] == p])
    ind = pd.DataFrame(rows).T
    z = (ind - ind.mean()) / ind.std(ddof=0).replace(0, 1)
    D = pd.DataFrame(index=PATCHES, columns=PATCHES, dtype=float)
    for a in PATCHES:
        for b in PATCHES:
            D.loc[a, b] = float(np.linalg.norm(z.loc[f"gen_{a}"] - z.loc[f"real_{b}"]))
    nearest = {a: D.loc[a].idxmin() for a in PATCHES}
    hits = {a: bool(nearest[a] == a) for a in PATCHES}
    grow_log = json.load(open(os.path.join(CACHE, "grow_log.json")))
    res = {
        "growth": {
            "sweeps": SWEEPS, "seed": SEED, "eps": EPS,
            "note": ("random-order Gibbs (kills raster-scan bias); composition prior "
                     "includes target/current feedback so token counts cannot drift"),
            "composition_target_vs_achieved": grow_log,
        },
        "collage": {
            "rule": ("each non-EMPTY cell copies one real source cell of the same token, "
                     "pools restricted to the same patch (signature); overhang beyond the "
                     "50 m cell kept, clipped only at the 1 km canvas edge"),
        },
        "compare_v2": {
            "indicators": ind.round(4).to_dict(orient="index"),
            "distance_matrix_zscored_euclidean": D.round(3).to_dict(orient="index"),
            "nearest": nearest, "hit": hits, "n_hits": int(sum(hits.values())),
            "v01_baseline": {"n_hits": 1, "note": "v0.1 probe: 1/3, all generations "
                             "attracted to laoximen (placer saturated at FAR 1.4-1.7)"},
        },
    }
    json.dump(res, open(os.path.join(V02, "results_gen_v02.json"), "w"),
              ensure_ascii=False, indent=2)
    print(D.round(2).to_string())
    print("hits:", res["compare_v2"]["n_hits"], "/3  (v0.1: 1/3)")


# ---------------------------------------------------------------- 3D axon
def render_axon(ax, gdf, simplify=1.2):
    cmap = cm.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=HCAP)
    polys, cols = [], []
    for geom, h in zip(gdf.geometry, gdf["height"]):
        if geom is None or geom.is_empty:
            continue
        parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        base = np.array(cmap(norm(min(h, HCAP))))
        side = base.copy()
        side[:3] *= 0.72
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            p2 = part.simplify(simplify, preserve_topology=True)
            if p2.is_empty or p2.geom_type != "Polygon":
                p2 = part
            xy = np.asarray(p2.exterior.coords)
            polys.append([(x, y, h) for x, y in xy])
            cols.append(tuple(base))
            for i in range(len(xy) - 1):
                (x1, y1), (x2, y2) = xy[i], xy[i + 1]
                polys.append([(x1, y1, 0), (x2, y2, 0), (x2, y2, h), (x1, y1, h)])
                cols.append(tuple(side))
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors="none")
    ax.add_collection3d(pc)
    ax.set_xlim(0, CANVAS); ax.set_ylim(0, CANVAS); ax.set_zlim(0, ZMAX)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def cmd_fig3():
    res = json.load(open(os.path.join(V02, "results_gen_v02.json"), encoding="utf-8"))
    ind = pd.DataFrame(res["compare_v2"]["indicators"]).T
    D = pd.DataFrame(res["compare_v2"]["distance_matrix_zscored_euclidean"]).loc[PATCHES, PATCHES]
    nearest, hits = res["compare_v2"]["nearest"], res["compare_v2"]["hit"]

    fig = plt.figure(figsize=(15.5, 14.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 1.05, 0.85], hspace=0.18, wspace=0.02)
    for j, p in enumerate(PATCHES):
        axr = fig.add_subplot(gs[0, j], projection="3d")
        render_axon(axr, real_canvas(p))
        ir = ind.loc[f"real_{p}"]
        axr.set_title(f"REAL {p}\nFAR {ir.FAR:.2f}  cov {ir.coverage:.2f}  "
                      f"h50 {ir.height_p50:.0f}m  h90 {ir.height_p90:.0f}m",
                      fontsize=9.5)
        axg = fig.add_subplot(gs[1, j], projection="3d")
        gen = gpd.read_parquet(os.path.join(CACHE, f"generated_self_{p}.parquet"))
        render_axon(axg, gen)
        ig = ind.loc[f"gen_{p}"]
        axg.set_title(f"GENERATED {p} (grow + cell collage)\n"
                      f"FAR {ig.FAR:.2f} (real {ir.FAR:.2f})  cov {ig.coverage:.2f} ({ir.coverage:.2f})\n"
                      f"h50 {ig.height_p50:.0f}m ({ir.height_p50:.0f})  "
                      f"h90 {ig.height_p90:.0f}m ({ir.height_p90:.0f})", fontsize=9)

    ax1 = fig.add_subplot(gs[2, 0])
    M = D.values.astype(float)
    im = ax1.imshow(M, cmap="RdYlGn_r")
    ax1.set_xticks(range(3), [f"real {p}" for p in PATCHES], fontsize=8)
    ax1.set_yticks(range(3), [f"gen {p}" for p in PATCHES], fontsize=8)
    for i, a in enumerate(PATCHES):
        for j, b in enumerate(PATCHES):
            star = " *HIT*" if (nearest[a] == b and hits[a]) else (
                " *nearest*" if nearest[a] == b else "")
            ax1.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center",
                     fontsize=8, fontweight="bold" if nearest[a] == b else "normal")
    n_hits = res["compare_v2"]["n_hits"]
    ax1.set_title(f"Compare v2: z-scored 10-indicator distance\n"
                  f"hits {n_hits}/3  (v0.1 baseline: 1/3, all pulled to laoximen)",
                  fontsize=9.5)
    fig.colorbar(im, ax=ax1, shrink=0.75)

    ax2 = fig.add_subplot(gs[2, 1:])
    z = (ind - ind.mean()) / ind.std(ddof=0).replace(0, 1)
    cols_ = list(ind.columns)
    x = np.arange(len(cols_))
    w = 0.13
    palette = {"gen_lujiazui": "#1d3557", "real_lujiazui": "#a8dadc",
               "gen_laoximen": "#9d0208", "real_laoximen": "#e76f51",
               "gen_caoyang": "#1b4332", "real_caoyang": "#2a9d8f"}
    order = [f"{k}_{p}" for p in PATCHES for k in ("gen", "real")]
    for k, name in enumerate(order):
        ax2.bar(x + (k - 2.5) * w, z.loc[name], w, label=name, color=palette[name],
                hatch="" if name.startswith("gen") else "..",
                edgecolor="white", linewidth=0.3)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.set_xticks(x, [c.replace("share_", "sh.") for c in cols_],
                   rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("z-score over 6 fabrics", fontsize=9)
    ax2.set_title("Indicator vectors (dotted = real)", fontsize=9.5)
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(axis="y", alpha=0.3)
    fig.suptitle("FIG3 — Self-generation x3 (token growth + cell collage) and Compare v2",
                 fontsize=14, y=0.985)
    fig.savefig(os.path.join(V02, "fig3_growth3d.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


# ---------------------------------------------------------------- FIG4 transplant
def largest_remainder(frac, total):
    base = np.floor(frac).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(frac - base))
        base[order[:rem]] += 1
    return base


def rank_match(site_grid, demand_counts, cgfa, site_patch, inten):
    """Assign demand (counts over T0..T6, summing 400) onto site's built cells.
    Site EMPTY cells stay empty. Built cells sorted by real GFA desc receive
    tokens sorted by intensity desc; surplus EMPTY demand pads the tail."""
    nonempty = [(r, c) for r in range(N) for c in range(N) if site_grid[r, c] > 0]
    nB = len(nonempty)
    d = np.asarray(demand_counts, dtype=float)
    built = d[1:]
    n_built = built.sum()
    if n_built > nB:                      # denser signature than the skeleton allows
        built = largest_remainder(built * nB / n_built, nB)
    else:                                 # sparser: keep absolute counts, pad EMPTY
        built = largest_remainder(built, int(round(n_built)))
    seq = []
    for t in sorted(range(1, 7), key=lambda t: -inten[t]):
        seq += [t] * int(built[t - 1])
    seq += [0] * (nB - len(seq))
    order = sorted(nonempty, key=lambda rc: -cgfa.get((site_patch, rc[0], rc[1]), 0.0))
    out = np.zeros_like(site_grid)
    for (r, c), t in zip(order, seq):
        out[r, c] = t
    return out


def cmd_transplant():
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    grids = load_grids()
    cgfa = cell_gfa_table()
    inten = token_intensity()
    out = {}
    for a in PATCHES:                     # signature
        demand = np.bincount(grids[a].ravel(), minlength=7)
        for b in PATCHES:                 # site
            g = rank_match(grids[b], demand, cgfa, b, inten)
            gdf = collage(g, pools, idx, rng, fixed(a))
            name = f"transplant_{a}_on_{b}"
            save_canvas(gdf, name)
            out[name] = {
                "FAR": round(float(gdf.gfa.sum() / AREA), 3),
                "coverage": round(float(gdf.fp_area.sum() / AREA), 3),
                "height_p90": round(float(np.percentile(gdf.height, 90)), 1),
                "n_buildings": int(len(gdf)),
                "n_built_cells": int((g > 0).sum()),
            }
            print(name, out[name])
    res = json.load(open(os.path.join(V02, "results_gen_v02.json"), encoding="utf-8"))
    res["transplant_3x3"] = {
        "rule": ("site keeps its real grid; EMPTY stays empty; built cells ranked by real "
                 "GFA receive signature demand ranked by token intensity; sparse signature "
                 "pads surplus cells with EMPTY, dense signature is compressed to skeleton; "
                 "pools restricted to signature patch"),
        "cells": out,
    }
    json.dump(res, open(os.path.join(V02, "results_gen_v02.json"), "w"),
              ensure_ascii=False, indent=2)


def cmd_fig4():
    res = json.load(open(os.path.join(V02, "results_gen_v02.json"), encoding="utf-8"))
    cells = res["transplant_3x3"]["cells"]
    fig = plt.figure(figsize=(13.5, 13.5))
    gs = fig.add_gridspec(3, 3, hspace=0.16, wspace=0.02)
    for i, a in enumerate(PATCHES):
        for j, b in enumerate(PATCHES):
            ax = fig.add_subplot(gs[i, j], projection="3d")
            gdf = gpd.read_parquet(os.path.join(CACHE, f"transplant_{a}_on_{b}.parquet"))
            render_axon(ax, gdf, simplify=1.8)
            st = cells[f"transplant_{a}_on_{b}"]
            tag = "  [self]" if a == b else ""
            ax.set_title(f"{a} signature on {b} site{tag}\n"
                         f"FAR {st['FAR']:.2f}   h90 {st['height_p90']:.0f} m",
                         fontsize=9.5)
    fig.suptitle("FIG4 — Signature transplant 3x3 (rows = signature: composition + pools; "
                 "cols = site skeleton; rank-matched by cell GFA)", fontsize=13, y=0.97)
    fig.text(0.5, 0.015, "Counterfactual collage on real skeletons — illustrative, not a development forecast.",
             ha="center", fontsize=9, style="italic")
    fig.savefig(os.path.join(V02, "fig4_transplant3x3.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig4 saved")


# ---------------------------------------------------------------- FIG5 temporal
W_STEPS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cmd_temporal():
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    grids = load_grids()
    cgfa = cell_gfa_table()
    inten = token_intensity()
    c_lao = np.bincount(grids["laoximen"].ravel(), minlength=7).astype(float)
    c_ljz = np.bincount(grids["lujiazui"].ravel(), minlength=7).astype(float)
    curves = []
    for k, w in enumerate(W_STEPS):
        demand = largest_remainder((1 - w) * c_lao + w * c_ljz, N * N)
        g = rank_match(grids["laoximen"], demand, cgfa, "laoximen", inten)

        def chooser(rng, w=w):
            return "lujiazui" if rng.random() < w else "laoximen"

        gdf = collage(g, pools, idx, rng, chooser)
        save_canvas(gdf, f"temporal_step{k}")
        tower = float((gdf.family == "tower").mean())
        curves.append({
            "w": w, "FAR": round(float(gdf.gfa.sum() / AREA), 3),
            "coverage": round(float(gdf.fp_area.sum() / AREA), 3),
            "height_p90": round(float(np.percentile(gdf.height, 90)), 1),
            "tower_share": round(tower, 3), "n_buildings": int(len(gdf)),
            "n_built_cells": int((g > 0).sum()),
        })
        print(curves[-1])
    res = json.load(open(os.path.join(V02, "results_gen_v02.json"), encoding="utf-8"))
    res["temporal_laoximen"] = {
        "rule": ("Laoximen skeleton; composition interpolated (1-w)*laoximen + w*lujiazui; "
                 "lujiazui = developer-regime PROXY, not a calibrated regime; pools blended "
                 "per cell with P(lujiazui)=w; illustrative, not predictive"),
        "steps": curves,
    }
    json.dump(res, open(os.path.join(V02, "results_gen_v02.json"), "w"),
              ensure_ascii=False, indent=2)


def cmd_fig5():
    res = json.load(open(os.path.join(V02, "results_gen_v02.json"), encoding="utf-8"))
    curves = res["temporal_laoximen"]["steps"]
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.45, 0.7], hspace=0.18, wspace=0.02)
    for k, st in enumerate(curves):
        ax = fig.add_subplot(gs[0, k], projection="3d")
        gdf = gpd.read_parquet(os.path.join(CACHE, f"temporal_step{k}.parquet"))
        render_axon(ax, gdf, simplify=1.8)
        lab = ("Laoximen signature" if st["w"] == 0 else
               "developer-regime proxy\n(= Lujiazui signature)" if st["w"] == 1 else
               f"w = {st['w']:.2f}")
        ax.set_title(f"step {k}  {lab}\nFAR {st['FAR']:.2f}  h90 {st['height_p90']:.0f} m",
                     fontsize=9)
    ws = [st["w"] for st in curves]
    axc = fig.add_subplot(gs[1, 0:2])
    axc.plot(ws, [st["FAR"] for st in curves], "o-", color="#1d3557", label="FAR")
    axc.plot(ws, [st["coverage"] for st in curves], "s--", color="#e76f51", label="coverage")
    axc.set_xlabel("interpolation weight w"); axc.legend(fontsize=8); axc.grid(alpha=0.3)
    axc.set_title("FAR / coverage", fontsize=9.5)
    axh = fig.add_subplot(gs[1, 2])
    axh.plot(ws, [st["height_p90"] for st in curves], "o-", color="#2a9d8f")
    axh.set_xlabel("w"); axh.grid(alpha=0.3); axh.set_title("height p90 (m)", fontsize=9.5)
    axt = fig.add_subplot(gs[1, 3])
    axt.plot(ws, [st["tower_share"] for st in curves], "o-", color="#9d0208")
    axt.set_xlabel("w"); axt.grid(alpha=0.3); axt.set_title("tower share (count)", fontsize=9.5)
    axn = fig.add_subplot(gs[1, 4])
    axn.plot(ws, [st["n_built_cells"] for st in curves], "o-", color="#6a4c93")
    axn.set_xlabel("w"); axn.grid(alpha=0.3); axn.set_title("built cells / 400", fontsize=9.5)
    fig.suptitle("FIG5 — Laoximen redevelopment path: signature interpolated towards a "
                 "developer-regime proxy (= Lujiazui)", fontsize=13, y=0.99)
    fig.text(0.5, 0.012, "ILLUSTRATIVE, NOT PREDICTIVE — proxy regime, single seed, no tenure / street / parcel constraints.",
             ha="center", fontsize=10, style="italic", fontweight="bold")
    fig.savefig(os.path.join(V02, "fig5_temporal.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig5 saved")


# ---------------------------------------------------------------- main
CMDS = {"grow": cmd_grow, "collage": cmd_collage, "compare": cmd_compare,
        "fig3": cmd_fig3, "transplant": cmd_transplant, "fig4": cmd_fig4,
        "temporal": cmd_temporal, "fig5": cmd_fig5}

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        if arg == "all":
            for f in CMDS.values():
                f()
        else:
            CMDS[arg]()
