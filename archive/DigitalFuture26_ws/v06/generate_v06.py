#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_v06.py — D3: generate the five doctrine cities and SHOW them.

For each of the 5 paradigms (garden_city, ville_radieuse, neighbourhood_unit,
microrayon, gb50180):
  UNIT  (1 km, 20x20): D2 token budget placed on the D2 skeleton by the
        doctrine's PRESCRIPTIVE geometry first; where the doctrine is silent on
        micro-arrangement, 3 random-order multiset-preserving Metropolis swap
        sweeps with the v0.4 12-token TRAIN adjacency ("Shanghai accent").
        Then SEMANTIC-CONDITIONED CELL COLLAGE from TF/v03 token pools: prefer
        pool cells whose GFA-dominant building semantic matches the demanded
        semantic; fall back to any same-token cell and COUNT the fallbacks
        (v05's named next step 語義條件化的生成).
  CITY  (3 km, 60x60): composite per the doctrine's city-level diagram
        (assembly rules documented in results_d3.json).
  FIGS  fig_d3_<p>_unit.png (plan + axon), fig_d3_<p>_city.png (plan),
        fig_d3_gallery.png (5 cols x 2 rows: plan / axon).

seed 0. No GPKG. Canvases -> V06/cache_gen/{unit,city}_<p>.parquet.
render_axon / make_tile / show_tile / indicators / load_pools_geo /
build_cell_index copied verbatim from TF/v04/run_generate_v04.py (provenance
comments at each).
"""
import json, math, os, textwrap
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03, V04, V06 = (os.path.join(TF, v) for v in ("v03", "v04", "v06"))
FIG = os.path.join(V06, "fig")
CACHE = os.path.join(V06, "cache_gen")
os.makedirs(FIG, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

SEED = 0
CELL = 50.0
EPS = 0.01
HCAP, ZMAX = 150.0, 350.0
ELEV, AZIM = 38, -60
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
PARADIGMS = ["garden_city", "ville_radieuse", "neighbourhood_unit",
             "microrayon", "gb50180"]
META = {  # name/year from V06/doctrine_recipes.json meta blocks
    "garden_city": ("Garden City (Howard)", "1898/1902"),
    "ville_radieuse": ("Ville Radieuse (Le Corbusier)", "1933/1935"),
    "neighbourhood_unit": ("Neighbourhood Unit (Perry)", "1929"),
    "microrayon": ("Microrayon (SNiP)", "1958-89"),
    "gb50180": ("GB 50180 (PRC code)", "2018"),
}
T_GREEN, T_ROAD, T_VAC, T_CIVIC = 11, 9, 0, -1


# ------------------------------------------------------------------ assets
def load_pools_geo():   # verbatim from TF/v04/run_generate_v04.py::load_pools_geo
    g = gpd.read_file(os.path.join(V03, "token_pools.fgb"))
    g["key"] = g["window"] + "_r" + g["row"].astype(str) + "_c" + g["col"].astype(str)
    return g


def build_cell_index(pools):  # verbatim from TF/v04/run_generate_v04.py
    cells = pools.drop_duplicates(subset="key")
    by_wt = cells.groupby(["window", "token"])["key"].apply(list).to_dict()
    by_t = cells.groupby("token")["key"].apply(list).to_dict()
    members = {k: v for k, v in pools.groupby("key").groups.items()}
    return by_wt, by_t, members


def build_semantic_index(pools):
    """NEW for v0.6: per pool cell the GFA-dominant building semantic; index
    (token, dominant_semantic) -> [keys]. This is the semantic-conditioned
    pool of v05's named next step."""
    p = pools.copy()
    p["gfa"] = p["fp_area"] * p["floors"]
    dom = (p.groupby(["key", "semantic"])["gfa"].sum().reset_index()
             .sort_values("gfa").groupby("key").last()["semantic"])
    cells = p.drop_duplicates(subset="key").set_index("key")
    by_ts = {}
    for key, sem in dom.items():
        t = int(cells.loc[key, "token"])
        by_ts.setdefault((t, sem), []).append(key)
    avail = {f"T{t}|{s}": len(v) for (t, s), v in sorted(by_ts.items())}
    return by_ts, avail


def load_adjacency12():
    adj = json.load(open(os.path.join(V04, "adjacency_v04.json"), encoding="utf-8"))
    return np.array(adj["row_normalized"], dtype=float)


# ------------------------------------------------------------------ Gibbs (Shanghai accent)
GIBBS_LABEL = ("Shanghai accent: 3 random-order multiset-preserving Metropolis "
               "swap sweeps under the v0.4 12-token TRAIN adjacency "
               "(doctrine silent on micro-arrangement of these cells)")


def swap_gibbs(grid, free, A, rng, sweeps=3):
    """Multiset-preserving variant of the v0.3/v0.4 Gibbs: instead of resampling
    (which drifts the budget), propose swapping the tokens of two free cells and
    accept with Metropolis prob exp(delta log-likelihood) under the row-
    normalized 12-token adjacency. Typed voids + prescriptive cells are fixed
    but vote as neighbours. Exact D2 budget preserved by construction."""
    logA = np.log(A + EPS)
    N = grid.shape[0]

    def contrib(r, c):
        e, t = 0.0, grid[r, c]
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < N and 0 <= cc < N:
                u = grid[rr, cc]
                e += logA[t, u] + logA[u, t]
        return e

    for _ in range(sweeps):
        for i in rng.permutation(len(free)):
            j = int(rng.integers(len(free)))
            (r1, c1), (r2, c2) = free[i], free[j]
            if grid[r1, c1] == grid[r2, c2]:
                continue
            before = contrib(r1, c1) + contrib(r2, c2)
            grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
            after = contrib(r1, c1) + contrib(r2, c2)
            if after - before < 0 and rng.random() >= math.exp(after - before):
                grid[r1, c1], grid[r2, c2] = grid[r2, c2], grid[r1, c1]
    return grid


# ------------------------------------------------------------------ unit generation
def angle_sort(cells, ctr=(9.5, 9.5)):
    return sorted(cells, key=lambda rc: math.atan2(rc[0] - ctr[0], rc[1] - ctr[1]))


def gen_garden_city(skel, A, rng):
    """Prescriptive (Howard 1902 diagrams 2-3): industry ring of alternating
    T7 shed / E_VACANT yard fronting the circular railway (outermost build
    band); Crystal Palace arcade = 12 T7 cells on the Central Park edge spread
    by angle; cottages (T1) fill the concentric residential bands; civic T2 on
    RESERVED cells (centre cluster = town hall/library/museum + hospital, 6
    ward cells = Grand Avenue schools). Fully prescriptive; no Gibbs."""
    g, sem = skel.copy(), np.full(skel.shape, None, dtype=object)
    build = [(r, c) for r in range(20) for c in range(20) if skel[r, c] == 0]
    civic = [(r, c) for r in range(20) for c in range(20) if skel[r, c] == -1]
    edge = [rc for rc in build if rc[0] in (1, 18) or rc[1] in (1, 18)]
    assert len(edge) == 64, len(edge)
    for i, (r, c) in enumerate(angle_sort(edge)):
        g[r, c] = 7 if i % 2 == 0 else 0
        sem[r, c] = "industrial" if i % 2 == 0 else None
    inner = [rc for rc in build if rc not in edge]
    park = {(r, c) for r in range(20) for c in range(20)
            if skel[r, c] == 11 and math.hypot(r - 9.5, c - 9.5) <= 5.5}
    cand = [rc for rc in inner if any(
        (rc[0] + dr, rc[1] + dc) in park
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)))]
    cand = angle_sort(cand)
    assert len(cand) >= 12, len(cand)
    palace = [cand[int(round(i * len(cand) / 12)) % len(cand)] for i in range(12)]
    palace = list(dict.fromkeys(palace))
    k = 0
    while len(palace) < 12:           # de-dup top-up, keep angle spread
        if cand[k] not in palace:
            palace.append(cand[k])
        k += 1
    for r, c in palace:
        g[r, c], sem[r, c] = 7, "commercial_retail"
    for r, c in inner:
        if g[r, c] == 0 and (r, c) not in palace:
            g[r, c], sem[r, c] = 1, "residential_lowrise"
    centre = sorted([rc for rc in civic if math.hypot(rc[0] - 9.5, rc[1] - 9.5) < 2])
    wards = sorted([rc for rc in civic if rc not in centre])
    for (r, c), s in zip(centre, ["public_admin", "public_admin", "public_admin",
                                  "healthcare"]):
        g[r, c], sem[r, c] = 2, s
    for r, c in wards:
        g[r, c], sem[r, c] = 2, "education"
    return g, sem, dict(prescriptive_cells=len(build) + len(civic), gibbs_cells=0,
                        prescriptive_rule="industry ring alternating T7/void on rail; "
                        "12-cell Crystal Palace arcade by angle on park edge; T1 "
                        "concentric cottage bands; civic centre PA/health + 6 ward schools")


def gen_ville_radieuse(skel, A, rng):
    """Prescriptive: the D2 skeleton already IS Corbusier's ~200 m tower
    lattice in continuous green; every build cell takes a T6 tower demanding
    residential_estate (unites d'habitation). Fully determined; no Gibbs."""
    g, sem = skel.copy(), np.full(skel.shape, None, dtype=object)
    n = 0
    for r in range(20):
        for c in range(20):
            if skel[r, c] == 0:
                g[r, c], sem[r, c], n = 6, "residential_estate", n + 1
    return g, sem, dict(prescriptive_cells=n, gibbs_cells=0,
                        prescriptive_rule="T6 tower on every lattice cell of the "
                        "skeleton (towers-in-the-park, ~200 m rhythm)")


def gen_neighbourhood_unit(skel, A, rng):
    """Prescriptive (Perry 1929 diagram): shops + apartments at the four
    arterial corner junctions (the 35 corner-nearest build cells take T2,
    the 2 nearest per corner demanding commercial_retail, the rest mixed
    lowrise = apartments grading toward corners); single-family T1 fills the
    interior; centre RESERVED cells = school + community (3 edu + 1 PA)."""
    g, sem = skel.copy(), np.full(skel.shape, None, dtype=object)
    build = [(r, c) for r in range(20) for c in range(20) if skel[r, c] == 0]
    civic = sorted([(r, c) for r in range(20) for c in range(20) if skel[r, c] == -1])
    corners = [(0, 0), (0, 19), (19, 0), (19, 19)]
    def cscore(rc):
        return min(max(abs(rc[0] - a), abs(rc[1] - b)) for a, b in corners)
    t2 = sorted(build, key=lambda rc: (cscore(rc), rc))[:35]
    shops = []
    for a, b in corners:
        near = sorted(t2, key=lambda rc: (max(abs(rc[0] - a), abs(rc[1] - b)), rc))[:2]
        shops += near
    shops = list(dict.fromkeys(shops))[:8]
    for r, c in t2:
        g[r, c] = 2
        sem[r, c] = "commercial_retail" if (r, c) in shops else "residential_lowrise"
    for r, c in build:
        if (r, c) not in t2:
            g[r, c], sem[r, c] = 1, "residential_lowrise"
    for (r, c), s in zip(civic, ["education", "education", "education", "public_admin"]):
        g[r, c], sem[r, c] = 2, s
    return g, sem, dict(prescriptive_cells=len(build) + len(civic), gibbs_cells=0,
                        prescriptive_rule="T2 mixed/apartment cells = 35 corner-nearest "
                        "(2/corner demand shops); T1 single-family interior; central "
                        "school+community on RESERVED cells")


def gen_microrayon(skel, A, rng):
    """Prescriptive: T7 services at the 4 centre-nearest cells (KBO daily-
    services block by the green spine); T3 slabs in parallel E-W rows with a
    1-row sanitary gap (SNiP insolation ~2H at 22 m = 1 cell). The doctrine's
    own 'free planning' (svobodnaya planirovka) leaves T4 vs E_VACANT
    micro-arrangement unspecified -> Shanghai-accent swap-Gibbs on the 170
    free cells. Civic RESERVED cells = schools/kindergartens (education)."""
    g, sem = skel.copy(), np.full(skel.shape, None, dtype=object)
    build = [(r, c) for r in range(20) for c in range(20) if skel[r, c] == 0]
    civic = sorted([(r, c) for r in range(20) for c in range(20) if skel[r, c] == -1])
    for r, c in civic:
        g[r, c], sem[r, c] = 2, "education"
    t7 = sorted(build, key=lambda rc: (math.hypot(rc[0] - 9.5, rc[1] - 9.5), rc))[:4]
    for r, c in t7:
        g[r, c], sem[r, c] = 7, "commercial_retail"
    used = set(t7)
    t3 = []
    for row in (2, 12, 4, 14, 6, 16):
        for rc in sorted([b for b in build if b[0] == row and b not in used]):
            if len(t3) < 54:
                t3.append(rc); used.add(rc)
    assert len(t3) == 54
    for r, c in t3:
        g[r, c], sem[r, c] = 3, "residential_estate"
    free = [rc for rc in build if rc not in used]
    seq = np.array([4] * 45 + [0] * (len(free) - 45))
    rng.shuffle(seq)
    for (r, c), t in zip(free, seq):
        g[r, c] = int(t)
        sem[r, c] = "residential_estate" if t == 4 else None
    g = swap_gibbs(g, free, A, rng, sweeps=3)
    for r, c in free:
        sem[r, c] = "residential_estate" if g[r, c] == 4 else None
    return g, sem, dict(prescriptive_cells=len(civic) + 4 + 54, gibbs_cells=len(free),
                        gibbs_label=GIBBS_LABEL,
                        prescriptive_rule="T7 services at centre; T3 slab rows on "
                        "alternating rows (1-row insolation gap); T4+void free-plan "
                        "cells arranged by Shanghai-accent swap-Gibbs")


def gen_gb50180(skel, A, rng):
    """Prescriptive (GB 50180-2018 sun-spacing logic, north = +y = high row in
    plot coords): within each road-bounded block the T6 60 m towers take the
    northernmost rows, T4 32 m slabs the southern rows (tall mass north so it
    shades the street, not the dwellings); T7 podium retail = the 16 centre-
    nearest cells fronting the two internal arterials (15-minute-circle
    street commerce); civic RESERVED cells = 4 edu + 1 PA + 1 health."""
    g, sem = skel.copy(), np.full(skel.shape, None, dtype=object)
    build = [(r, c) for r in range(20) for c in range(20) if skel[r, c] == 0]
    civic = sorted([(r, c) for r in range(20) for c in range(20) if skel[r, c] == -1])
    west = [rc for rc in civic if rc[1] < 10]
    east = [rc for rc in civic if rc[1] >= 10]
    for (r, c), s in zip(west, ["education", "education", "public_admin", "healthcare"]):
        g[r, c], sem[r, c] = 2, s
    for r, c in east:
        g[r, c], sem[r, c] = 2, "education"
    cand = [rc for rc in build if rc[0] in (5, 7, 12, 14)]
    t7 = sorted(cand, key=lambda rc: (math.hypot(rc[0] - 9.5, rc[1] - 9.5), rc))[:16]
    for r, c in t7:
        g[r, c], sem[r, c] = 7, "commercial_retail"
    rest = [rc for rc in build if rc not in t7]
    def block(rc):
        rb = 0 if rc[0] < 6 else (1 if rc[0] < 13 else 2)
        cb = 0 if rc[1] < 6 else (1 if rc[1] < 13 else 2)
        return rb, cb
    groups = {}
    for rc in rest:
        groups.setdefault(block(rc), []).append(rc)
    nrest, t6_total = len(rest), 75
    quota, fr = {}, []
    for b, cells in sorted(groups.items()):
        q = t6_total * len(cells) / nrest
        quota[b] = int(q); fr.append((q - int(q), b))
    for _, b in sorted(fr, reverse=True)[: t6_total - sum(quota.values())]:
        quota[b] += 1
    for b, cells in groups.items():
        cells = sorted(cells, key=lambda rc: (-rc[0], rc[1]))  # north (high row=+y) first
        for i, (r, c) in enumerate(cells):
            g[r, c] = 6 if i < quota[b] else 4
            sem[r, c] = "residential_estate"
    return g, sem, dict(prescriptive_cells=len(build) + len(civic), gibbs_cells=0,
                        prescriptive_rule="per-block T6 tower rows north / T4 slab rows "
                        "south (sun-spacing); T7 street-front podium retail at the 16 "
                        "centre-nearest arterial-facing cells; civic 4edu+1PA+1health")


GEN = {"garden_city": gen_garden_city, "ville_radieuse": gen_ville_radieuse,
       "neighbourhood_unit": gen_neighbourhood_unit, "microrayon": gen_microrayon,
       "gb50180": gen_gb50180}


# ------------------------------------------------------------------ semantic-conditioned collage
def collage_semantic(grid, sem, pools, idx, by_ts, rng, canvas):
    """v0.3/v0.4 cell collage with NEW semantic conditioning: for each built
    cell prefer pool cells whose GFA-dominant semantic == demanded semantic;
    fall back to any same-token cell when that pool is empty (counted).
    No keep-out (roads are authored); clip at canvas edge; drop <1 m2."""
    by_wt, by_t, members = idx
    N = grid.shape[0]
    clipbox = box(0, 0, canvas, canvas)
    rows = []
    stats = {}   # demanded sem -> {cells, fallback, keys_used:set}
    for r in range(N):
        for c in range(N):
            t = int(grid[r, c])
            if t in (0, 9, 10, 11, -1):
                continue
            s = sem[r, c] or "unlabeled"
            st = stats.setdefault(s, {"cells": 0, "fallback": 0, "keys": set()})
            st["cells"] += 1
            keys = by_ts.get((t, s))
            matched = bool(keys)
            if not matched:
                keys = by_t[t]
                st["fallback"] += 1
            key = keys[int(rng.integers(len(keys)))]
            st["keys"].add(key)
            sub = pools.loc[members[key]]
            for (_, b) in sub.iterrows():
                geom = translate(b.geometry, xoff=c * CELL, yoff=r * CELL)
                geom = geom.intersection(clipbox)
                if geom.is_empty or geom.area < 1.0:
                    continue
                rows.append(dict(geometry=geom, height=float(b.height),
                                 floors=int(b.floors), family=b.family,
                                 semantic=b.semantic, demanded_semantic=s,
                                 sem_matched=matched, token=t, src_window=b.window,
                                 src_key=key, tgt_row=r, tgt_col=c))
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32651")
    gdf["fp_area"] = gdf.geometry.area
    gdf["gfa"] = gdf["fp_area"] * gdf["floors"]
    n_cells = sum(v["cells"] for v in stats.values())
    n_fb = sum(v["fallback"] for v in stats.values())
    log = {"n_built_cells": n_cells, "n_semantic_matched": n_cells - n_fb,
           "n_fallback_any_token_cell": n_fb,
           "fallback_rate": round(n_fb / max(n_cells, 1), 4),
           "per_semantic": {s: {"demanded_cells": v["cells"], "fallback": v["fallback"],
                                "unique_pool_cells_used": len(v["keys"])}
                            for s, v in sorted(stats.items())}}
    return gdf, log


# ------------------------------------------------------------------ indicators / render
def indicators(df, area):  # adapted from TF/v04/run_generate_v04.py::indicators (area param)
    row = {
        "FAR": float((df["fp_area"] * df["floors"]).sum() / area),
        "coverage": float(df["fp_area"].sum() / area),
        "height_p50": float(np.percentile(df["height"], 50)),
        "height_p90": float(np.percentile(df["height"], 90)),
        "median_fp_area": float(df["fp_area"].median()),
    }
    sh = df["family"].value_counts(normalize=True)
    for f in FAMILIES:
        row[f"share_{f}"] = float(sh.get(f, 0.0))
    return row


def render_axon(ax, gdf, canvas, simplify=1.5):
    # copied verbatim from TF/v04/run_generate_v04.py::render_axon (itself
    # verbatim v0.3), except CANVAS -> canvas parameter for 3 km composites.
    cmap = plt.get_cmap("viridis")
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
    ax.set_xlim(0, canvas); ax.set_ylim(0, canvas); ax.set_zlim(0, ZMAX)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def make_tile(gdf, path, canvas, simplify=1.8):
    # adapted from TF/v04/run_generate_v04.py::make_tile (bigger figsize/dpi)
    fig = plt.figure(figsize=(5.2, 4.2))
    ax = fig.add_subplot(111, projection="3d")
    render_axon(ax, gdf, canvas, simplify=simplify)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def plot_plan(ax, grid, gdf, canvas, title=None, fs=9.5):
    """Figure-ground with typed-void underlay: green=E_GREEN, gray=E_ROAD."""
    N = grid.shape[0]
    img = np.ones((N, N, 3))
    img[grid == T_GREEN] = (0.71, 0.83, 0.64)
    img[grid == T_ROAD] = (0.83, 0.83, 0.83)
    ax.imshow(img, origin="lower", extent=(0, canvas, 0, canvas),
              interpolation="nearest", zorder=0)
    for geom in gdf.geometry:
        gs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in gs:
            if p.geom_type != "Polygon":
                continue
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color="#1a1a1a", lw=0, zorder=2)
    ax.set_xlim(0, canvas); ax.set_ylim(0, canvas)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999"); sp.set_linewidth(0.6)
    if title:
        ax.set_title(title, fontsize=fs)


# ------------------------------------------------------------------ composites (60x60)
def stamp(G, S, M, unit_g, unit_s, r0, c0):
    """Paste unit grid/sem at (r0,c0); cells already stamped keep precedence.
    Returns number of unit cells lost to overlap/out-of-bounds."""
    lost = 0
    for r in range(20):
        for c in range(20):
            R, C = r0 + r, c0 + c
            if not (0 <= R < G.shape[0] and 0 <= C < G.shape[1]) or M[R, C]:
                lost += 1
                continue
            G[R, C], S[R, C], M[R, C] = unit_g[r, c], unit_s[r, c], True
    return lost


def compose(paradigm, unit_g, unit_s, A, rng):
    NG = 60
    G = np.full((NG, NG), T_VAC, dtype=np.int16)
    S = np.full((NG, NG), None, dtype=object)
    M = np.zeros((NG, NG), bool)
    lost, n_units = 0, 0
    if paradigm == "garden_city":
        # Howard's Social City fragment: central city + 6 ward/satellite units at
        # R=19 cells, agricultural GREEN BELT fills the remainder, rail ring edge.
        G[:] = T_GREEN
        G[0, :] = G[-1, :] = G[:, 0] = G[:, -1] = T_ROAD   # circular railway
        lost += stamp(G, S, M, unit_g, unit_s, 20, 20); n_units += 1
        for k in range(6):
            th = math.radians(90 + k * 60)
            cy, cx = 30 + 19 * math.sin(th), 30 + 19 * math.cos(th)
            r0 = min(max(int(round(cy - 10)), 1), NG - 21)
            c0 = min(max(int(round(cx - 10)), 1), NG - 21)
            lost += stamp(G, S, M, unit_g, unit_s, r0, c0); n_units += 1
        rule = ("central-park core unit + 6 satellite ward units on a 950 m ring "
                "(Howard Social City diagram, compressed), agricultural green-belt "
                "E_GREEN fills the remainder, circular-railway E_ROAD edge; "
                "overlaps resolved centre-first (lost cells counted)")
    elif paradigm == "ville_radieuse":
        # 3x3 lattice units; business-centre unit (T8/T6, office_finance) at the
        # head of the main axis (top-centre in plot = slot row 2, col 1).
        biz_g, biz_s = unit_g.copy(), unit_s.copy()
        nb8 = 0
        for r in range(20):
            for c in range(20):
                if biz_g[r, c] == 6:
                    biz_s[r, c] = "office_finance"
                    if math.hypot(r - 9.5, c - 9.5) <= 5.0:
                        biz_g[r, c], nb8 = 8, nb8 + 1
        for i in range(3):
            for j in range(3):
                ug, us = (biz_g, biz_s) if (i, j) == (2, 1) else (unit_g, unit_s)
                lost += stamp(G, S, M, ug, us, 20 * i, 20 * j); n_units += 1
        rule = (f"3x3 units; head-of-axis business-centre unit = same lattice with "
                f"the {nb8} cells within 250 m of centre as T8 megastructures, rest "
                "T6, all demanding office_finance (cite donnee des affaires); 8 "
                "residential units")
    elif paradigm in ("neighbourhood_unit", "microrayon"):
        for r0 in (10, 30):
            for c0 in (10, 30):
                lost += stamp(G, S, M, unit_g, unit_s, r0, c0); n_units += 1
        if paradigm == "neighbourhood_unit":
            nctr = 0
            for r in range(26, 34):
                for c in range(26, 34):
                    if 1 <= G[r, c] <= 8 and max(abs(r - 29.5), abs(c - 29.5)) <= 2.5:
                        G[r, c], S[r, c], nctr = 7, "commercial_retail", nctr + 1
            rule = (f"2x2 units; shared 100 m arterials = merged unit edge roads; "
                    f"town-center cell-block = {nctr} cells at the central junction "
                    "set to T7 commercial (Perry: business at arterial junctions); "
                    "periphery E_VACANT (city context unspecified)")
        else:
            rule = ("2x2 microrayons; the merged 100 m edge-road corridor between "
                    "the rows = the transit arterial (magistral); periphery "
                    "E_VACANT (rayon context unspecified)")
    else:  # gb50180
        for r0 in (10, 30):
            for c0 in (8, 32):
                lost += stamp(G, S, M, unit_g, unit_s, r0, c0); n_units += 1
        cyc = ["education", "public_admin", "healthcare"]
        k = 0
        for r in range(10, 50):
            G[r, 29] = G[r, 30] = T_ROAD       # transit spine
            for c in (28, 31):
                m = (r + c) % 3
                if m == 0:
                    G[r, c], S[r, c] = 7, "commercial_retail"
                elif m == 1:
                    G[r, c], S[r, c] = 2, cyc[k % 3]; k += 1
                else:
                    G[r, c] = T_GREEN
            M[r, 28:32] = True
        rule = ("2x2 superblock units + central 200 m 15-minute-circle civic strip "
                "(transit spine + alternating T7 retail / T2 civic / pocket green) "
                "serving all four; periphery E_VACANT")
    return G, S, dict(assembly_rule=rule, n_units=n_units, cells_lost_to_overlap=lost)


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    print("loading pools ...")
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    by_ts, avail = build_semantic_index(pools)
    A = load_adjacency12()
    d2 = json.load(open(os.path.join(V06, "results_d2.json"), encoding="utf-8"))
    res = {"_rule": ("D3 generation: prescriptive doctrine geometry first, "
                     "Shanghai-accent swap-Gibbs where doctrine silent, then "
                     "semantic-conditioned cell collage from TF/v03 token_pools "
                     "(prefer GFA-dominant-semantic match, fallback any same-token "
                     "cell, counted). seed 0. Units 1 km/20x20, composites 3 km/60x60."),
           "pool_semantic_availability": avail, "paradigms": {}}
    plans = {}
    for p in PARADIGMS:
        print("==", p)
        skel = np.load(os.path.join(V06, f"skeleton_{p}.npy"))
        g, sem, meta = GEN[p](skel, A, rng)
        demanded = d2["paradigms"][p]["token_budget"]
        ach = {f"T{t}": int((g == t).sum()) for t in range(1, 9) if (g == t).sum()}
        ach.update({"E_VACANT": int((g == 0).sum()), "E_ROAD": int((g == 9).sum()),
                    "E_GREEN": int((g == 11).sum())})
        gdf, slog = collage_semantic(g, sem, pools, idx, by_ts, rng, 1000.0)
        gdf.to_parquet(os.path.join(CACHE, f"unit_{p}.parquet"))
        np.save(os.path.join(CACHE, f"grid_unit_{p}.npy"), g)
        ind = indicators(gdf, 1e6)
        exact = all(ach.get(k, 0) == v for k, v in demanded.items())
        res["paradigms"][p] = {"unit": {
            "indicators": {k: round(v, 4) for k, v in ind.items()},
            "n_buildings": int(len(gdf)),
            "token_composition": {"demanded_d2": demanded, "achieved": ach,
                                  "exact_budget_match": bool(exact)},
            "semantic_collage": slog, "placement": meta}}
        plans[p] = (g, gdf, ind)
        make_tile(gdf, os.path.join(CACHE, f"tile_unit_{p}.png"), 1000.0)
        print("  unit FAR", round(ind["FAR"], 2), "cov", round(ind["coverage"], 2),
              "fallback", slog["fallback_rate"], "exact", exact)

        # ---- composite
        G, S, cmeta = compose(p, g, sem, A, rng)
        gdfc, slogc = collage_semantic(G, S, pools, idx, by_ts, rng, 3000.0)
        gdfc.to_parquet(os.path.join(CACHE, f"city_{p}.parquet"))
        np.save(os.path.join(CACHE, f"grid_city_{p}.npy"), G)
        indc = indicators(gdfc, 9e6)
        achc = {f"T{t}": int((G == t).sum()) for t in range(1, 9) if (G == t).sum()}
        achc.update({"E_VACANT": int((G == 0).sum()), "E_ROAD": int((G == 9).sum()),
                     "E_GREEN": int((G == 11).sum())})
        res["paradigms"][p]["city"] = {
            "indicators": {k: round(v, 4) for k, v in indc.items()},
            "n_buildings": int(len(gdfc)),
            "token_composition_achieved": achc,
            "semantic_collage": slogc, **cmeta}
        print("  city FAR", round(indc["FAR"], 2), "bld", len(gdfc),
              "lost", cmeta["cells_lost_to_overlap"])

        # ---- per-paradigm unit figure (plan + axon)
        name, year = META[p]
        fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
        plot_plan(axes[0], g, gdf, 1000.0,
                  title="figure-ground plan (1 km) — green=E_GREEN, gray=E_ROAD")
        axes[1].imshow(plt.imread(os.path.join(CACHE, f"tile_unit_{p}.png")))
        axes[1].set_axis_off()
        axes[1].set_title("3D axonometric (height-coloured)", fontsize=9.5)
        fig.suptitle(f"{name}, {year} — UNIT  |  FAR {ind['FAR']:.2f}   "
                     f"coverage {ind['coverage']:.2f}   h_p90 {ind['height_p90']:.0f} m   "
                     f"{len(gdf)} buildings", fontsize=12.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(FIG, f"fig_d3_{p}_unit.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

        # ---- per-paradigm city figure (plan)
        fig, ax = plt.subplots(figsize=(8.6, 9.0))
        plot_plan(ax, G, gdfc, 3000.0)
        ax.set_title(f"{name}, {year} — CITY COMPOSITE (3 x 3 km)\n"
                     f"FAR {indc['FAR']:.2f}   coverage {indc['coverage']:.2f}   "
                     f"h_p90 {indc['height_p90']:.0f} m   {len(gdfc)} buildings\n"
                     + "\n".join(textwrap.wrap(
                         "assembly: " + cmeta["assembly_rule"], width=84)),
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"fig_d3_{p}_city.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- gallery: 5 cols x 2 rows (plan above, axon below)
    fig = plt.figure(figsize=(20, 9.0))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.88], hspace=0.06, wspace=0.05)
    for j, p in enumerate(PARADIGMS):
        g, gdf, ind = plans[p]
        name, year = META[p]
        ax = fig.add_subplot(gs[0, j])
        plot_plan(ax, g, gdf, 1000.0)
        ax.set_title(f"{name}\n{year}\nFAR {ind['FAR']:.2f}   cov {ind['coverage']:.2f}   "
                     f"h$_{{90}}$ {ind['height_p90']:.0f} m", fontsize=11)
        ax2 = fig.add_subplot(gs[1, j])
        ax2.imshow(plt.imread(os.path.join(CACHE, f"tile_unit_{p}.png")))
        ax2.set_axis_off()
    fig.suptitle("Five classical doctrines, one Shanghai vocabulary — 1 km generated units "
                 "(figure-ground above: green = doctrinal green, gray = authored roads; "
                 "height-coloured axonometric below)", fontsize=14, y=0.985)
    fig.text(0.5, 0.012, "Token budgets from D2 doctrine translation; cell collage from "
             "measured Shanghai pools (TF/v03), semantic-conditioned; seed 0. "
             "Prescriptive doctrine geometry; Shanghai-accent Gibbs only where the "
             "doctrine is silent.", ha="center", fontsize=9.5, style="italic")
    fig.savefig(os.path.join(FIG, "fig_d3_gallery.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    json.dump(res, open(os.path.join(V06, "results_d3.json"), "w"),
              ensure_ascii=False, indent=1)
    print("done")


if __name__ == "__main__":
    main()
