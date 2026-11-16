#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generate_v03.py — Agent 2 generation suite on the v0.3 TRAIN-only grammar
(k=8 tokens + EMPTY, 18 TRAIN / 6 HELDOUT windows).

Subcommands (each call <45 s; everything cached in cache_gen/):
  realprep [ids]      : heldout real canvases in local 0-1000 m coords + 10-indicator
                        vectors + per-cell real GFA (for rank-matching) + 3D tiles.
  recon <ids>         : TASK 1 — held-out reconstruction exam. The window's OBSERVABLE
                        token composition (bincount of its real token grid; heldout cells
                        were nearest-centroid assigned by Agent 1, never in the grammar)
                        -> FREE growth on a blank 20x20 (no skeleton) with TRAIN adjacency:
                        exact-multiset random init + 5 random-order Gibbs sweeps,
                        P(t|nbrs) ∝ Π A[t,n] * share_t * (target/current)^β, β=8
                        (the v0.2 composition-drift fix, kept verbatim) -> CELL COLLAGE
                        from ALL-TRAIN pools (not window-restricted: the heldout window
                        owns no pool, that is the point) -> parquet + .fgb + tile.
  compare             : 10-indicator vectors for 6 generated + 6 real heldout fabrics,
                        z-scored over the 12, 6x6 distance matrix -> OUT-OF-SAMPLE hit
                        rate -> merged into results_v03.json.
  fig3                : FIG3 — 6 columns (real / generated 3D) + distance heatmap.
  transplant <sig>    : TASK 2 — source signature (token demand + SOURCE-window-only pools)
                        on each heldout site skeleton: site's real token grid, EMPTY cells
                        stay empty, non-empty cells ranked by real cell GFA receive the
                        demand ranked by token intensity (pool-mean GFA). Sparser demand
                        pads EMPTY (evacuation); denser demand is compressed to skeleton.
                        Pool fallback to ALL-TRAIN only when the source window lacks a
                        token entirely (logged).
  fig4                : FIG4 — 4x6 transplant gallery, most asymmetric cells marked.
  path                : TASK 3 — programmatic pick: heldout window nearest the slab-mat
                        band centre (mean PCA xy of TRAIN windows whose largest non-EMPTY
                        token share is T3) among the four heldouts Agent 1 places in the
                        slab band (A20/A29/A35/A43). Signature lerped toward A01 Lujiazui
                        in 5 steps; site skeleton = its own real grid (EMPTY preserved);
                        pools blended per cell: P(A01-only pool)=w else ALL-TRAIN pool
                        (w=0 cannot use the window's own pool — it has none; ALL-TRAIN is
                        the honest reconstruction setting).
  fig5                : FIG5 — 5 axons + FAR/coverage/h_p90/tower+T8 curves + PCA inset.

seed 0 (each subcommand reseeds default_rng(0)). Vectors as FlatGeobuf via /tmp
(NEVER GPKG on this mount). Figures dpi 130, English-only labels.
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = "/sessions/epic-peaceful-feynman/mnt/Urban-Stakeholder-Workshop"
if not os.path.isdir(ROOT):
    ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
V03 = os.path.join(ROOT, "typology_foundation", "v03")
CACHE = os.path.join(V03, "cache_gen")
os.makedirs(CACHE, exist_ok=True)
RESULTS = os.path.join(V03, "results_v03.json")

HELDOUT = ["A02", "A07", "A20", "A29", "A35", "A43"]
SIGS4 = ["A01", "A10", "A36", "A26"]          # Lujiazui / Laoximen / Zhangjiang / Jiading
PATH_CANDS = ["A20", "A29", "A35", "A43"]     # the four heldouts in A1's slab-mat band
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
SHORT = {"A01": "Lujiazui CBD", "A02": "Nanjing Rd E.", "A07": "Bund",
         "A10": "Laoximen lilong", "A20": "Sanlin", "A26": "Jiading new town",
         "A29": "Nanqiao", "A35": "Caohejing", "A36": "Zhangjiang ind.",
         "A43": "Liangcheng"}

K = 9               # T0..T8
SEED = 0
SWEEPS = 5
EPS = 0.01          # smoothing for never-observed adjacencies (e.g. T1-T8)
BETA = 8.0          # composition feedback exponent — the v0.2 drift fix, kept
CELL = 50.0
CANVAS = 1000.0
AREA = 1_000_000.0
N = 20
HCAP = 150.0        # shared viridis cap; ZMAX 350 (tallest building 338 m, A01)
ZMAX = 350.0
ELEV, AZIM = 38, -60  # one camera for every axon (v0.2 convention)
PATH_W = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------- assets
def load_split():
    return json.load(open(os.path.join(V03, "split.json"), encoding="utf-8"))


def origins():
    return {w["id"]: (w["cx"] - 500.0, w["cy"] - 500.0) for w in load_split()["windows"]}


def load_adjacency():
    adj = json.load(open(os.path.join(V03, "adjacency.json"), encoding="utf-8"))
    return np.array(adj["row_normalized"], dtype=float)


def load_signatures():
    s = json.load(open(os.path.join(V03, "signatures.json"), encoding="utf-8"))
    return s, {w["id"]: w for w in s["windows"]}


def token_intensity():
    tk = json.load(open(os.path.join(V03, "tokens.json"), encoding="utf-8"))
    inten = {0: 0.0}
    for t in range(1, K):
        inten[t] = float(tk["stats"][str(t)]["gfa"])
    return inten


def load_grid(wid):
    return np.load(os.path.join(V03, f"token_grid_{wid}.npy"))


def load_pools_geo():
    g = gpd.read_file(os.path.join(V03, "token_pools.fgb"))
    g["key"] = g["window"] + "_r" + g["row"].astype(str) + "_c" + g["col"].astype(str)
    return g


def build_cell_index(pools):
    cells = pools.drop_duplicates(subset="key")
    by_wt = cells.groupby(["window", "token"])["key"].apply(list).to_dict()
    by_t = cells.groupby("token")["key"].apply(list).to_dict()
    members = {k: v for k, v in pools.groupby("key").groups.items()}
    return by_wt, by_t, members


def merge_results(key, payload):
    res = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
    res[key] = payload
    json.dump(res, open(RESULTS, "w"), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- growth (v0.2 mechanics, K=9)
def grow_grid(target_counts, A, rng, sweeps=SWEEPS):
    """Exact-multiset random init then random-order Gibbs sweeps.
    P(t | nbrs) ∝ prod_j A[t, nbr_j] * share_t * (target_t / current_t)^BETA.
    Random order kills raster-scan bias; the feedback term keeps the composition
    from drifting (v0.2: beta=1 bled rare tokens into T0)."""
    target = np.asarray(target_counts, dtype=float)
    seq = np.repeat(np.arange(K), target_counts)
    rng.shuffle(seq)
    grid = seq.reshape(N, N).copy()
    Ae = A + EPS
    share = np.maximum(target, 1e-9) / target.sum()
    counts = np.bincount(grid.ravel(), minlength=K).astype(float)
    for _ in range(sweeps):
        for idx in rng.permutation(N * N):
            r, c = divmod(idx, N)
            like = np.ones(K)
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
            new = int(rng.choice(K, p=p))
            grid[r, c] = new
            counts[new] += 1
    return grid


# ---------------------------------------------------------------- collage
def collage(grid, pools, idx, rng, src_window):
    """Instantiate a token grid by cell collage. src_window: callable(rng) -> window id
    or None (= ALL-TRAIN pool). Overhang kept; clip only at the 1 km canvas edge."""
    by_wt, by_t, members = idx
    rows = []
    fallbacks = 0
    clipbox = box(0, 0, CANVAS, CANVAS)
    for r in range(N):
        for c in range(N):
            t = int(grid[r, c])
            if t == 0:
                continue
            w = src_window(rng)
            keys = by_wt.get((w, t)) if w is not None else by_t[t]
            if not keys:                      # source window lacks this token entirely
                keys = by_t[t]
                fallbacks += 1
            key = keys[int(rng.integers(len(keys)))]
            sub = pools.loc[members[key]]
            for _, b in sub.iterrows():
                geom = translate(b.geometry, xoff=c * CELL, yoff=r * CELL)
                geom = geom.intersection(clipbox)
                if geom.is_empty or geom.area < 1.0:
                    continue
                rows.append(dict(geometry=geom, height=float(b.height),
                                 floors=int(b.floors), family=b.family,
                                 semantic=b.semantic, token=t, src_window=b.window,
                                 src_key=key, tgt_row=r, tgt_col=c))
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32651")
    gdf["fp_area"] = gdf.geometry.area
    gdf["gfa"] = gdf["fp_area"] * gdf["floors"]
    gdf.attrs["fallbacks"] = fallbacks
    return gdf, fallbacks


def all_train(rng):
    return None


def fixed(wid):
    return lambda rng: wid


# ---------------------------------------------------------------- rank-match (9 tokens)
def largest_remainder(frac, total):
    base = np.floor(frac).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(frac - base))
        base[order[:rem]] += 1
    return base


def rank_match(site_grid, demand_counts, cgfa, inten):
    """Site EMPTY cells stay empty. Built cells sorted by real cell GFA desc receive
    tokens sorted by intensity desc; sparse demand pads EMPTY, dense demand compressed."""
    nonempty = [(r, c) for r in range(N) for c in range(N) if site_grid[r, c] > 0]
    nB = len(nonempty)
    d = np.asarray(demand_counts, dtype=float)
    built = d[1:]
    n_built = built.sum()
    if n_built > nB:
        built = largest_remainder(built * nB / n_built, nB)
    else:
        built = largest_remainder(built, int(round(n_built)))
    seq = []
    for t in sorted(range(1, K), key=lambda t: -inten[t]):
        seq += [t] * int(built[t - 1])
    seq += [0] * (nB - len(seq))
    order = sorted(nonempty, key=lambda rc: -cgfa.get(f"{rc[0]}_{rc[1]}", 0.0))
    out = np.zeros_like(site_grid)
    for (r, c), t in zip(order, seq):
        out[r, c] = t
    return out


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


# ---------------------------------------------------------------- 3D axon + tiles
def render_axon(ax, gdf, simplify=1.5):
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
    ax.set_xlim(0, CANVAS); ax.set_ylim(0, CANVAS); ax.set_zlim(0, ZMAX)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def make_tile(gdf, path, simplify=1.8):
    fig = plt.figure(figsize=(3.6, 2.9))
    ax = fig.add_subplot(111, projection="3d")
    render_axon(ax, gdf, simplify=simplify)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(path, dpi=130, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def show_tile(ax, path):
    ax.imshow(plt.imread(path))
    ax.set_axis_off()


def save_canvas(gdf, name, fgb=False):
    gdf.to_parquet(os.path.join(CACHE, f"{name}.parquet"))
    if fgb:
        tmp = os.path.join("/tmp", f"{name}.fgb")
        gdf.to_file(tmp, driver="FlatGeobuf")
        shutil.copy2(tmp, os.path.join(V03, f"{name}.fgb"))


# ---------------------------------------------------------------- realprep
def cmd_realprep(ids=None):
    ids = ids or HELDOUT
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))
    org = origins()
    for hid in ids:
        g = lib[lib["window"] == hid].copy()
        ox, oy = org[hid]
        g["geometry"] = g.geometry.apply(lambda gm: translate(gm, xoff=-ox, yoff=-oy))
        g.to_parquet(os.path.join(CACHE, f"real_{hid}.parquet"))
        cent = g.geometry.centroid
        col = np.clip((cent.x // CELL).astype(int), 0, N - 1)
        row = np.clip((cent.y // CELL).astype(int), 0, N - 1)
        gfa = (g["fp_area"] * g["floors"]).groupby([row, col]).sum()
        cgfa = {f"{r}_{c}": float(v) for (r, c), v in gfa.items()}
        json.dump(cgfa, open(os.path.join(CACHE, f"cellgfa_{hid}.json"), "w"))
        json.dump(indicators(g), open(os.path.join(CACHE, f"ind_real_{hid}.json"), "w"))
        make_tile(g, os.path.join(CACHE, f"tile_real_{hid}.png"), simplify=1.5)
        print("realprep", hid, len(g), "bld, FAR",
              round((g.fp_area * g.floors).sum() / AREA, 2))


# ---------------------------------------------------------------- TASK 1 recon
def cmd_recon(ids):
    rng = np.random.default_rng(SEED)
    A = load_adjacency()
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    logp = os.path.join(CACHE, "recon_grow_log.json")
    log = json.load(open(logp)) if os.path.exists(logp) else {}
    for hid in ids:
        target = np.bincount(load_grid(hid).ravel(), minlength=K)
        grid = grow_grid(target, A, rng)
        np.save(os.path.join(CACHE, f"recon_grid_{hid}.npy"), grid)
        gdf, fb = collage(grid, pools, idx, rng, all_train)
        save_canvas(gdf, f"generated_recon_{hid}", fgb=True)
        make_tile(gdf, os.path.join(CACHE, f"tile_recon_{hid}.png"), simplify=1.5)
        ach = np.bincount(grid.ravel(), minlength=K)
        log[hid] = {"target": target.tolist(), "achieved": ach.tolist()}
        print(hid, "target", target.tolist(), "achieved", ach.tolist(),
              "FAR", round(gdf.gfa.sum() / AREA, 2))
    json.dump(log, open(logp, "w"), indent=2)


def cmd_compare():
    rows = {}
    for hid in HELDOUT:
        gen = gpd.read_parquet(os.path.join(CACHE, f"generated_recon_{hid}.parquet"))
        rows[f"gen_{hid}"] = indicators(gen)
        rows[f"real_{hid}"] = json.load(
            open(os.path.join(CACHE, f"ind_real_{hid}.json"), encoding="utf-8"))
    ind = pd.DataFrame(rows).T
    z = (ind - ind.mean()) / ind.std(ddof=0).replace(0, 1)
    D = pd.DataFrame(index=HELDOUT, columns=HELDOUT, dtype=float)
    for a in HELDOUT:
        for b in HELDOUT:
            D.loc[a, b] = float(np.linalg.norm(z.loc[f"gen_{a}"] - z.loc[f"real_{b}"]))
    nearest = {a: D.loc[a].idxmin() for a in HELDOUT}
    hits = {a: bool(nearest[a] == a) for a in HELDOUT}
    grow_log = json.load(open(os.path.join(CACHE, "recon_grow_log.json")))
    payload = {
        "rule": ("OUT-OF-SAMPLE exam: heldout window's observable token composition -> "
                 "FREE growth on blank 20x20 (no skeleton) with TRAIN adjacency "
                 f"(beta={BETA}, {SWEEPS} random-order Gibbs sweeps, seed {SEED}) -> "
                 "cell collage from ALL-TRAIN pools (heldout owns no pool). "
                 "Distances: 10 indicators z-scored over the 12 fabrics."),
        "grow_target_vs_achieved": grow_log,
        "indicators": ind.round(4).to_dict(orient="index"),
        "distance_matrix": D.round(3).to_dict(orient="index"),
        "nearest": nearest, "hit": hits,
        "n_hits": int(sum(hits.values())), "hit_rate": f"{sum(hits.values())}/6",
        "v02_baseline": "3/3 but IN-sample (each window collaged from its own pool)",
    }
    merge_results("heldout_exam", payload)
    print(D.round(2).to_string())
    print("out-of-sample hits:", payload["hit_rate"])


def cmd_fig3():
    res = json.load(open(RESULTS, encoding="utf-8"))["heldout_exam"]
    ind = pd.DataFrame(res["indicators"]).T
    D = pd.DataFrame(res["distance_matrix"]).loc[HELDOUT, HELDOUT]
    nearest, hits = res["nearest"], res["hit"]

    fig = plt.figure(figsize=(19, 12.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.92], hspace=0.26, wspace=0.04)
    for j, hid in enumerate(HELDOUT):
        ir, ig = ind.loc[f"real_{hid}"], ind.loc[f"gen_{hid}"]
        axr = fig.add_subplot(gs[0, j])
        show_tile(axr, os.path.join(CACHE, f"tile_real_{hid}.png"))
        axr.set_title(f"REAL {hid} {SHORT[hid]}\nFAR {ir.FAR:.2f}  cov {ir.coverage:.2f}  "
                      f"h90 {ir.height_p90:.0f}m", fontsize=9)
        axg = fig.add_subplot(gs[1, j])
        show_tile(axg, os.path.join(CACHE, f"tile_recon_{hid}.png"))
        mark = "HIT" if hits[hid] else f"miss->{nearest[hid]}"
        axg.set_title(f"GEN {hid} [{mark}]\nFAR {ig.FAR:.2f} ({ir.FAR:.2f})  "
                      f"cov {ig.coverage:.2f} ({ir.coverage:.2f})\n"
                      f"h90 {ig.height_p90:.0f}m ({ir.height_p90:.0f})", fontsize=8.6,
                      color="#1b4332" if hits[hid] else "#9d0208")

    ax1 = fig.add_subplot(gs[2, 0:2])
    M = D.values.astype(float)
    im = ax1.imshow(M, cmap="RdYlGn_r")
    ax1.set_xticks(range(6), [f"real\n{h}" for h in HELDOUT], fontsize=7.5)
    ax1.set_yticks(range(6), [f"gen {h}" for h in HELDOUT], fontsize=7.5)
    for i, a in enumerate(HELDOUT):
        for j, b in enumerate(HELDOUT):
            star = "*" if (nearest[a] == b) else ""
            ax1.text(j, i, f"{M[i, j]:.1f}{star}", ha="center", va="center", fontsize=7.5,
                     fontweight="bold" if nearest[a] == b else "normal")
    ax1.set_title(f"z-scored 10-indicator distance  (* = nearest)\n"
                  f"out-of-sample hits {res['n_hits']}/6   "
                  f"(v0.2: 3/3 but in-sample)", fontsize=9.5)
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[2, 2:4])
    diag = [D.loc[h, h] for h in HELDOUT]
    offmin = [D.loc[h].drop(h).min() for h in HELDOUT]
    x = np.arange(6)
    ax2.bar(x - 0.19, diag, 0.38, label="d(gen, own real)", color="#1d3557")
    ax2.bar(x + 0.19, offmin, 0.38, label="min d(gen, other real)", color="#e76f51")
    ax2.set_xticks(x, HELDOUT, fontsize=8)
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("hit margin: own vs nearest other", fontsize=9.5)

    ax3 = fig.add_subplot(gs[2, 4:6])
    fr = [ind.loc[f"real_{h}", "FAR"] for h in HELDOUT]
    fg = [ind.loc[f"gen_{h}", "FAR"] for h in HELDOUT]
    lim = max(max(fr), max(fg)) * 1.15
    ax3.plot([0, lim], [0, lim], "--", color="grey", lw=0.8)
    ax3.scatter(fr, fg, color="#2a9d8f", s=45, zorder=3)
    for h, xr, yg in zip(HELDOUT, fr, fg):
        ax3.annotate(h, (xr, yg), textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax3.set_xlabel("real FAR", fontsize=9); ax3.set_ylabel("generated FAR", fontsize=9)
    ax3.set_xlim(0, lim); ax3.set_ylim(0, lim); ax3.grid(alpha=0.3)
    ax3.set_title("FAR parity (free growth, all-TRAIN pools)", fontsize=9.5)

    fig.suptitle("FIG3 — Held-out reconstruction exam: observable composition -> free growth "
                 "(TRAIN grammar) -> all-TRAIN cell collage vs reality", fontsize=14, y=0.985)
    fig.savefig(os.path.join(V03, "fig3_heldout_exam.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


# ---------------------------------------------------------------- TASK 2 transplant
def cmd_transplant(sigs):
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    inten = token_intensity()
    for a in sigs:
        demand = np.bincount(load_grid(a).ravel(), minlength=K)
        recs = {}
        for b in HELDOUT:
            cgfa = json.load(open(os.path.join(CACHE, f"cellgfa_{b}.json")))
            g = rank_match(load_grid(b), demand, cgfa, inten)
            gdf, fb = collage(g, pools, idx, rng, fixed(a))
            name = f"transplant_{a}_on_{b}"
            save_canvas(gdf, name)
            make_tile(gdf, os.path.join(CACHE, f"tile_tr_{a}_{b}.png"), simplify=2.0)
            rec = indicators(gdf)
            rec.update(n_buildings=int(len(gdf)), n_built_cells=int((g > 0).sum()),
                       n_site_built_cells=int((load_grid(b) > 0).sum()),
                       pool_fallback_cells=int(fb))
            recs[name] = {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in rec.items()}
            print(name, "FAR", recs[name]["FAR"], "h90", recs[name]["height_p90"],
                  "fallback", fb)
        json.dump(recs, open(os.path.join(CACHE, f"tr_{a}.json"), "w"), indent=2)


def cmd_fig4():
    _, sigw = load_signatures()
    table = {}
    for a in SIGS4:
        table.update(json.load(open(os.path.join(CACHE, f"tr_{a}.json"), encoding="utf-8")))
    real_ind = {h: json.load(open(os.path.join(CACHE, f"ind_real_{h}.json"))) for h in HELDOUT}
    # asymmetry vs what the site really holds: ratio of transplanted FAR to site real FAR
    ratios = {(a, b): table[f"transplant_{a}_on_{b}"]["FAR"] / real_ind[b]["FAR"]
              for a in SIGS4 for b in HELDOUT}
    hi = max(ratios, key=ratios.get)
    lo = min(ratios, key=ratios.get)
    payload = {
        "rule": ("signature = source window's token demand + SOURCE-window-only pools; "
                 "site = heldout window's real token grid (EMPTY preserved), built cells "
                 "rank-matched by real cell GFA vs token intensity; sparse demand "
                 "evacuates surplus cells, dense demand compressed to skeleton"),
        "asymmetry_metric": "FAR(transplant)/FAR(site real); max = strongest densification, min = strongest evacuation",
        "max_densification": {"pair": f"{hi[0]} on {hi[1]}", "ratio": round(ratios[hi], 2)},
        "max_evacuation": {"pair": f"{lo[0]} on {lo[1]}", "ratio": round(ratios[lo], 2)},
        "table": table,
    }
    merge_results("transplant_gallery", payload)

    fig = plt.figure(figsize=(19.5, 14.2))
    gs = fig.add_gridspec(4, 6, hspace=0.34, wspace=0.04, left=0.045, top=0.90)
    for i, a in enumerate(SIGS4):
        for j, b in enumerate(HELDOUT):
            ax = fig.add_subplot(gs[i, j])
            show_tile(ax, os.path.join(CACHE, f"tile_tr_{a}_{b}.png"))
            st = table[f"transplant_{a}_on_{b}"]
            rt = ratios[(a, b)]
            ax.set_title(f"FAR {st['FAR']:.2f} (site {real_ind[b]['FAR']:.2f}, x{rt:.1f})"
                         f"\nh90 {st['height_p90']:.0f} m", fontsize=8.6)
            if (a, b) == hi:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       ec="#9d0208", lw=3, clip_on=False))
                ax.text(0.5, -0.10, "MOST ASYMMETRIC: densification", fontsize=8.5,
                        color="#9d0208", fontweight="bold", ha="center",
                        transform=ax.transAxes)
            if (a, b) == lo:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       ec="#1d3557", lw=3, clip_on=False))
                ax.text(0.5, -0.10, "MOST ASYMMETRIC: evacuation", fontsize=8.5,
                        color="#1d3557", fontweight="bold", ha="center",
                        transform=ax.transAxes)
            if j == 0:
                ax.text(-0.08, 0.5, f"{a} {SHORT[a]}\nsignature", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=10,
                        fontweight="bold")
            if i == 0:
                ax.text(0.5, 1.24, f"site {b} {SHORT[b]}", transform=ax.transAxes,
                        ha="center", fontsize=10.5, fontweight="bold")
    fig.suptitle("FIG4 — Transplant gallery: 4 TRAIN signatures x 6 HELDOUT site skeletons "
                 "(rank-matched by real cell GFA; pools = source window only)",
                 fontsize=13.5, y=0.985)
    fig.text(0.5, 0.012, "Counterfactual collage on real heldout skeletons — illustrative, "
             "not a development forecast. xN = FAR vs the site's real FAR.",
             ha="center", fontsize=9.5, style="italic")
    fig.savefig(os.path.join(V03, "fig4_transplant_gallery.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print("fig4 saved; densification", payload["max_densification"],
          "evacuation", payload["max_evacuation"])


# ---------------------------------------------------------------- TASK 3 path
def pick_path_window():
    sig, sigw = load_signatures()
    slab_train = [w for w in sig["windows"] if w["role"] == "train"
                  and int(np.argmax(w["token_comp_T0_T8"][1:])) + 1 == 3]
    centre = np.mean([w["pca_xy"] for w in slab_train], axis=0)
    dists = {h: float(np.linalg.norm(np.array(sigw[h]["pca_xy"]) - centre))
             for h in PATH_CANDS}
    chosen = min(dists, key=dists.get)
    rationale = (f"slab-mat band = TRAIN windows whose largest non-EMPTY token share is T3 "
                 f"({[w['id'] for w in slab_train]}); band centre PCA "
                 f"({centre[0]:.2f},{centre[1]:.2f}); candidates = the four heldouts "
                 f"Agent 1 places in the slab band; distances "
                 f"{ {k: round(v, 3) for k, v in dists.items()} } -> {chosen}")
    return chosen, centre, dists, rationale


def cmd_path():
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    inten = token_intensity()
    chosen, centre, dists, rationale = pick_path_window()
    print("path window:", chosen, "|", rationale)
    c_src = np.bincount(load_grid(chosen).ravel(), minlength=K).astype(float)
    c_ljz = np.bincount(load_grid("A01").ravel(), minlength=K).astype(float)
    cgfa = json.load(open(os.path.join(CACHE, f"cellgfa_{chosen}.json")))
    site = load_grid(chosen)
    steps = []
    for k, w in enumerate(PATH_W):
        demand = largest_remainder((1 - w) * c_src + w * c_ljz, N * N)
        g = rank_match(site, demand, cgfa, inten)

        def chooser(rng, w=w):
            return "A01" if rng.random() < w else None   # None = ALL-TRAIN pool

        gdf, fb = collage(g, pools, idx, rng, chooser)
        save_canvas(gdf, f"path_step{k}")
        make_tile(gdf, os.path.join(CACHE, f"tile_path_{k}.png"), simplify=1.8)
        steps.append({
            "w": w, "FAR": round(float(gdf.gfa.sum() / AREA), 3),
            "coverage": round(float(gdf.fp_area.sum() / AREA), 3),
            "height_p90": round(float(np.percentile(gdf.height, 90)), 1),
            "tower_share": round(float((gdf.family == "tower").mean()), 3),
            "t8_cell_share": round(float((g == 8).sum() / (N * N)), 4),
            "n_built_cells": int((g > 0).sum()), "n_buildings": int(len(gdf)),
        })
        print(steps[-1])
    payload = {
        "chosen_window": chosen, "rationale": rationale,
        "band_centre_pca": [round(float(x), 3) for x in centre],
        "candidate_distances": {k: round(v, 3) for k, v in dists.items()},
        "rule": ("site skeleton = chosen heldout's real grid (EMPTY preserved); token "
                 "composition lerp (1-w)*own + w*A01, renormalized to 400 by largest "
                 "remainder, rank-matched by real cell GFA; pools per cell: P(A01-only)=w "
                 "else ALL-TRAIN (the heldout owns no pool). ILLUSTRATIVE, NOT PREDICTIVE."),
        "steps": steps,
    }
    merge_results("path_interpolation", payload)


def cmd_fig5():
    res = json.load(open(RESULTS, encoding="utf-8"))["path_interpolation"]
    chosen, steps = res["chosen_window"], res["steps"]
    sig, sigw = load_signatures()
    fig = plt.figure(figsize=(18.5, 9.8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.35, 0.85], hspace=0.22, wspace=0.04)
    for k, st in enumerate(steps):
        ax = fig.add_subplot(gs[0, k])
        show_tile(ax, os.path.join(CACHE, f"tile_path_{k}.png"))
        lab = (f"{chosen} {SHORT[chosen]} signature" if st["w"] == 0 else
               f"A01 {SHORT['A01']} signature" if st["w"] == 1 else f"w = {st['w']:.2f}")
        ax.set_title(f"step {k}  {lab}\nFAR {st['FAR']:.2f}  h90 {st['height_p90']:.0f} m",
                     fontsize=9.5)
    ws = [st["w"] for st in steps]
    axc = fig.add_subplot(gs[1, 0])
    axc.plot(ws, [st["FAR"] for st in steps], "o-", color="#1d3557", label="FAR")
    axc.plot(ws, [st["coverage"] for st in steps], "s--", color="#e76f51", label="coverage")
    axc.set_xlabel("w"); axc.legend(fontsize=8); axc.grid(alpha=0.3)
    axc.set_title("FAR / coverage", fontsize=9.5)
    axh = fig.add_subplot(gs[1, 1])
    axh.plot(ws, [st["height_p90"] for st in steps], "o-", color="#2a9d8f")
    axh.set_xlabel("w"); axh.grid(alpha=0.3); axh.set_title("height p90 (m)", fontsize=9.5)
    axt = fig.add_subplot(gs[1, 2])
    axt.plot(ws, [st["tower_share"] for st in steps], "o-", color="#9d0208",
             label="tower share (bld)")
    axt.plot(ws, [st["t8_cell_share"] for st in steps], "s--", color="#6a4c93",
             label="T8 cell share")
    axt.set_xlabel("w"); axt.grid(alpha=0.3); axt.legend(fontsize=8)
    axt.set_title("tower + T8 megaform", fontsize=9.5)
    axp = fig.add_subplot(gs[1, 3:5])
    for wd in sig["windows"]:
        x, y = wd["pca_xy"]
        if wd["role"] == "train":
            axp.scatter(x, y, marker="o", s=28, color="#9aa5b1", zorder=2)
        else:
            axp.scatter(x, y, marker="*", s=70, color="#e9c46a", zorder=3)
    x0, y0 = sigw[chosen]["pca_xy"]; x1, y1 = sigw["A01"]["pca_xy"]
    axp.plot([x0 + (x1 - x0) * w for w in ws], [y0 + (y1 - y0) * w for w in ws],
             "-o", color="#9d0208", lw=1.6, ms=5, zorder=4)
    axp.scatter([x0], [y0], marker="*", s=160, color="#9d0208", zorder=5)
    axp.scatter([x1], [y1], marker="D", s=60, color="#1d3557", zorder=5)
    axp.annotate(f"{chosen} {SHORT[chosen]}", (x0, y0), xytext=(6, -10),
                 textcoords="offset points", fontsize=8.5, color="#9d0208")
    axp.annotate("A01 Lujiazui", (x1, y1), xytext=(6, 4),
                 textcoords="offset points", fontsize=8.5, color="#1d3557")
    axp.set_xlabel("PC1 (intensity/verticality)", fontsize=8.5)
    axp.set_ylabel("PC2 (open-spiky vs dense-mat)", fontsize=8.5)
    axp.grid(alpha=0.3)
    axp.set_title("path on the signature cloud (PCA, TRAIN-fit; lerp projects to a "
                  "straight line)", fontsize=9.5)
    fig.suptitle(f"FIG5 — Continuous path on the cloud: {chosen} {SHORT[chosen]} signature "
                 "interpolated toward A01 Lujiazui on its own skeleton", fontsize=13.5, y=0.99)
    fig.text(0.5, 0.012, "ILLUSTRATIVE, NOT PREDICTIVE — proxy endpoint, single seed, "
             "no tenure / street / parcel constraints.", ha="center", fontsize=10,
             style="italic", fontweight="bold")
    fig.savefig(os.path.join(V03, "fig5_path.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig5 saved")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    args = sys.argv[1:]
    cmd, rest = args[0], args[1:]
    if cmd == "realprep":
        cmd_realprep(rest or None)
    elif cmd == "recon":
        cmd_recon(rest)
    elif cmd == "compare":
        cmd_compare()
    elif cmd == "fig3":
        cmd_fig3()
    elif cmd == "transplant":
        cmd_transplant(rest)
    elif cmd == "fig4":
        cmd_fig4()
    elif cmd == "path":
        cmd_path()
    elif cmd == "fig5":
        cmd_fig5()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
