#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_generate_v04.py — v0.4 generation side: keep-out collage on typed voids.

OWNER DIRECTIVE (2026-06-12): generation may occupy the original built places
but NOT the road (nor the river). Two mechanisms, both implemented here on top
of the v0.3 grammar (built tokens T1..T8, pools, β=8 Gibbs — all verbatim):

  (1) typed-void skeleton: E_ROAD / E_WATER / E_GREEN cells (run_v04.py) are
      FIXED during growth — the multiset of {built tokens + E_VACANT} is grown
      only on the remaining cells, with the 12-token TRAIN adjacency.
  (2) keep-out collage: every pasted footprint is tested against the site's
      HARD no-build mask (road ∪ water, sub-cell resolution). Per cell the
      best of TRIES pool draws (least violating area) is kept, then buildings
      with > VIOL_DROP of their own footprint inside the mask are dropped
      (logged: n_dropped, dropped GFA share). Green is typed at cell level
      but NOT keep-out (real parks legitimately contain buildings).

Subcommands:
  exam               TASK 1' — held-out reconstruction, typed-skeleton variant.
                     HONESTY: this is an EASIER exam than v0.3's free growth —
                     road/water/green cell positions are public observable data,
                     but they do hand the generator part of the spatial
                     structure (PC2 is void-dominated). Both numbers reported
                     side by side; they are NOT the same test.
  exam_baseline      v0.3 exam re-run verbatim (free growth, 9-token adjacency,
                     no keep-out) for pipeline-integrity verification only.
  transplant         TASK 2' — the v0.3 4x6 transplant with keep-out collage;
                     site built cells = typed-grid tokens in 1..8.
  fig3 / fig4        exam figure / before-after transplant gallery.

seed 0 (each subcommand reseeds default_rng(0)). FlatGeobuf via /tmp, no GPKG.
"""
import json, os, shutil, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.affinity import translate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
V04 = os.path.join(TF, "v04")
C3 = os.path.join(V03, "cache_gen")
CACHE = os.path.join(V04, "cache_gen")
MASKC = os.path.join(V04, "cache")
os.makedirs(CACHE, exist_ok=True)
RESULTS = os.path.join(V04, "results_v04.json")

HELDOUT = ["A02", "A07", "A20", "A29", "A35", "A43"]
SIGS4 = ["A01", "A10", "A36", "A26"]
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
SHORT = {"A01": "Lujiazui CBD", "A02": "Nanjing Rd E.", "A07": "Bund",
         "A10": "Laoximen lilong", "A20": "Sanlin", "A26": "Jiading new town",
         "A29": "Nanqiao", "A35": "Caohejing", "A36": "Zhangjiang ind.",
         "A43": "Liangcheng"}

NTOK = 12              # 0=E_VAC, 1..8 built, 9=E_ROAD, 10=E_WATER, 11=E_GREEN
CAND = list(range(9))  # classes growth may place: E_VACANT + built
FIXED_TOKENS = (9, 10, 11)
SEED = 0
SWEEPS = 5
EPS = 0.01
BETA = 8.0
CELL = 50.0
CANVAS = 1000.0
AREA = 1_000_000.0
N = 20
TRIES = 8              # pool draws per cell, keep least-violating
VIOL_DROP = 0.2        # drop building if > this fraction of footprint in hard mask
HCAP, ZMAX = 150.0, 350.0
ELEV, AZIM = 38, -60


# ---------------------------------------------------------------- assets
def load_split():
    return json.load(open(os.path.join(V03, "split.json"), encoding="utf-8"))


def wmap():
    return {w["id"]: w for w in load_split()["windows"]}


def load_adjacency12():
    adj = json.load(open(os.path.join(V04, "adjacency_v04.json"), encoding="utf-8"))
    return np.array(adj["row_normalized"], dtype=float)


def load_grid4(wid):
    return np.load(os.path.join(V04, f"token_grid_v04_{wid}.npy"))


def load_hard_mask(wid, local=True):
    """Keep-out mask = MAJOR roads ∪ water in canvas-local coords (calibrated in
    run_v04.py calib: minor lanes are fabric texture pasted cells already carry).
    None when the window has neither."""
    g = gpd.read_parquet(os.path.join(MASKC, f"{wid}_mask.parquet"))
    g = g[g["cls"].isin(["road_major", "water"])]
    if not len(g):
        return None
    geom = shapely.unary_union(list(g.geometry))
    if local:
        w = wmap()[wid]
        geom = translate(geom, xoff=-(w["cx"] - 500.0), yoff=-(w["cy"] - 500.0))
    return geom


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


def token_intensity():
    tk = json.load(open(os.path.join(V03, "tokens.json"), encoding="utf-8"))
    return {0: 0.0, **{t: float(tk["stats"][str(t)]["gfa"]) for t in range(1, 9)}}


def merge_results(key, payload):
    res = json.load(open(RESULTS, encoding="utf-8")) if os.path.exists(RESULTS) else {}
    res[key] = payload
    json.dump(res, open(RESULTS, "w"), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- growth with fixed cells
def grow_grid_fixed(init_grid, fixed_mask, target_counts, A, rng, sweeps=SWEEPS):
    """v0.2/v0.3 Gibbs verbatim except: cells where fixed_mask is True are never
    resampled (their tokens — typed voids — still vote as neighbours), and the
    candidate set is CAND (E_VACANT + built). target_counts is over CAND on the
    free cells only."""
    grid = init_grid.copy()
    target = np.asarray(target_counts, dtype=float)
    share = np.maximum(target, 1e-9) / max(target.sum(), 1e-9)
    Ae = A + EPS
    free = [(r, c) for r in range(N) for c in range(N) if not fixed_mask[r, c]]
    counts = np.bincount([grid[r, c] for r, c in free], minlength=len(CAND)).astype(float)
    for _ in range(sweeps):
        for i in rng.permutation(len(free)):
            r, c = free[i]
            like = np.ones(len(CAND))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < N and 0 <= cc < N:
                    like *= Ae[CAND, grid[rr, cc]]
            old = grid[r, c]
            counts[old] -= 1
            prior = share * (target / np.maximum(counts, 0.5)) ** BETA
            p = like * prior
            p /= p.sum()
            new = int(rng.choice(len(CAND), p=p))
            grid[r, c] = new
            counts[new] += 1
    return grid


def init_fixed(typed_grid, rng):
    """Skeleton = typed voids fixed; the free multiset (E_VAC + built, exact
    composition of the real window's free cells) shuffled over free cells."""
    fixed_mask = np.isin(typed_grid, FIXED_TOKENS)
    free_vals = typed_grid[~fixed_mask]
    target = np.bincount(free_vals, minlength=len(CAND))
    seq = free_vals.copy()
    rng.shuffle(seq)
    grid = typed_grid.copy()
    grid[~fixed_mask] = seq
    return grid, fixed_mask, target


# ---------------------------------------------------------------- keep-out collage
def collage_keepout(grid, pools, idx, rng, src_window, hard_mask):
    """v0.3 collage + keep-out: per cell pick the least-violating of TRIES draws
    against hard_mask (road ∪ water, canvas-local), then drop buildings with
    > VIOL_DROP of their footprint inside it. hard_mask=None -> plain collage."""
    by_wt, by_t, members = idx
    clipbox = box(0, 0, CANVAS, CANVAS)
    rows, fallbacks, n_drop, gfa_drop, gfa_tot = [], 0, 0, 0.0, 0.0
    # per-cell local mask pieces (overhang reach ~150 m beyond the cell)
    pieces = {}
    if hard_mask is not None:
        for r in range(N):
            for c in range(N):
                if int(grid[r, c]) in (0, 9, 10, 11):
                    continue
                cb = box(c * CELL - 80, r * CELL - 80, (c + 1) * CELL + 160, (r + 1) * CELL + 160)
                piece = hard_mask.intersection(cb)
                if not piece.is_empty:
                    pieces[(r, c)] = piece
    for r in range(N):
        for c in range(N):
            t = int(grid[r, c])
            if t in (0, 9, 10, 11):
                continue
            w = src_window(rng)
            keys = by_wt.get((w, t)) if w is not None else by_t[t]
            if not keys:
                keys = by_t[t]
                fallbacks += 1
            piece = pieces.get((r, c))
            n_try = TRIES if piece is not None else 1
            best = None
            for _ in range(n_try):
                key = keys[int(rng.integers(len(keys)))]
                sub = pools.loc[members[key]]
                geoms = [translate(g, xoff=c * CELL, yoff=r * CELL) for g in sub.geometry]
                if piece is not None:
                    viol = sum(g.intersection(piece).area for g in geoms)
                else:
                    viol = 0.0
                if best is None or viol < best[0]:
                    best = (viol, key, sub, geoms)
                if viol == 0.0:
                    break
            _, key, sub, geoms = best
            for (_, b), geom in zip(sub.iterrows(), geoms):
                geom = geom.intersection(clipbox)
                if geom.is_empty or geom.area < 1.0:
                    continue
                gfa_b = geom.area * int(b.floors)
                gfa_tot += gfa_b
                if piece is not None:
                    vf = geom.intersection(piece).area / geom.area
                    if vf > VIOL_DROP:
                        n_drop += 1
                        gfa_drop += gfa_b
                        continue
                rows.append(dict(geometry=geom, height=float(b.height),
                                 floors=int(b.floors), family=b.family,
                                 semantic=b.semantic, token=t, src_window=b.window,
                                 src_key=key, tgt_row=r, tgt_col=c))
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:32651")
    gdf["fp_area"] = gdf.geometry.area
    gdf["gfa"] = gdf["fp_area"] * gdf["floors"]
    log = {"fallbacks": int(fallbacks), "n_dropped": int(n_drop),
           "dropped_gfa_share": round(float(gfa_drop / gfa_tot), 4) if gfa_tot else 0.0}
    return gdf, log


def all_train(rng):
    return None


def fixed_w(wid):
    return lambda rng: wid


# ---------------------------------------------------------------- indicators / render
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


def render_axon(ax, gdf, simplify=1.5):    # verbatim v0.3
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


def plot_fg(ax, gdf, hard_mask, title, fs=8.6):
    """2D figure-ground + hard mask: roads/water gray-blue under black footprints."""
    if hard_mask is not None:
        parts = ([hard_mask] if hard_mask.geom_type == "Polygon"
                 else list(hard_mask.geoms))
        for p in parts:
            if p.geom_type != "Polygon":
                continue
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color="#b8cbe0", lw=0, zorder=1)
    for geom in gdf.geometry:
        gs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for p in gs:
            if p.geom_type != "Polygon":
                continue
            xs, ys = p.exterior.xy
            ax.fill(xs, ys, color="#1a1a1a", lw=0, zorder=2)
    ax.set_xlim(0, CANVAS); ax.set_ylim(0, CANVAS)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=fs)


# ---------------------------------------------------------------- EXAM (typed skeleton)
def cmd_exam():
    rng = np.random.default_rng(SEED)
    A = load_adjacency12()
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    log = {}
    for hid in HELDOUT:
        g4 = load_grid4(hid)
        init, fmask, target = init_fixed(g4, rng)
        grid = grow_grid_fixed(init, fmask, target, A, rng)
        np.save(os.path.join(CACHE, f"exam_grid_{hid}.npy"), grid)
        hard = load_hard_mask(hid)
        gdf, klog = collage_keepout(grid, pools, idx, rng, all_train, hard)
        gdf.to_parquet(os.path.join(CACHE, f"generated_exam_{hid}.parquet"))
        tmp = f"/tmp/generated_exam_{hid}.fgb"
        gdf.to_file(tmp, driver="FlatGeobuf")
        shutil.copy2(tmp, os.path.join(V04, f"generated_exam_{hid}.fgb"))
        make_tile(gdf, os.path.join(CACHE, f"tile_exam_{hid}.png"), simplify=1.5)
        ach = np.bincount(grid[~fmask].ravel(), minlength=9)
        log[hid] = {"target_free_cells": target.tolist(), "achieved": ach.tolist(),
                    "n_fixed_void_cells": int(fmask.sum()), **klog}
        print(hid, "fixed", int(fmask.sum()), "drop", klog["n_dropped"],
              f"({klog['dropped_gfa_share']:.1%} GFA)",
              "FAR", round(float(gdf.gfa.sum() / AREA), 2))
    json.dump(log, open(os.path.join(CACHE, "exam_log.json"), "w"), indent=2)

    rows = {}
    for hid in HELDOUT:
        gen = gpd.read_parquet(os.path.join(CACHE, f"generated_exam_{hid}.parquet"))
        rows[f"gen_{hid}"] = indicators(gen)
        rows[f"real_{hid}"] = json.load(
            open(os.path.join(C3, f"ind_real_{hid}.json"), encoding="utf-8"))
    ind = pd.DataFrame(rows).T
    z = (ind - ind.mean()) / ind.std(ddof=0).replace(0, 1)
    D = pd.DataFrame(index=HELDOUT, columns=HELDOUT, dtype=float)
    for a in HELDOUT:
        for b in HELDOUT:
            D.loc[a, b] = float(np.linalg.norm(z.loc[f"gen_{a}"] - z.loc[f"real_{b}"]))
    nearest = {a: D.loc[a].idxmin() for a in HELDOUT}
    hits = {a: bool(nearest[a] == a) for a in HELDOUT}
    v03_exam = json.load(open(os.path.join(V03, "results_v03.json"),
                              encoding="utf-8"))["heldout_exam"]
    payload = {
        "rule": ("TYPED-SKELETON exam: E_ROAD/E_WATER/E_GREEN cells of the real "
                 "typed grid are FIXED (public observable data); the free multiset "
                 "(E_VACANT + built composition of the free cells) grows with the "
                 f"12-token TRAIN adjacency (beta={BETA}, {SWEEPS} sweeps, seed {SEED}); "
                 "all-TRAIN cell collage with keep-out vs the site's road∪water mask "
                 f"(best of {TRIES} draws, drop if >{VIOL_DROP:.0%} footprint inside). "
                 "EASIER than v0.3 free growth — the void layout (PC2's main driver) "
                 "is partly given. NOT the same test; both reported."),
        "grow_and_keepout_log": log,
        "indicators": ind.round(4).to_dict(orient="index"),
        "distance_matrix": D.round(3).to_dict(orient="index"),
        "nearest": nearest, "hit": hits,
        "n_hits": int(sum(hits.values())), "hit_rate": f"{sum(hits.values())}/6",
        "v03_free_growth_baseline": {
            "hit_rate": v03_exam["hit_rate"], "nearest": v03_exam["nearest"],
            "distance_matrix": v03_exam["distance_matrix"]},
    }
    merge_results("exam_typed_skeleton", payload)
    print(D.round(2).to_string())
    print("typed-skeleton hits:", payload["hit_rate"], "| v0.3 free growth:",
          v03_exam["hit_rate"])


# ---------------------------------------------------------------- SPATIAL metric
def cmd_spatial():
    """What the 10-indicator distance cannot see: (a) does the generated fabric
    put GFA where the real window puts it (Pearson r over the 400-cell GFA
    maps), and (b) how much generated GFA stands inside the hard no-build mask.
    Computed for v0.3 free-growth recon and v0.4 typed-skeleton exam."""
    out = {}
    for hid in HELDOUT:
        cgfa = json.load(open(os.path.join(C3, f"cellgfa_{hid}.json"), encoding="utf-8"))
        real = np.zeros((N, N))
        for k, v in cgfa.items():
            r, c = map(int, k.split("_"))
            real[r, c] = v
        hard = load_hard_mask(hid)
        rec = {}
        for tag, path in (("v03_free", os.path.join(C3, f"generated_recon_{hid}.parquet")),
                          ("v04_typed", os.path.join(CACHE, f"generated_exam_{hid}.parquet"))):
            g = gpd.read_parquet(path)
            gen = np.zeros((N, N))
            for (r, c), v in g.groupby(["tgt_row", "tgt_col"])["gfa"].sum().items():
                gen[r, c] = v
            rr = float(np.corrcoef(real.ravel(), gen.ravel())[0, 1])
            if hard is not None:
                viol = float((g.geometry.intersection(hard).area * g["floors"]).sum()
                             / g["gfa"].sum())
            else:
                viol = 0.0
            rec[tag] = {"cell_gfa_pearson_r": round(rr, 4),
                        "gfa_share_in_mask": round(viol, 4)}
        out[hid] = rec
        print(hid, "r:", rec["v03_free"]["cell_gfa_pearson_r"], "->",
              rec["v04_typed"]["cell_gfa_pearson_r"],
              "| GFA in mask:", rec["v03_free"]["gfa_share_in_mask"], "->",
              rec["v04_typed"]["gfa_share_in_mask"])
    merge_results("spatial_metric", {
        "note": ("The aspatial 10-indicator distance cannot credit the typed-void/"
                 "keep-out fix; these two numbers can. cell_gfa_pearson_r: 400-cell "
                 "GFA-map correlation generated-vs-real (v0.3 free growth has no "
                 "spatial anchor, expected ~0). gfa_share_in_mask: generated GFA "
                 "standing in road_major ∪ water."),
        "windows": out,
    })


# ---------------------------------------------------------------- TRANSPLANT keep-out
def largest_remainder(frac, total):
    base = np.floor(frac).astype(int)
    rem = total - base.sum()
    if rem > 0:
        order = np.argsort(-(frac - base))
        base[order[:rem]] += 1
    return base


def rank_match_v04(site_typed, demand_built, cgfa, inten):
    """v0.3 rank_match with explicit built mask: site built cells = typed tokens
    1..8 (typed voids 9/10/11 are NOT buildable). Sparse demand evacuates to
    E_VACANT(0); dense demand compressed."""
    built_cells = [(r, c) for r in range(N) for c in range(N)
                   if 1 <= site_typed[r, c] <= 8]
    nB = len(built_cells)
    d = np.asarray(demand_built, dtype=float)
    n_built = d.sum()
    if n_built > nB:
        d = largest_remainder(d * nB / n_built, nB)
    else:
        d = largest_remainder(d, int(round(n_built)))
    seq = []
    for t in sorted(range(1, 9), key=lambda t: -inten[t]):
        seq += [t] * int(d[t - 1])
    seq += [0] * (nB - len(seq))
    order = sorted(built_cells, key=lambda rc: -cgfa.get(f"{rc[0]}_{rc[1]}", 0.0))
    out = site_typed.copy()
    for (r, c), t in zip(order, seq):
        out[r, c] = t
    return out


def cmd_transplant():
    rng = np.random.default_rng(SEED)
    pools = load_pools_geo()
    idx = build_cell_index(pools)
    inten = token_intensity()
    v03_table = json.load(open(os.path.join(V03, "results_v03.json"),
                               encoding="utf-8"))["transplant_gallery"]["table"]
    real_ind = {h: json.load(open(os.path.join(C3, f"ind_real_{h}.json"),
                                  encoding="utf-8")) for h in HELDOUT}
    table = {}
    for a in SIGS4:
        demand_built = np.bincount(load_grid4(a).ravel(), minlength=NTOK)[1:9]
        for b in HELDOUT:
            cgfa = json.load(open(os.path.join(C3, f"cellgfa_{b}.json"), encoding="utf-8"))
            site = load_grid4(b)
            g = rank_match_v04(site, demand_built, cgfa, inten)
            hard = load_hard_mask(b)
            gdf, klog = collage_keepout(g, pools, idx, rng, fixed_w(a), hard)
            name = f"transplant_{a}_on_{b}"
            gdf.to_parquet(os.path.join(CACHE, f"{name}.parquet"))
            rec = indicators(gdf)
            rec.update(n_buildings=int(len(gdf)), **klog,
                       FAR_v03_no_keepout=v03_table[name]["FAR"],
                       FAR_site_real=real_ind[b]["FAR"])
            table[name] = {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in rec.items()}
            print(name, "FAR", table[name]["FAR"],
                  "(v0.3", v03_table[name]["FAR"], ") drop",
                  klog["n_dropped"], f"({klog['dropped_gfa_share']:.1%})")
    far_deltas = {k: round(v["FAR"] - v["FAR_v03_no_keepout"], 3) for k, v in table.items()}
    merge_results("transplant_keepout", {
        "rule": ("v0.3 transplant verbatim (source token demand, source-window pools, "
                 "rank-match by real cell GFA) except: site built cells = typed-grid "
                 "tokens 1..8, and the collage keeps out of the site's road∪water "
                 f"mask (best of {TRIES} draws, drop >{VIOL_DROP:.0%} violators)."),
        "table": table,
        "FAR_delta_vs_v03": far_deltas,
    })


# ---------------------------------------------------------------- FIG 3 (exam)
def cmd_fig3():
    res = json.load(open(RESULTS, encoding="utf-8"))["exam_typed_skeleton"]
    ind = pd.DataFrame(res["indicators"]).T
    D = pd.DataFrame(res["distance_matrix"]).loc[HELDOUT, HELDOUT]
    D3 = pd.DataFrame(res["v03_free_growth_baseline"]["distance_matrix"]).loc[HELDOUT, HELDOUT]
    nearest, hits = res["nearest"], res["hit"]
    log = res["grow_and_keepout_log"]

    fig = plt.figure(figsize=(19, 12.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 0.92], hspace=0.27, wspace=0.04)
    for j, hid in enumerate(HELDOUT):
        ir, ig = ind.loc[f"real_{hid}"], ind.loc[f"gen_{hid}"]
        axr = fig.add_subplot(gs[0, j])
        show_tile(axr, os.path.join(C3, f"tile_real_{hid}.png"))
        axr.set_title(f"REAL {hid} {SHORT[hid]}\nFAR {ir.FAR:.2f}  cov {ir.coverage:.2f}  "
                      f"h90 {ir.height_p90:.0f}m", fontsize=9)
        axg = fig.add_subplot(gs[1, j])
        show_tile(axg, os.path.join(CACHE, f"tile_exam_{hid}.png"))
        mark = "HIT" if hits[hid] else f"miss->{nearest[hid]}"
        axg.set_title(
            f"GEN {hid} [{mark}]  fixed voids {log[hid]['n_fixed_void_cells']}\n"
            f"FAR {ig.FAR:.2f} ({ir.FAR:.2f})  cov {ig.coverage:.2f} ({ir.coverage:.2f})\n"
            f"dropped {log[hid]['n_dropped']} bld "
            f"({log[hid]['dropped_gfa_share']:.1%} GFA)",
            fontsize=8.4, color="#1b4332" if hits[hid] else "#9d0208")

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
    ax1.set_title(f"typed-skeleton exam distances (* = nearest)\nhits {res['n_hits']}/6",
                  fontsize=9.5)
    fig.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[2, 2:4])
    sp = json.load(open(RESULTS, encoding="utf-8"))["spatial_metric"]["windows"]
    x = np.arange(6)
    v3 = [sp[h]["v03_free"]["gfa_share_in_mask"] for h in HELDOUT]
    v4 = [sp[h]["v04_typed"]["gfa_share_in_mask"] for h in HELDOUT]
    ax2.bar(x - 0.19, v3, 0.38, label="v0.3 free growth", color="#9d0208")
    ax2.bar(x + 0.19, v4, 0.38, label="v0.4 typed + keep-out", color="#1b4332")
    for xi, h in zip(x, HELDOUT):
        r3 = sp[h]["v03_free"]["cell_gfa_pearson_r"]
        r4 = sp[h]["v04_typed"]["cell_gfa_pearson_r"]
        ax2.text(xi, max(v3[xi], 0.02) + 0.015, f"r {r3:.2f}->{r4:.2f}",
                 ha="center", fontsize=6.6, rotation=90)
    ax2.set_xticks(x, HELDOUT, fontsize=8)
    ax2.set_ylabel("generated GFA standing in road∪water", fontsize=8.5)
    ax2.legend(fontsize=7.5); ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("what the 10 indicators cannot see: GFA in the no-build "
                  "mask\n(text: 400-cell GFA-map correlation, v0.3 -> v0.4)", fontsize=9.5)

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
    ax3.set_title("FAR parity (typed skeleton + keep-out)", fontsize=9.5)

    fig.suptitle("v0.4 FIG 3 — Typed-skeleton exam: public void structure given, "
                 "fabric grown with 12-token grammar, keep-out collage  "
                 f"[{res['n_hits']}/6 vs v0.3 free-growth "
                 f"{res['v03_free_growth_baseline']['hit_rate']} — different tests]",
                 fontsize=13.5, y=0.985)
    fig.savefig(os.path.join(V04, "fig3_typed_exam.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig3 saved")


# ---------------------------------------------------------------- FIG 4 (before/after)
def cmd_fig4():
    res = json.load(open(RESULTS, encoding="utf-8"))["transplant_keepout"]
    table = res["table"]
    flagship = [("A01", "A20"), ("A36", "A02")]   # max densification / max evacuation in v0.3
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 4, hspace=0.22, wspace=0.08)
    for j, (a, b) in enumerate(flagship):
        hard = load_hard_mask(b)
        old = gpd.read_parquet(os.path.join(C3, f"transplant_{a}_on_{b}.parquet"))
        new = gpd.read_parquet(os.path.join(CACHE, f"transplant_{a}_on_{b}.parquet"))
        st = table[f"transplant_{a}_on_{b}"]
        ax = fig.add_subplot(gs[j, 0])
        plot_fg(ax, old, hard, f"{a} {SHORT[a]} on {b} {SHORT[b]}\n"
                f"v0.3 NO keep-out — FAR {st['FAR_v03_no_keepout']:.2f}", fs=9)
        ax = fig.add_subplot(gs[j, 1])
        plot_fg(ax, new, hard, f"v0.4 WITH keep-out — FAR {st['FAR']:.2f}\n"
                f"dropped {st['n_dropped']} bld ({st['dropped_gfa_share']:.1%} GFA)", fs=9)
        # violation closeups: violating buildings of the OLD canvas highlighted
        ax = fig.add_subplot(gs[j, 2])
        if hard is not None:
            viol = old.geometry.intersection(hard).area / old.geometry.area
            bad = old[viol > VIOL_DROP]
            plot_fg(ax, old[viol <= VIOL_DROP], hard,
                    f"v0.3 violators highlighted: {len(bad)} bld\n"
                    f"({float(viol[viol > VIOL_DROP].mean() if len(bad) else 0):.0%} "
                    "mean footprint-in-mask)", fs=9)
            for geom in bad.geometry:
                gp = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                for p in gp:
                    if p.geom_type != "Polygon":
                        continue
                    xs, ys = p.exterior.xy
                    ax.fill(xs, ys, color="#d62828", lw=0, zorder=3)
        ax = fig.add_subplot(gs[j, 3])
        deltas = [v for k, v in res["FAR_delta_vs_v03"].items()]
        ax.hist(deltas, bins=12, color="#457b9d", edgecolor="white")
        ax.axvline(res["FAR_delta_vs_v03"][f"transplant_{a}_on_{b}"], color="#d62828", lw=2)
        ax.set_xlabel("FAR delta (keep-out − v0.3)", fontsize=8.5)
        ax.set_title("all 24 transplant cells\n(red = this pair)", fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("v0.4 FIG 4 — Keep-out before/after on the flagship transplants: "
                 "occupy the built places, not the road or the river "
                 "(mask = blue underlay; red = v0.3 buildings standing in it)",
                 fontsize=13, y=0.985)
    fig.text(0.5, 0.012, "Counterfactual collage — illustrative, not a development "
             "forecast. Keep-out = best-of-8 pool draws, drop footprints >20% inside "
             "road∪water. Green typed at cell level only.",
             ha="center", fontsize=9.5, style="italic")
    fig.savefig(os.path.join(V04, "fig4_keepout_transplant.png"), dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print("fig4 saved")


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "exam":
        cmd_exam()
    elif cmd == "spatial":
        cmd_spatial()
    elif cmd == "transplant":
        cmd_transplant()
    elif cmd == "fig3":
        cmd_fig3()
    elif cmd == "fig4":
        cmd_fig4()
    else:
        raise SystemExit(f"unknown subcommand {cmd}")
