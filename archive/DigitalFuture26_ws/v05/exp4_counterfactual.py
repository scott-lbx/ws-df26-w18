#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp4_counterfactual.py — v0.5 Experiment 4: counterfactual generation.
Same site (A20 Sanlin), same built-cell skeleton, three power configurations.

Chain (the probe's Generate, upgraded to v0.4 infrastructure):
  power config -> semantic mix (literature priors, no_age cell for estates)
               -> family GFA demand (GFA-weighted row-normalized bridge, E2)
               -> token cell demand (NNLS vs token family-share x mean cell GFA,
                  rescaled to the site's 293 built cells — disclosed)
               -> rank-match onto A20's typed skeleton (v0.4 grid, real cell-GFA order)
               -> keep-out collage from ALL-TRAIN pools vs A20's road∪water mask
               -> rendered canvas + 10 indicators + direction checks.

ILLUSTRATIVE, NOT PREDICTIVE — literature priors, single seed, one site.
seed 0; a FRESH default_rng(0) per config so configs differ only by demand.
Outputs: V05/results_e4.json, V05/fig/fig_e4_counterfactual.png,
         V05/cache_gen/cf_<CFG>.parquet + tile pngs.
"""
import json, os, sys
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
from shapely.affinity import translate
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
V04 = os.path.join(TF, "v04")
V05 = os.path.join(TF, "v05")
C3 = os.path.join(V03, "cache_gen")
MASKC = os.path.join(V04, "cache")
CACHE = os.path.join(V05, "cache_gen")
FIG = os.path.join(V05, "fig")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
RESULTS = os.path.join(V05, "results_e4.json")

SITE = "A20"                       # Sanlin (heldout) — counterfactual site
SEED = 0
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]
SEMS = ["residential_estate", "residential_lowrise", "commercial_retail",
        "office_finance", "industrial", "public_admin", "education", "healthcare"]
SH_KEYS = ["sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
PUBLIC_SEMS = {"public_admin", "education", "healthcare"}

CONFIGS = {  # additive, non-exclusive — the probe's three
    "GOV": {"government": .8, "developer_capital": .2, "resident_community": .3, "informal": .0},
    "DEV": {"government": .3, "developer_capital": .8, "resident_community": .2, "informal": .0},
    "RES": {"government": .2, "developer_capital": .2, "resident_community": .8, "informal": .4},
}
CFG_COLORS = {"GOV": "#1d3557", "DEV": "#9d0208", "RES": "#1b4332"}

# ---------------------------------------------------------------------------
# Helpers copied VERBATIM from TF/v04/run_generate_v04.py (provenance: that
# script, 2026-06-12 owner-directive version) — load_split, wmap, load_grid4,
# load_hard_mask, load_pools_geo, build_cell_index, token_intensity,
# largest_remainder, rank_match_v04, collage_keepout, all_train, indicators,
# render_axon, make_tile, show_tile — plus the constants they reference.
# ---------------------------------------------------------------------------
NTOK = 12              # 0=E_VAC, 1..8 built, 9=E_ROAD, 10=E_WATER, 11=E_GREEN
CELL = 50.0
CANVAS = 1000.0
AREA = 1_000_000.0
N = 20
TRIES = 8              # pool draws per cell, keep least-violating
VIOL_DROP = 0.2        # drop building if > this fraction of footprint in hard mask
HCAP, ZMAX = 150.0, 350.0
ELEV, AZIM = 38, -60


def load_split():
    return json.load(open(os.path.join(V03, "split.json"), encoding="utf-8"))


def wmap():
    return {w["id"]: w for w in load_split()["windows"]}


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
# ------------------------------------------------------ end verbatim helpers


# ---------------------------------------------------------------- chain steps
def semantic_mixes(priors):
    """score_s = sum_dim config[dim] * prior_w[s][dim]; estates use the no_age
    cell (this counterfactual posits no age information); linear normalize."""
    out = {}
    for cfg, cw in CONFIGS.items():
        scores = []
        for s in SEMS:
            p = priors["priors"][s]
            w = p["cells"]["no_age"]["w"] if p.get("time_conditioned") else p["w"]
            scores.append(sum(cw[d] * w[d] for d in priors["dims"]))
        scores = np.asarray(scores, dtype=float)
        out[cfg] = scores / scores.sum()
    return out


def family_demand(mix, bridge_gfa):
    """mix (1x8) x GFA-weighted row-normalized bridge (8x5) -> family share f."""
    B = np.array([[bridge_gfa[s][f] for f in FAMILIES] for s in SEMS], dtype=float)
    f = mix @ B
    return f / f.sum()


def token_cell_demand(f, G, nB):
    """NNLS: minimize || (M^T diag(g)) n - G*f ||, n >= 0.
    M = per-token family footprint shares (tokens.json sh_*), g = per-token mean
    cell GFA. Then rescale n to sum to nB built cells, largest-remainder round."""
    tk = json.load(open(os.path.join(V03, "tokens.json"), encoding="utf-8"))
    M = np.array([[float(tk["stats"][str(t)][k]) for k in SH_KEYS]
                  for t in range(1, 9)], dtype=float)          # (8 tokens, 5 fams)
    g = np.array([float(tk["stats"][str(t)]["gfa"]) for t in range(1, 9)])
    A = M.T * g[None, :]                                       # (5, 8)
    b = G * np.asarray(f, dtype=float)
    n_raw, resid = nnls(A, b)
    scale = nB / n_raw.sum()
    n_int = largest_remainder(n_raw * scale, nB)
    return n_raw, float(resid), scale, n_int, M, g


# ---------------------------------------------------------------- main run
def main():
    priors = json.load(open(os.path.join(V05, "power_priors_v05.json"), encoding="utf-8"))
    e2 = json.load(open(os.path.join(V05, "results_e2.json"), encoding="utf-8"))
    bridge_gfa = e2["bridge_train"]["gfa_row_normalized"]

    site = load_grid4(SITE)
    nB = int(((site >= 1) & (site <= 8)).sum())
    cgfa = json.load(open(os.path.join(C3, f"cellgfa_{SITE}.json"), encoding="utf-8"))
    G = float(sum(cgfa.values()))
    real_ind = json.load(open(os.path.join(C3, f"ind_real_{SITE}.json"), encoding="utf-8"))
    hard = load_hard_mask(SITE)
    inten = token_intensity()
    pools = load_pools_geo()
    idx = build_cell_index(pools)

    mixes = semantic_mixes(priors)
    res_cfg = {}
    for cfg in CONFIGS:
        mix = mixes[cfg]
        f = family_demand(mix, bridge_gfa)
        n_raw, resid, scale, n_int, M, g = token_cell_demand(f, G, nB)
        grid = rank_match_v04(site, n_int, cgfa, inten)
        rng = np.random.default_rng(SEED)        # FRESH rng per config
        gdf, klog = collage_keepout(grid, pools, idx, rng, all_train, hard)
        gdf.to_parquet(os.path.join(CACHE, f"cf_{cfg}.parquet"))
        make_tile(gdf, os.path.join(CACHE, f"tile_cf_{cfg}.png"), simplify=1.5)
        ind = indicators(gdf)
        t68 = float((n_int[5] + n_int[7]) / nB)  # T6 + T8 cell share
        pub = float(gdf.loc[gdf["semantic"].isin(PUBLIC_SEMS), "gfa"].sum()
                    / gdf["gfa"].sum())
        unl = float(gdf.loc[gdf["semantic"] == "unlabeled", "gfa"].sum()
                    / gdf["gfa"].sum())
        res_cfg[cfg] = {
            "config": CONFIGS[cfg],
            "semantic_mix": {s: round(float(v), 4) for s, v in zip(SEMS, mix)},
            "family_gfa_demand": {fam: round(float(v), 4) for fam, v in zip(FAMILIES, f)},
            "nnls_raw_cells": [round(float(v), 2) for v in n_raw],
            "nnls_residual": round(resid, 1),
            "rescale_factor_to_built_cells": round(float(scale), 4),
            "token_cells_T1_T8": n_int.tolist(),
            "keepout_log": klog,
            "n_buildings": int(len(gdf)),
            "indicators": {k: round(v, 4) for k, v in ind.items()},
            "t68_cell_share": round(t68, 4),
            "public_gfa_share_rendered": round(pub, 4),
            "unlabeled_gfa_share_rendered": round(unl, 4),
        }
        print(cfg, "mix:", np.round(mix, 3), "\n   fam:", np.round(f, 3),
              "\n   n:", n_int.tolist(), "FAR", round(ind["FAR"], 2),
              "drop", klog["n_dropped"], f"({klog['dropped_gfa_share']:.1%})")

    # ------------------------------------------------ direction checks
    def argmax_cfg(key_fn):
        vals = {c: key_fn(res_cfg[c]) for c in CONFIGS}
        best = max(vals.values())
        winners = [c for c, v in vals.items() if abs(v - best) < 1e-9]
        return winners, vals

    checks = {}
    for name, expect, fn in [
        ("DEV_highest_tower_share", "DEV", lambda r: r["indicators"]["share_tower"]),
        ("DEV_highest_T6T8_cell_share", "DEV", lambda r: r["t68_cell_share"]),
        ("DEV_highest_h_p90", "DEV", lambda r: r["indicators"]["height_p90"]),
        ("RES_highest_house_share", "RES", lambda r: r["indicators"]["share_house"]),
        ("GOV_highest_large_footprint_share", "GOV",
         lambda r: r["indicators"]["share_large_footprint"]),
        ("GOV_highest_public_gfa_share_rendered", "GOV",
         lambda r: r["public_gfa_share_rendered"]),
    ]:
        winners, vals = argmax_cfg(fn)
        tie = len(winners) > 1
        checks[name] = {"expected": expect,
                        "winner": "=".join(winners) if tie else winners[0],
                        "held": bool(not tie and winners[0] == expect),
                        "tie": tie,
                        "values": {c: round(float(v), 4) for c, v in vals.items()}}
    n_held = sum(v["held"] for v in checks.values())
    n_tie = sum(v["tie"] for v in checks.values())

    payload = {
        "meta": {
            "script": "v05/exp4_counterfactual.py", "seed": SEED,
            "site": f"{SITE} Sanlin (heldout)", "n_built_cells": nB,
            "site_real_total_gfa_m2": round(G, 0), "site_real_FAR": round(G / AREA, 4),
            "chain": ("power config -> semantic mix (priors, no_age estate cell) -> "
                      "family GFA demand (E2 GFA-weighted bridge) -> token cell demand "
                      "(NNLS vs token family shares x mean cell GFA, rescaled to 293 "
                      "built cells) -> rank-match on A20 typed skeleton -> keep-out "
                      "collage, ALL-TRAIN pools, fresh default_rng(0) per config"),
            "disclaimer": ("ILLUSTRATIVE, NOT PREDICTIVE — literature priors, single "
                           "seed, one site. Semantics of pasted buildings come from "
                           "the pools (form-matched), not from the demanded mix; the "
                           "public-share check therefore tests whether the FORM chain "
                           "carries the semantic signal through."),
        },
        "semantic_mixes": {c: res_cfg[c]["semantic_mix"] for c in CONFIGS},
        "per_config": res_cfg,
        "direction_checks": checks,
        "n_checks_held": f"{n_held}/{len(checks)}",
        "n_checks_tied": n_tie,
        "real_site_indicators": {k: round(float(v), 4) for k, v in real_ind.items()},
        "notes": [
            ("RESCALE DISTORTION: forcing each config's NNLS cell counts to fill all "
             "293 built cells multiplies total demand by "
             + ", ".join(f"{c} x{res_cfg[c]['rescale_factor_to_built_cells']:.3f}"
                         for c in CONFIGS)
             + " — so total FAR is NOT comparable across configs as 'demand'; only "
               "composition and verticality are. GOV FAR > DEV FAR is partly this "
               "artifact (GOV demand under-filled the site and was inflated 22%)."),
            ("NNLS residual is exactly 0 for all configs: 5 family-GFA equations, 8 "
             "token unknowns — the system is underdetermined, NNLS picks a sparse "
             "solution (several tokens get 0 cells, e.g. T2 and T4 never used)."),
            ("Rendered semantics come from form-matched pool buildings, not from the "
             "demanded mix: GOV demands 47.8% public-ish classes but renders only "
             "7.5% public GFA — the form-token bottleneck strips semantic identity. "
             "This is a finding about the chain, not about Shanghai."),
            ("RES config receives 1 T8 mega-cell (115k m2 mean GFA) from NNLS trying "
             "to meet its 11% large_footprint GFA demand with whole cells — a "
             "granularity artifact of the k=8 vocabulary (T8 pool has only 21 cells)."),
        ],
    }
    json.dump(payload, open(RESULTS, "w"), ensure_ascii=False, indent=2)
    print("checks held:", payload["n_checks_held"])
    return payload


# ---------------------------------------------------------------- figure
def make_fig():
    res = json.load(open(RESULTS, encoding="utf-8"))
    cfgs = list(CONFIGS)
    sem_short = ["res.estate", "res.lowrise", "commercial", "office", "industrial",
                 "public", "education", "health"]
    fam_short = ["house", "slab", "tower", "large", "midrise"]

    fig = plt.figure(figsize=(13.2, 13.6))
    gs = fig.add_gridspec(5, 3, height_ratios=[1.55, 0.78, 0.55, 0.78, 0.52],
                          hspace=0.42, wspace=0.26,
                          left=0.07, right=0.97, top=0.90, bottom=0.045)
    for j, cfg in enumerate(cfgs):
        r = res["per_config"][cfg]
        col = CFG_COLORS[cfg]
        # row 0: axon tile
        ax = fig.add_subplot(gs[0, j])
        show_tile(ax, os.path.join(CACHE, f"tile_cf_{cfg}.png"))
        cw = CONFIGS[cfg]
        ax.set_title(f"{cfg}  (gov {cw['government']:.1f} dev {cw['developer_capital']:.1f} "
                     f"res {cw['resident_community']:.1f} inf {cw['informal']:.1f})\n"
                     f"FAR {r['indicators']['FAR']:.2f}   "
                     f"dropped {r['keepout_log']['n_dropped']} bld "
                     f"({r['keepout_log']['dropped_gfa_share']:.1%} GFA)",
                     fontsize=10, color=col, fontweight="bold")
        # row 1: semantic mix
        ax = fig.add_subplot(gs[1, j])
        v = [r["semantic_mix"][s] for s in SEMS]
        ax.barh(range(8)[::-1], v, color=col, alpha=0.85)
        ax.set_yticks(range(8)[::-1])
        ax.set_yticklabels(sem_short if j == 0 else [""] * 8, fontsize=7.5)
        ax.set_xlim(0, 0.42)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)
        ax.set_title("semantic mix (priors)", fontsize=8.5)
        # row 2: family GFA demand
        ax = fig.add_subplot(gs[2, j])
        v = [r["family_gfa_demand"][f] for f in FAMILIES]
        ax.bar(range(5), v, color=col, alpha=0.85)
        ax.set_xticks(range(5))
        ax.set_xticklabels(fam_short, fontsize=7.5)
        ax.set_ylim(0, 0.5)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("family GFA demand (bridge)", fontsize=8.5)
        # row 3: compact indicator table
        ax = fig.add_subplot(gs[3, j])
        ax.set_axis_off()
        i = r["indicators"]
        lines = [
            f"FAR            {i['FAR']:.2f}   (real site {res['real_site_indicators']['FAR']:.2f})",
            f"coverage       {i['coverage']:.2f}   (real {res['real_site_indicators']['coverage']:.2f})",
            f"h_p90          {i['height_p90']:.0f} m",
            f"tower share    {i['share_tower']:.3f}",
            f"house share    {i['share_house']:.3f}",
            f"large_fp share {i['share_large_footprint']:.3f}",
            f"T6+T8 cells    {r['token_cells_T1_T8'][5] + r['token_cells_T1_T8'][7]} / 293",
            f"public GFA     {r['public_gfa_share_rendered']:.3f} (rendered)",
            f"tokens T1..T8  {r['token_cells_T1_T8']}",
        ]
        ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=8, family="monospace")
        ax.set_title("indicators", fontsize=8.5)
    # row 4: direction-check verdicts (span all columns)
    ax = fig.add_subplot(gs[4, :])
    ax.set_axis_off()
    parts = []
    for name, c in res["direction_checks"].items():
        if c["held"]:
            mark = "HELD"
        elif c.get("tie"):
            mark = f"TIE ({c['winner']})"
        else:
            mark = f"FAILED (winner {c['winner']})"
        parts.append(f"{name}: {mark}")
    txt = ("Direction checks ({} held)   ".format(res["n_checks_held"])
           + " | ".join(parts[:3]) + "\n" + " | ".join(parts[3:]))
    ax.text(0.5, 0.72, txt, transform=ax.transAxes, ha="center", va="top", fontsize=8.6)
    ax.text(0.5, 0.12, "ILLUSTRATIVE, NOT PREDICTIVE — literature priors, single seed, "
            "one site. Same A20 skeleton (built cells, road/water keep-out) in all three.",
            transform=ax.transAxes, ha="center", fontsize=10, style="italic",
            color="#9d0208", fontweight="bold")
    fig.suptitle("v0.5 E4 — Counterfactual: one site (A20 Sanlin), one built-cell "
                 "skeleton, three power configurations\n"
                 "power weights -> semantic mix -> family GFA -> token cells -> "
                 "rank-matched keep-out collage (all-TRAIN pools, seed 0)",
                 fontsize=12.5, y=0.995)
    out = os.path.join(FIG, "fig_e4_counterfactual.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("fig saved:", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        make_fig()
    else:
        main()
        make_fig()
