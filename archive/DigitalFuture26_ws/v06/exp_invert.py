#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D6 — stakeholder-first inversion: pick/blend the doctrine for a target
stakeholder profile (mirrors v05 E4 targets).

CANDIDATES (15): 5 pure paradigms (D2 token budgets) + all 10 pairwise 50/50
blends (budget = mean of the two budgets, largest-remainder to 400 cells;
skeleton = first-named paradigm's skeleton; per-token semantic mix = mean of
the parents' per-token demanded-semantic cell distributions from the D3 units).

SEARCH on EXPECTATION (deterministic, no generation): expected read-back per
candidate = sum over built tokens T1..T8 of
    [token count x E(labeled GFA per pool cell of that token)
                 x E(power vector of that token's pool)]
normalized by total expected labeled GFA, where E(power vector) = GFA-weighted
mean over the token pool's LABELED buildings' v05 prior vectors, estates on the
no_age cell (the D4 chain's documented fallback). This pool-average expectation
deliberately ignores semantic conditioning of the collage — the gap it leaves
shows up in expectation-vs-realized.

Rank by cosine similarity AND L1 distance; if winners disagree, disclose and
pick by L1. GENERATE the 3 winners (seed 0, D3 machinery incl. semantic-
conditioned collage; pure winners reuse the cached D3 unit verbatim), compute
REALIZED read-back (D4 R0 chain: donor semantics, estates->no_age) and the
expectation-vs-realized gap (SINGLE SEED — disclosed).

HONEST FRAMING: the search optimizes ASSERTED profiles built from literature
priors (TF/v05/power_priors_v05.json). It selects doctrines under our priors'
worldview, nothing more; it does not predict who would actually hold power.

Outputs: V06/results_d6.json, V06/fig/fig_d6_invert.png. Seed 0.
"""
import json
import math
import os
import sys
import textwrap

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation"
V05, V06 = f"{ROOT}/v05", f"{ROOT}/v06"
CACHE = f"{V06}/cache_gen"
sys.path.insert(0, V06)
import generate_v06 as gv   # D3 machinery imported, not duplicated

SEED = 0
DIMS = ["government", "developer_capital", "resident_community", "informal"]
DIM_SHORT = ["gov", "dev", "res", "inf"]
PARADIGMS = ["garden_city", "ville_radieuse", "neighbourhood_unit",
             "microrayon", "gb50180"]
SHORT = {"garden_city": "GC", "ville_radieuse": "VR",
         "neighbourhood_unit": "NU", "microrayon": "MR", "gb50180": "GB"}

# v05 E4-mirroring stakeholder targets (raw [0,1] non-exclusive weights)
TARGETS = {
    "GOV-first": {"government": .8, "developer_capital": .2,
                  "resident_community": .3, "informal": .0},
    "DEV-first": {"government": .3, "developer_capital": .8,
                  "resident_community": .2, "informal": .0},
    "RES-first": {"government": .2, "developer_capital": .2,
                  "resident_community": .8, "informal": .4},
}

# ---------------------------------------------------------------- priors
# prior_cell copied verbatim from V06/exp_readback.py (importing that module
# would execute the whole D4 script; copying with provenance is the sanctioned
# alternative).
P = json.load(open(f"{V05}/power_priors_v05.json"))
PRIORS, BASELINE = P["priors"], P["baseline_unspecified"]
LOW_CONF = {"low", "medium-low"}


def prior_cell(sem, estate_cell="no_age"):
    # verbatim from V06/exp_readback.py::prior_cell
    if sem == "residential_estate":
        c = PRIORS["residential_estate"]["cells"][estate_cell]
        return c["w"], c["confidence"]
    if sem in PRIORS:
        return PRIORS[sem]["w"], PRIORS[sem]["confidence"]
    return BASELINE, "low"


def readback(df, sem_col="semantic", estate_cell="no_age"):
    # verbatim from V06/exp_readback.py::readback (D4 chain)
    g = df[df[sem_col] != "unlabeled"].copy()
    W = np.array([prior_cell(s, estate_cell)[0][d]
                  for s in g[sem_col] for d in DIMS]).reshape(len(g), 4)
    wts = g["gfa"].to_numpy()
    prof = {d: float(np.average(W[:, i], weights=wts))
            for i, d in enumerate(DIMS)}
    conf = np.array([prior_cell(s, estate_cell)[1] in LOW_CONF
                     for s in g[sem_col]])
    low_share = float(wts[conf].sum() / wts.sum())
    unl_share = float(1 - wts.sum() / df["gfa"].sum())
    return prof, low_share, unl_share, int(len(g))


# ---------------------------------------------------------------- helpers
def largest_remainder(weights, total):
    """weights: {key: float >= 0} -> {key: int} summing to total (Hamilton)."""
    items = sorted(weights.items())
    s = sum(v for _, v in items)
    raw = [(k, v * total / s) for k, v in items]
    out = {k: int(math.floor(x)) for k, x in raw}
    rem = total - sum(out.values())
    order = sorted(((x - math.floor(x), k) for k, x in raw), reverse=True)
    for i in range(rem):
        out[order[i][1]] += 1
    return {k: v for k, v in out.items() if v > 0}


def vec(d):
    return np.array([d[k] for k in DIMS], float)


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def l1(a, b):
    return float(np.abs(a - b).sum())


# ---------------------------------------------------------------- pool expectation
print("loading pools ...")
pools = gv.load_pools_geo()
idx = gv.build_cell_index(pools)
by_ts, _avail = gv.build_semantic_index(pools)
A12 = gv.load_adjacency12()
pools["gfa_b"] = pools["fp_area"] * pools["floors"]

TOK_EXP = {}   # token -> {n_cells, E_gfa_labeled_per_cell, w (4,), unl_gfa_share}
for t, grp in pools.groupby("token"):
    t = int(t)
    if not 1 <= t <= 8:
        continue
    n_cells = grp["key"].nunique()
    lab = grp[grp["semantic"] != "unlabeled"]
    W = np.array([vec(prior_cell(s)[0]) for s in lab["semantic"]])
    wts = lab["gfa_b"].to_numpy()
    TOK_EXP[t] = {
        "n_pool_cells": int(n_cells),
        "E_gfa_labeled_per_cell": float(wts.sum() / n_cells),
        "w": np.average(W, axis=0, weights=wts),
        "unlabeled_gfa_share": float(1 - wts.sum() / grp["gfa_b"].sum()),
    }


def expected_profile(budget):
    """Deterministic expectation of the D4 R0 read-back for a token budget:
    GFA-weighted mean of pool-level expected power vectors (no generation)."""
    num = np.zeros(4)
    den = 0.0
    for t in range(1, 9):
        n = budget.get(f"T{t}", 0)
        if n == 0:
            continue
        e = TOK_EXP[t]
        num += n * e["E_gfa_labeled_per_cell"] * e["w"]
        den += n * e["E_gfa_labeled_per_cell"]
    return num / den


# ---------------------------------------------------------------- candidates
D2 = json.load(open(f"{V06}/results_d2.json"))["paradigms"]
BUDGET = {p: D2[p]["token_budget"] for p in PARADIGMS}


def parent_token_sem(p):
    """Per-token demanded-semantic CELL distribution of the D3 unit of p."""
    df = gpd.read_parquet(f"{CACHE}/unit_{p}.parquet")
    cells = df.drop_duplicates(subset=["tgt_row", "tgt_col"])
    out = {}
    for t, grp in cells.groupby("token"):
        out[int(t)] = grp["demanded_semantic"].value_counts(normalize=True).to_dict()
    return out


PARENT_SEM = {p: parent_token_sem(p) for p in PARADIGMS}

CANDS = {}   # name -> dict(budget, parents, skeleton_of)
for p in PARADIGMS:
    CANDS[p] = {"budget": dict(BUDGET[p]), "parents": [p], "skeleton_of": p}
for i, a in enumerate(PARADIGMS):
    for b in PARADIGMS[i + 1:]:
        keys = set(BUDGET[a]) | set(BUDGET[b])
        avg = {k: (BUDGET[a].get(k, 0) + BUDGET[b].get(k, 0)) / 2 for k in keys}
        CANDS[f"{a}+{b}"] = {"budget": largest_remainder(avg, 400),
                             "parents": [a, b], "skeleton_of": a}

# ---------------------------------------------------------------- search
EXPECT = {name: expected_profile(c["budget"]) for name, c in CANDS.items()}

ranking = {}
winners = {}
for tname, tgt in TARGETS.items():
    tv = vec(tgt)
    rows = []
    for name in CANDS:
        ev = EXPECT[name]
        rows.append({"candidate": name,
                     "expected_profile": {d: round(float(v), 4)
                                          for d, v in zip(DIMS, ev)},
                     "L1_to_target": round(l1(ev, tv), 4),
                     "cosine_to_target": round(cosine(ev, tv), 4)})
    by_l1 = sorted(rows, key=lambda r: r["L1_to_target"])
    by_cos = sorted(rows, key=lambda r: -r["cosine_to_target"])
    agree = by_l1[0]["candidate"] == by_cos[0]["candidate"]
    winners[tname] = by_l1[0]["candidate"]
    for r in rows:
        r["rank_L1"] = 1 + by_l1.index(r)
        r["rank_cosine"] = 1 + by_cos.index(r)
    ranking[tname] = {
        "target": tgt,
        "winner_L1": by_l1[0]["candidate"],
        "winner_cosine": by_cos[0]["candidate"],
        "metrics_agree_on_winner": agree,
        "picked": by_l1[0]["candidate"],
        "pick_rule": "L1 (cosine reported; on disagreement L1 decides, as specified)",
        "table": sorted(rows, key=lambda r: r["rank_L1"]),
    }
    print(f"{tname}: L1 winner {by_l1[0]['candidate']} "
          f"(L1 {by_l1[0]['L1_to_target']:.3f}) | cosine winner "
          f"{by_cos[0]['candidate']} (cos {by_cos[0]['cosine_to_target']:.4f}) "
          f"| agree {agree}")

# ---------------------------------------------------------------- blend generation
def gen_blend(name, rng):
    """Generate a 50/50 blend unit: first parent's skeleton (roads/green/civic
    kept; civic RESERVED cells keep the first parent's T2 civic program), the
    blended BUILT-token multiset reconciled to the skeleton's build cells by
    largest remainder (blend has more/fewer built cells than the skeleton
    offers — counted), micro-arrangement by Shanghai-accent swap-Gibbs (a blend
    has NO prescriptive doctrine geometry of its own — disclosed), per-token
    demanded semantics = largest-remainder draw from the mean of the parents'
    per-token semantic distributions, then D3 semantic-conditioned collage."""
    a, b = CANDS[name]["parents"]
    budget = CANDS[name]["budget"]
    skel = np.load(f"{V06}/skeleton_{a}.npy")
    g = skel.copy()
    sem = np.full(skel.shape, None, dtype=object)
    build = sorted((int(r), int(c)) for r, c in zip(*np.where(skel == 0)))
    reserved = sorted((int(r), int(c)) for r, c in zip(*np.where(skel == -1)))

    # civic RESERVED cells: T2 carrier with parent a's civic semantics (verbatim
    # cell-wise lookup from the cached D3 unit of a)
    pa = gpd.read_parquet(f"{CACHE}/unit_{a}.parquet").drop_duplicates(
        subset=["tgt_row", "tgt_col"]).set_index(["tgt_row", "tgt_col"])
    for r, c in reserved:
        g[r, c] = 2
        sem[r, c] = pa.loc[(r, c), "demanded_semantic"]

    # blended built multiset minus the T2 cells consumed by the civic reserve
    built = {k: v for k, v in budget.items() if k.startswith("T")}
    t2_used = min(built.get("T2", 0), len(reserved))
    free_ms = dict(built)
    if "T2" in free_ms:
        free_ms["T2"] -= t2_used
        if free_ms["T2"] == 0:
            del free_ms["T2"]
    civic_overdraw = len(reserved) - t2_used   # reserved cells beyond budget T2
    n_free_built = sum(free_ms.values())
    recon = {"blend_built_budget": built, "reserved_civic_cells": len(reserved),
             "civic_T2_overdraw_beyond_budget": civic_overdraw,
             "skeleton_build_cells": len(build)}
    if n_free_built > len(build):
        free_ms = largest_remainder(free_ms, len(build))
        recon["reconciliation"] = (f"blend demands {n_free_built} non-civic built "
                                   f"cells but skeleton of {a} offers {len(build)}; "
                                   "scaled down by largest remainder")
    else:
        recon["reconciliation"] = (f"{len(build) - n_free_built} skeleton build "
                                   "cells left E_VACANT (blend under-fills)")
    seq = [int(k[1:]) for k, v in sorted(free_ms.items()) for _ in range(v)]
    seq += [0] * (len(build) - len(seq))
    seq = np.array(seq)
    rng.shuffle(seq)
    for (r, c), t in zip(build, seq):
        g[r, c] = int(t)
    g = gv.swap_gibbs(g, build, A12, rng, sweeps=3)   # Shanghai accent

    # per-token semantics: mean of parents' distributions, largest remainder
    for t in sorted(set(int(x) for x in seq if x > 0)):
        dists = [PARENT_SEM[p][t] for p in (a, b) if t in PARENT_SEM[p]]
        sems = sorted({s for d in dists for s in d})
        share = {s: sum(d.get(s, 0.0) for d in dists) / len(dists) for s in sems}
        cells_t = sorted(rc for rc in build if g[rc[0], rc[1]] == t)
        alloc = largest_remainder(share, len(cells_t))
        lab = [s for s in sorted(alloc) for _ in range(alloc[s])]
        for (r, c), s in zip(cells_t, lab):
            sem[r, c] = s
    gdf, slog = gv.collage_semantic(g, sem, pools, idx, by_ts, rng, 1000.0)
    ach = {f"T{t}": int((g == t).sum()) for t in range(1, 9) if (g == t).sum()}
    ach.update({"E_VACANT": int((g == 0).sum()), "E_ROAD": int((g == 9).sum()),
                "E_GREEN": int((g == 11).sum())})
    recon["achieved_tokens"] = ach
    return g, gdf, slog, recon


# ---------------------------------------------------------------- winners: generate + realized read-back
results = {"_meta": {
    "script": "v06/exp_invert.py", "seed": SEED,
    "task": "D6 stakeholder-first inversion (mirrors v05 E4 targets)",
    "search": "deterministic expectation over token pools: sum_t N_t * E[labeled GFA/cell] * "
              "E[pool power vector] / sum_t N_t * E[labeled GFA/cell]; estates on no_age cell; "
              "expectation ignores semantic conditioning (pool average) — by design, the "
              "expectation-vs-realized gap then measures what conditioning + collage sampling move",
    "honest_framing": "the search optimizes ASSERTED profiles assembled from literature priors "
                      "(TF/v05/power_priors_v05.json). It selects doctrines under our priors' "
                      "worldview, nothing more — it does NOT predict realized power relations. "
                      "Realized read-back is a single seed-0 draw (no variance estimate).",
    "candidates": "5 pure paradigms + 10 pairwise 50/50 budget blends (largest remainder to 400 "
                  "cells; skeleton = first-named parent; per-token semantics = mean of parents' "
                  "per-token demanded-semantic distributions from the D3 units)",
    "blend_generation": "no doctrine prescribes a blend's micro-geometry: tokens are placed "
                        "randomly then arranged by 3 Shanghai-accent swap-Gibbs sweeps "
                        "(v0.4 12-token TRAIN adjacency); first parent's civic reserve kept",
}, "token_pool_expectation": {
    f"T{t}": {"n_pool_cells": e["n_pool_cells"],
              "E_gfa_labeled_per_cell_m2": round(e["E_gfa_labeled_per_cell"], 1),
              "expected_power": {d: round(float(v), 4) for d, v in zip(DIMS, e["w"])},
              "unlabeled_gfa_share": round(e["unlabeled_gfa_share"], 4)}
    for t, e in sorted(TOK_EXP.items())},
    "targets": TARGETS, "ranking": ranking, "winners": {}}

rng = np.random.default_rng(SEED)
win_assets = {}
for tname in TARGETS:
    wname = winners[tname]
    print("generating winner for", tname, "->", wname)
    if wname in PARADIGMS:
        g = np.load(f"{CACHE}/grid_unit_{wname}.npy")
        gdf = gpd.read_parquet(f"{CACHE}/unit_{wname}.parquet")
        slog = json.load(open(f"{V06}/results_d3.json"))["paradigms"][wname]["unit"]["semantic_collage"]
        recon = {"note": "pure paradigm: cached D3 unit reused verbatim (seed 0)"}
        tile = f"{CACHE}/tile_unit_{wname}.png"
    else:
        rng_w = np.random.default_rng(SEED)   # fresh seed-0 stream per winner
        g, gdf, slog, recon = gen_blend(wname, rng_w)
        stem = wname.replace("+", "__")
        gdf.to_parquet(f"{CACHE}/unit_blend_{stem}.parquet")
        np.save(f"{CACHE}/grid_unit_blend_{stem}.npy", g)
        tile = f"{CACHE}/tile_unit_blend_{stem}.png"
        gv.make_tile(gdf, tile, 1000.0)
    prof, low_share, unl_share, n_lab = readback(gdf, "semantic", "no_age")
    ev, tv = EXPECT[wname], vec(TARGETS[tname])
    rv = vec(prof)
    ind = gv.indicators(gdf, 1e6)
    results["winners"][tname] = {
        "winner": wname,
        "expected_profile": {d: round(float(v), 4) for d, v in zip(DIMS, ev)},
        "realized_readback_R0": {d: round(prof[d], 4) for d in DIMS},
        "target": TARGETS[tname],
        "L1_expected_vs_target": round(l1(ev, tv), 4),
        "L1_realized_vs_target": round(l1(rv, tv), 4),
        "L1_expected_vs_realized_gap": round(l1(ev, rv), 4),
        "single_seed_disclosure": "realized = one seed-0 generation; no variance estimate",
        "low_confidence_gfa_share": round(low_share, 4),
        "unlabeled_gfa_share": round(unl_share, 4),
        "n_buildings_labeled": n_lab,
        "indicators": {k: round(v, 4) for k, v in ind.items()},
        "semantic_collage_fallback_rate": slog.get("fallback_rate"),
        "generation": recon,
    }
    win_assets[tname] = (wname, g, gdf, tile, ev, rv, tv)
    print(f"  expected L1->target {l1(ev, tv):.3f} | realized L1->target "
          f"{l1(rv, tv):.3f} | exp-vs-realized {l1(ev, rv):.3f}")

# ---------------------------------------------------------------- post-hoc check on metric disagreement
# Where L1 and cosine disagree on the winner, generate/read back the cosine
# winner too and report which pick REALIZES closer to the target (single seed).
for tname in TARGETS:
    rk = ranking[tname]
    if rk["metrics_agree_on_winner"]:
        continue
    alt = rk["winner_cosine"]
    if alt in PARADIGMS:
        gdf_alt = gpd.read_parquet(f"{CACHE}/unit_{alt}.parquet")
    else:
        g_a, gdf_alt, _, _ = gen_blend(alt, np.random.default_rng(SEED))
    prof_alt, _, _, _ = readback(gdf_alt, "semantic", "no_age")
    tv = vec(TARGETS[tname])
    l1_alt = l1(vec(prof_alt), tv)
    l1_win = results["winners"][tname]["L1_realized_vs_target"]
    rk["post_hoc_realized_check"] = {
        "cosine_winner_realized_readback": {d: round(prof_alt[d], 4) for d in DIMS},
        "L1_realized_vs_target_cosine_winner": round(l1_alt, 4),
        "L1_realized_vs_target_L1_winner": l1_win,
        "realized_favors": rk["picked"] if l1_win <= l1_alt else alt,
        "note": "single seed-0 realization each; margin between expected L1s was "
                f"{abs(rk['table'][0]['L1_to_target'] - rk['table'][1]['L1_to_target']):.4f} "
                "(within surrogate error — disclosed)",
    }
    print(f"post-hoc {tname}: L1 pick {rk['picked']} realizes {l1_win:.3f}, "
          f"cosine pick {alt} realizes {l1_alt:.3f} -> realized favors "
          f"{rk['post_hoc_realized_check']['realized_favors']}")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(16.2, 11.8))
gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.85, 0.72],
                      hspace=0.26, wspace=0.18,
                      left=0.05, right=0.98, top=0.90, bottom=0.135)
BAR_C = {"target": "#4d4d4d", "expected": "#80b1d3", "realized": "#fb8072"}
for j, tname in enumerate(TARGETS):
    wname, g, gdf, tile, ev, rv, tv = win_assets[tname]
    pretty = " + ".join(SHORT[p] for p in CANDS[wname]["parents"])
    kind = "pure" if wname in PARADIGMS else "50/50 blend"
    ax = fig.add_subplot(gs[0, j])
    gv.plot_plan(ax, g, gdf, 1000.0)
    tgt_s = ", ".join(f"{k} {tv[i]:.1f}" for i, k in enumerate(DIM_SHORT))
    ax.set_title(f"TARGET {tname}  ({tgt_s})\nwinner: {wname}\n[{kind}: {pretty}]",
                 fontsize=10.5)
    ax2 = fig.add_subplot(gs[1, j])
    ax2.imshow(plt.imread(tile))
    ax2.set_axis_off()
    ax2.set_title("seed-0 generation (height-coloured axon)", fontsize=10.5)
    ax3 = fig.add_subplot(gs[2, j])
    x = np.arange(4)
    ax3.bar(x - 0.27, tv, 0.27, color=BAR_C["target"], label="target")
    ax3.bar(x, ev, 0.27, color=BAR_C["expected"], label="expected (search)")
    ax3.bar(x + 0.27, rv, 0.27, color=BAR_C["realized"], label="realized (read-back)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(DIM_SHORT, fontsize=10.5)
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis="y", alpha=0.3)
    w = results["winners"][tname]
    ax3.set_title(f"L1 to target: expected {w['L1_expected_vs_target']:.2f} / "
                  f"realized {w['L1_realized_vs_target']:.2f}", fontsize=10.5)
    if j == 0:
        ax3.set_ylabel("stakeholder weight [0,1]")
        ax3.legend(fontsize=9.5, loc="upper right")
fig.suptitle("D6 — Stakeholder-first inversion: which doctrine (or 50/50 blend) best serves a "
             "target stakeholder?\nSearch over 15 candidates on deterministic pool-expectation; "
             "winner generated at seed 0 and read back through v05 power priors", fontsize=13)
# honesty stamp as a bold footer line (kept OUT of the drawing area)
fig.text(0.5, 0.068, "ILLUSTRATIVE, NOT PREDICTIVE", ha="center", va="bottom",
         fontsize=14, fontweight="bold", color="#666666")
fig.text(0.5, 0.006, textwrap.fill(
         "The search optimizes ASSERTED profiles assembled from literature priors "
         "(TF/v05/power_priors_v05.json): it selects doctrines under our priors' worldview, "
         "nothing more — it does not predict who would hold power in a built fabric. "
         "Estates read on the no-age prior cell (v05-documented low-confidence fallback); "
         "realized profile = one seed-0 generation (no variance estimate); blends carry no "
         "prescriptive geometry — arrangement is Shanghai-accent Gibbs only.", 158),
         ha="center", va="bottom", fontsize=10.5, style="italic")
fig.savefig(f"{V06}/fig/fig_d6_invert.png", dpi=140)
plt.close(fig)

with open(f"{V06}/results_d6.json", "w") as f:
    json.dump(results, f, indent=1)
print("written: results_d6.json, fig/fig_d6_invert.png")
