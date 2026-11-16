#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_learn_v02.py — v0.2 Learn: POI-primary semantic typing + grid token grammar.

Owner redirection vs probe: the 5-source fusion cascade was too complex to teach.
v0.2 uses ONE PRIMARY JUDGE + gap-fillers:
    POI is the single primary source (human-contributed open data, NOT AI-generated).
    Teaching narrative: "POI 說了算，POI 沉默時才問別人".

Usage:
    python3 run_learn_v02.py extract <lujiazui|laoximen|caoyang>  # POI 30 m join (cached)
    python3 run_learn_v02.py step1                                # fuse + coverage + agreement + FIG1
    python3 run_learn_v02.py step2                                # 50 m grid tokens + adjacency + pools + FIG2

Outputs (all inside v02/):
    cache/<patch>_poi30.parquet      POI 30 m majority join (tie -> nearest)
    fused_v02.fgb / fused_v02.parquet   per-building: semantic, judge, family, height, fp_area, floors, age
    results_v02.json                 coverage, agreement, token summary
    tokens.json, adjacency.json, token_grid_<patch>.npy
    token_pools.parquet / token_pools.fgb   per-token instantiation pools (cell-local coords)
    fig1_learn_poi.png, fig2_grammar.png

Code patterns and morphological-family rules reused verbatim from ../probe/run_learn.py.
Reuses probe/cache/<patch>_{euluc,osm,year}.parquet (bid-keyed joins, identical patch load order).
"""
import json, os, sys, shutil, math
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

ROOT = "/sessions/epic-peaceful-feynman/mnt/Urban-Stakeholder-Workshop"
if not os.path.isdir(ROOT):  # host fallback
    ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
PROBE = os.path.join(TF, "probe")
V02 = os.path.join(TF, "v02")
CACHE = os.path.join(V02, "cache")
os.makedirs(CACHE, exist_ok=True)

DC = os.path.join(ROOT, "data_collection", "上海城市数据集")
SRC_POI = os.path.join(DC, "02-POI&AOI", "1-POI", "24", "shp格式", "上海市-POI.shp")

PATCHES = {  # (lat, lon) centers — same as probe
    "lujiazui": (31.2362, 121.4980),
    "laoximen": (31.2230, 121.4880),
    "caoyang":  (31.2475, 121.3855),
}
PAD = 0.008
CRS_M = "EPSG:32651"

SEMANTIC = ["residential_estate", "residential_lowrise", "commercial_retail",
            "office_finance", "industrial", "public_admin", "education", "healthcare"]
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]

SEM_COLORS = {
    "residential_estate": "#f4a261", "residential_lowrise": "#e9c46a",
    "commercial_retail": "#e76f51", "office_finance": "#6a4c93",
    "industrial": "#8d99ae", "public_admin": "#1d3557",
    "education": "#2a9d8f", "healthcare": "#d62828", "unlabeled": "#d9d9d9",
}
JUDGES = ["poi", "osm", "function", "euluc", "morph", "none"]
JUDGE_COLORS = {"poi": "#2a9d8f", "osm": "#e9c46a", "function": "#f4a261",
                "euluc": "#e76f51", "morph": "#8d99ae", "none": "#d9d9d9"}

# ---------------------------------------------------------------- mappings (verbatim from probe)
def poi_to_sem(big, mid):
    if big in ("购物服务", "餐饮服务", "生活服务", "住宿服务", "体育休闲服务"):
        return "commercial_retail"
    if big in ("公司企业", "金融保险服务"):
        return "office_finance"
    if big == "商务住宅":  # context split by midType
        if isinstance(mid, str) and "住宅" in mid:
            return "residential_estate"
        if isinstance(mid, str) and "楼宇" in mid:
            return "office_finance"
        if isinstance(mid, str) and "产业园" in mid:
            return "industrial"
        return None
    if big == "政府机构及社会团体":
        return "public_admin"
    if big == "科教文化服务":
        return "education"
    if big == "医疗保健服务":
        return "healthcare"
    return None

OSM_MAP = {
    "apartments": "residential_estate", "residential": "residential_estate", "dormitory": "residential_estate",
    "house": "residential_lowrise", "detached": "residential_lowrise",
    "semidetached_house": "residential_lowrise", "terrace": "residential_lowrise",
    "retail": "commercial_retail", "commercial": "commercial_retail",
    "supermarket": "commercial_retail", "hotel": "commercial_retail", "kiosk": "commercial_retail",
    "office": "office_finance", "skyscraper": "office_finance", "bank": "office_finance",
    "industrial": "industrial", "warehouse": "industrial", "storage_tank": "industrial",
    "school": "education", "kindergarten": "education", "college": "education", "university": "education",
    "hospital": "healthcare", "clinic": "healthcare",
    "government": "public_admin", "civic": "public_admin", "public": "public_admin", "townhall": "public_admin",
}
FUNC_MAP = {"Residence": "residential_generic", "Office": "office_finance",
            "Business": "commercial_retail", "Industry": "industrial",
            "Public service": "public_admin"}
EULUC_MAP = {
    "居住用地": "residential_generic", "商务办公用地": "office_finance",
    "行政办公用地": "public_admin", "教育科研用地": "education",
    "医疗卫生用地": "healthcare", "商业服务用地": "commercial_retail",
    "工业用地": "industrial",
}


def patch_bbox(name):
    lat, lon = PATCHES[name]
    return (lon - PAD, lat - PAD, lon + PAD, lat + PAD)


def load_patch(name):  # verbatim from probe — bid order matches probe caches & fused_buildings.fgb
    g = gpd.read_file(os.path.join(ROOT, "simple_workflows", "data", f"patch_{name}.fgb"))
    g = g.reset_index(drop=True)
    g["bid"] = [f"{name}_{i}" for i in range(len(g))]
    return g


# ---------------------------------------------------------------- STEP 1a: POI 30 m extraction
def extract(name):
    """ONE teachable rule: each building takes the majority power-relevant POI class
    within 30 m (tighter than probe's 50 m); ties -> class of the NEAREST POI.
    Power-relevant = bigType maps to one of the 8 canonical classes
    (bigType 地名地址信息 excluded; unmapped bigTypes do not vote)."""
    bbox = patch_bbox(name)
    bld = load_patch(name)
    p = gpd.read_file(SRC_POI, bbox=bbox, columns=["bigType", "midType"]).to_crs(CRS_M)
    p = p[p["bigType"] != "地名地址信息"].copy()
    p["poi_sem"] = [poi_to_sem(b, m) for b, m in zip(p["bigType"], p["midType"])]
    p = p.dropna(subset=["poi_sem"]).reset_index(drop=True)

    buf = bld[["bid", "geometry"]].copy()
    buf["geometry"] = bld.geometry.buffer(30)
    j = gpd.sjoin(buf, p[["poi_sem", "geometry"]], predicate="intersects", how="inner")
    # true distance footprint -> POI point (for tie-break by nearest)
    bg = bld.set_index("bid").geometry
    j["dist"] = gpd.GeoSeries(bg.loc[j["bid"]].values, crs=CRS_M).distance(
        gpd.GeoSeries(p.geometry.loc[j["index_right"]].values, crs=CRS_M))
    grp = (j.groupby(["bid", "poi_sem"])
             .agg(n=("dist", "size"), dmin=("dist", "min")).reset_index())
    grp = grp.sort_values(["n", "dmin"], ascending=[False, True])  # majority; tie -> nearest
    win = grp.drop_duplicates("bid").rename(columns={"poi_sem": "poi_sem30", "n": "poi_top_n"})
    tot = j.groupby("bid").size().rename("poi_n30")
    out = bld[["bid"]].merge(win[["bid", "poi_sem30", "poi_top_n"]], on="bid", how="left")
    out = out.merge(tot, on="bid", how="left")
    out["poi_n30"] = out["poi_n30"].fillna(0).astype(int)
    out.to_parquet(os.path.join(CACHE, f"{name}_poi30.parquet"))
    print(f"[extract:{name}] buildings={len(bld)} mapped_pois={len(p)} "
          f"with_poi30={int((out['poi_n30']>0).sum())} ({(out['poi_n30']>0).mean()*100:.1f}%)")


# ---------------------------------------------------------------- STEP 1b: fuse
def minimum_rotated_rect_aspect(geom):  # verbatim from probe
    try:
        mrr = geom.minimum_rotated_rectangle
        xs, ys = mrr.exterior.coords.xy
        d1 = math.hypot(xs[1] - xs[0], ys[1] - ys[0])
        d2 = math.hypot(xs[2] - xs[1], ys[2] - ys[1])
        lo, hi = sorted([d1, d2])
        return hi / lo if lo > 1e-6 else np.nan
    except Exception:
        return np.nan


def fuse_patch(name):
    bld = load_patch(name)
    for layer, path in [("poi30", os.path.join(CACHE, f"{name}_poi30.parquet")),
                        ("osm", os.path.join(PROBE, "cache", f"{name}_osm.parquet")),
                        ("year", os.path.join(PROBE, "cache", f"{name}_year.parquet")),
                        ("euluc", os.path.join(PROBE, "cache", f"{name}_euluc.parquet"))]:
        bld = bld.merge(pd.read_parquet(path), on="bid", how="left")
    bld["elong"] = [minimum_rotated_rect_aspect(g) for g in bld.geometry]
    bld["gfa"] = bld["fp_area"] * bld["floors"]
    bld["age"] = pd.to_numeric(bld["age"], errors="coerce")

    # morphological family (verbatim probe cascade & thresholds)
    fam = pd.Series(["midrise_other"] * len(bld), index=bld.index, dtype=object)
    cond_house = (bld["floors"] <= 3) & (bld["fp_area"] < 400)
    cond_slab = (bld["elong"] >= 2) & (bld["floors"] >= 4) & (bld["floors"] <= 12)
    cond_tower = (bld["height"] >= 35) & (bld["elong"] < 2)
    cond_large = bld["fp_area"] >= 2000
    fam[cond_large] = "large_footprint"
    fam[cond_tower & ~cond_large] = "tower"
    fam[cond_slab & ~cond_tower & ~cond_large] = "slab"
    fam[cond_house] = "house"
    bld["family"] = fam

    # gap-filler semantics (used only where POI is silent)
    bld["sem_osm"] = bld["osm_type"].map(OSM_MAP)
    name_admin = bld["osm_name"].astype(str).str.contains("政府|法院|派出所|海关|委员会", na=False)
    name_bank = bld["osm_name"].astype(str).str.contains("银行|证券|保险", na=False)
    bld.loc[bld["sem_osm"].isna() & name_admin, "sem_osm"] = "public_admin"
    bld.loc[bld["sem_osm"].isna() & name_bank, "sem_osm"] = "office_finance"
    bld["sem_func"] = bld["func"].map(FUNC_MAP)
    bld["sem_euluc"] = bld["euluc_class2"].map(EULUC_MAP)

    # ---- THE rule: POI is the single primary judge; fillers in strict priority after.
    sem = pd.Series([None] * len(bld), index=bld.index, dtype=object)
    judge = pd.Series(["none"] * len(bld), index=bld.index, dtype=object)
    m = bld["poi_sem30"].notna()
    sem[m] = bld.loc[m, "poi_sem30"]; judge[m] = "poi"
    for col, tag in [("sem_osm", "osm"), ("sem_func", "function"), ("sem_euluc", "euluc")]:
        m = sem.isna() & bld[col].notna()
        sem[m] = bld.loc[m, col]; judge[m] = tag
    # morphology fallback (conservative): tiny lowrise with no metadata -> residential_lowrise
    is_low = (bld["floors"] <= 3) & (bld["fp_area"] < 400)
    m = sem.isna() & is_low
    sem[m] = "residential_lowrise"; judge[m] = "morph"
    # residential split estate/lowrise by morphology (applies to all judges)
    m = sem.isin(["residential_generic", "residential_estate"])
    sem[m & is_low] = "residential_lowrise"
    sem[m & ~is_low] = "residential_estate"
    sem[sem.isna()] = "unlabeled"
    bld["semantic"] = sem
    bld["judge"] = judge
    bld["patch"] = name
    return bld


def step1():
    patches = {n: fuse_patch(n) for n in PATCHES}
    allb = pd.concat(patches.values(), ignore_index=True)
    allb_g = gpd.GeoDataFrame(allb, geometry="geometry", crs=CRS_M)

    # ---- coverage: % decided by POI alone vs each filler, per patch
    coverage = {}
    for n, g in patches.items():
        vc = g["judge"].value_counts()
        coverage[n] = {j: round(float(vc.get(j, 0)) / len(g) * 100, 1) for j in JUDGES}
        coverage[n]["n_buildings"] = int(len(g))
        coverage[n]["labeled_pct"] = round(float((g["semantic"] != "unlabeled").mean()) * 100, 1)

    # ---- agreement with old 5-source fusion (probe), matched on bid
    old = gpd.read_file(os.path.join(PROBE, "fused_buildings.fgb"),
                        columns=["bid", "semantic"]).rename(columns={"semantic": "sem_old"})
    cmp_ = allb.merge(old[["bid", "sem_old"]], on="bid", how="left")
    agreement = {}
    for n in PATCHES:
        c = cmp_[cmp_["patch"] == n]
        both = c[(c["judge"] == "poi") & (c["sem_old"].notna()) & (c["sem_old"] != "unlabeled")]
        alll = c[(c["semantic"] != "unlabeled") & (c["sem_old"].notna()) & (c["sem_old"] != "unlabeled")]
        agreement[n] = {
            "poi_judged_vs_old_n": int(len(both)),
            "poi_judged_vs_old_agree_pct": round(float((both["semantic"] == both["sem_old"]).mean()) * 100, 1) if len(both) else None,
            "all_labeled_vs_old_n": int(len(alll)),
            "all_labeled_vs_old_agree_pct": round(float((alll["semantic"] == alll["sem_old"]).mean()) * 100, 1) if len(alll) else None,
        }
    # top disagreement pairs (POI-judged subset, pooled) — honest numbers
    bo = cmp_[(cmp_["judge"] == "poi") & cmp_["sem_old"].notna() & (cmp_["sem_old"] != "unlabeled")]
    dis = bo[bo["semantic"] != bo["sem_old"]]
    top_dis = (dis.groupby(["semantic", "sem_old"]).size().sort_values(ascending=False)
                  .head(8).reset_index().values.tolist())

    # ---- persist
    keep = ["bid", "patch", "semantic", "judge", "family", "height", "fp_area",
            "floors", "elong", "gfa", "age", "geometry"]
    tmp = "/tmp/fused_v02.fgb"
    allb_g[keep].to_file(tmp, driver="FlatGeobuf")
    shutil.copy(tmp, os.path.join(V02, "fused_v02.fgb"))
    allb_g[keep].drop(columns="geometry").to_parquet(os.path.join(V02, "fused_v02.parquet"))

    res_path = os.path.join(V02, "results_v02.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results["step1"] = {
        "rule": "POI primary: majority power-relevant POI class within 30 m, tie -> nearest; "
                "gap-fillers strictly after POI: OSM type -> Function -> EULUC -> morphology(house->lowrise); "
                "residential split: floors<=3 & fp<400 -> lowrise",
        "judge_share_pct": coverage,
        "agreement_with_old_fusion": agreement,
        "top_disagreements_poi_vs_old": [{"v02_poi": a, "old_fusion": b, "n": int(c)} for a, b, c in top_dis],
        "semantic_counts": {n: g["semantic"].value_counts().astype(int).to_dict() for n, g in patches.items()},
    }
    json.dump(results, open(res_path, "w"), ensure_ascii=False, indent=2)

    fig1(patches, coverage, agreement)
    print(json.dumps({"coverage": coverage, "agreement": agreement}, indent=1))


def fig1(patches, coverage, agreement):
    fig = plt.figure(figsize=(15, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], hspace=0.18)
    # row 1: maps
    for k, (n, g) in enumerate(patches.items()):
        ax = fig.add_subplot(gs[0, k])
        for s, c in SEM_COLORS.items():
            sub = g[g["semantic"] == s]
            if len(sub):
                gpd.GeoDataFrame(sub, geometry="geometry", crs=CRS_M).plot(
                    ax=ax, color=c, linewidth=0.15, edgecolor="white")
        ax.set_title(f"{n}  (n={len(g)}, labeled {coverage[n]['labeled_pct']}%)", fontsize=11)
        ax.set_aspect("equal"); ax.axis("off")
    handles = [Patch(facecolor=c, label=s) for s, c in SEM_COLORS.items()]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.52), ncol=5,
               fontsize=8.5, frameon=False)
    # row 2 left+mid: who judged (stacked bars)
    ax = fig.add_subplot(gs[1, :2])
    names = list(patches.keys()); bottoms = np.zeros(len(names))
    for j in JUDGES:
        vals = np.array([coverage[n][j] for n in names])
        ax.bar(names, vals, bottom=bottoms, color=JUDGE_COLORS[j], label=j,
               width=0.55, edgecolor="white")
        for x, (b, v) in enumerate(zip(bottoms, vals)):
            if v > 4:
                ax.text(x, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if j in ("poi", "euluc") else "black")
        bottoms += vals
    ax.set_ylabel("% of buildings"); ax.set_ylim(0, 100)
    ax.set_title("Who judged? POI primary vs gap-fillers (strict priority)", fontsize=11)
    ax.legend(fontsize=8.5, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    # row 2 right: agreement with old fusion
    ax = fig.add_subplot(gs[1, 2])
    ag = [agreement[n]["poi_judged_vs_old_agree_pct"] or 0 for n in names]
    nn = [agreement[n]["poi_judged_vs_old_n"] for n in names]
    bars = ax.bar(names, ag, color="#6a4c93", width=0.5, edgecolor="white")
    for b, a, m in zip(bars, ag, nn):
        ax.text(b.get_x() + b.get_width() / 2, a + 2, f"{a:.0f}%\n(n={m})",
                ha="center", fontsize=9)
    ax.set_ylim(0, 100); ax.set_ylabel("% same label")
    ax.set_title("POI-judged vs old 5-source fusion\n(where both labels exist)", fontsize=10.5)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle('v0.2 STEP 1 — POI is the single primary judge ("POI decides; ask others only when POI is silent")',
                 fontsize=13)
    fig.savefig(os.path.join(V02, "fig1_learn_poi.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- STEP 2: grid token grammar
CELL = 50.0
NCELL = 20
FEATS = ["gfa", "coverage", "mean_h", "sh_house", "sh_slab", "sh_tower",
         "sh_large", "sh_mid", "dom_sem_share"]


def step2():
    from sklearn.cluster import KMeans
    fused = pd.read_parquet(os.path.join(V02, "fused_v02.parquet"))
    geom = gpd.read_file(os.path.join(V02, "fused_v02.fgb"), columns=["bid"])
    geom = geom.set_index("bid").geometry
    fused["bidx"] = np.arange(len(fused))
    fused["geometry"] = geom.loc[fused["bid"]].values
    fused = gpd.GeoDataFrame(fused, geometry="geometry", crs=CRS_M)

    # patch origins from the source patch files (exact 1 km squares)
    origins = {}
    for n in PATCHES:
        b = gpd.read_file(os.path.join(ROOT, "simple_workflows", "data", f"patch_{n}.fgb")).total_bounds
        origins[n] = (b[0], b[1])

    cent = fused.geometry.centroid
    fused["col"] = 0; fused["row"] = 0
    for n, (ox, oy) in origins.items():
        m = fused["patch"] == n
        fused.loc[m, "col"] = np.clip(((cent[m].x - ox) // CELL).astype(int), 0, NCELL - 1)
        fused.loc[m, "row"] = np.clip(((cent[m].y - oy) // CELL).astype(int), 0, NCELL - 1)

    # ---- cell features (building assigned to cell by centroid)
    rows = []
    for n in PATCHES:
        g = fused[fused["patch"] == n]
        grp = g.groupby(["row", "col"])
        for r in range(NCELL):
            for c in range(NCELL):
                rec = {"patch": n, "row": r, "col": c}
                try:
                    sub = grp.get_group((r, c))
                except KeyError:
                    sub = None
                if sub is None or len(sub) == 0:
                    rec.update({f: 0.0 for f in FEATS}); rec["n_bld"] = 0
                else:
                    fp = sub["fp_area"].sum()
                    rec["gfa"] = float(sub["gfa"].sum())
                    rec["coverage"] = float(min(fp / CELL**2, 1.0))
                    rec["mean_h"] = float(sub["height"].mean())
                    fam_fp = sub.groupby("family")["fp_area"].sum()
                    for f, key in zip(["sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"],
                                      ["house", "slab", "tower", "large_footprint", "midrise_other"]):
                        rec[f] = float(fam_fp.get(key, 0.0) / fp) if fp > 0 else 0.0
                    lab = sub[sub["semantic"] != "unlabeled"]
                    rec["dom_sem_share"] = float(lab["semantic"].value_counts().iloc[0] / len(lab)) if len(lab) else 0.0
                    rec["n_bld"] = int(len(sub))
                rows.append(rec)
    cells = pd.DataFrame(rows)

    # ---- tokens: 1 reserved EMPTY (coverage < 0.02) + KMeans k=6 on z-scored features
    empty = cells["coverage"] < 0.02
    X = cells.loc[~empty, FEATS].values
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Z = (X - mu) / sd
    km = KMeans(n_clusters=6, random_state=0, n_init=10).fit(Z)
    # order clusters by mean GFA ascending -> T1..T6 (T0 = EMPTY)
    raw_cent = pd.DataFrame(km.cluster_centers_ * sd + mu, columns=FEATS)
    order = np.argsort(raw_cent["gfa"].values)
    remap = {int(old): i + 1 for i, old in enumerate(order)}
    raw_cent = raw_cent.iloc[order].reset_index(drop=True)  # row i -> token T(i+1)
    cells["token"] = 0
    cells.loc[~empty, "token"] = [remap[l] for l in km.labels_]

    # measured-stat labels ONLY
    tok_stats, tok_labels = {}, {}
    fam_short = {"sh_house": "house", "sh_slab": "slab", "sh_tower": "tower",
                 "sh_large": "large", "sh_mid": "mid"}
    for t in range(7):
        sub = cells[cells["token"] == t]
        if t == 0:
            tok_labels[0] = f"T0 EMPTY: cov<.02 (n={len(sub)})"
            tok_stats[0] = {"n_cells": int(len(sub))}
            continue
        st = sub[FEATS].mean()
        shares = {k: st[k] for k in fam_short}
        top_f = max(shares, key=shares.get)
        tok_labels[t] = (f"T{t}: cov {st['coverage']:.2f} h̄ {st['mean_h']:.0f}m "
                         f"{fam_short[top_f]} {shares[top_f]:.2f}")
        tok_stats[t] = {**{k: round(float(st[k]), 3) for k in FEATS},
                        "n_cells": int(len(sub)),
                        "top_family": fam_short[top_f]}

    # ---- adjacency: 4-neighbour transition counts pooled, row-normalized
    grids = {}
    A = np.zeros((7, 7))
    for n in PATCHES:
        gpatch = np.zeros((NCELL, NCELL), dtype=int)
        sub = cells[cells["patch"] == n]
        gpatch[sub["row"].values, sub["col"].values] = sub["token"].values
        grids[n] = gpatch
        np.save(os.path.join(V02, f"token_grid_{n}.npy"), gpatch)
        for r in range(NCELL):
            for c in range(NCELL):
                for dr, dc in ((0, 1), (1, 0)):
                    rr, cc = r + dr, c + dc
                    if rr < NCELL and cc < NCELL:
                        a, b = gpatch[r, c], gpatch[rr, cc]
                        A[a, b] += 1; A[b, a] += 1
    A_norm = A / A.sum(axis=1, keepdims=True).clip(min=1)

    # ---- per-token instantiation pools (cell-local coords)
    fused["token"] = 0
    cell_tok = cells.set_index(["patch", "row", "col"])["token"]
    fused["token"] = cell_tok.loc[list(zip(fused["patch"], fused["row"], fused["col"]))].values
    pool = fused[fused["token"] > 0].copy()
    ox = pool["patch"].map({n: o[0] for n, o in origins.items()})
    oy = pool["patch"].map({n: o[1] for n, o in origins.items()})
    cx = pool.geometry.centroid.x; cy = pool.geometry.centroid.y
    pool["dx"] = (cx - ox - pool["col"] * CELL).round(2)   # centroid offset within its 50 m cell
    pool["dy"] = (cy - oy - pool["row"] * CELL).round(2)
    shift_x = -(ox + pool["col"] * CELL); shift_y = -(oy + pool["row"] * CELL)
    from shapely.affinity import translate
    pool["geometry"] = [translate(g, xoff=sx, yoff=sy)
                        for g, sx, sy in zip(pool.geometry, shift_x, shift_y)]
    cols = ["bidx", "bid", "patch", "token", "row", "col", "dx", "dy",
            "height", "floors", "fp_area", "family", "semantic"]
    pool[cols].to_parquet(os.path.join(V02, "token_pools.parquet"))
    tmp = "/tmp/token_pools.fgb"
    gpd.GeoDataFrame(pool[cols + ["geometry"]], geometry="geometry", crs=CRS_M).to_file(
        tmp, driver="FlatGeobuf")
    shutil.copy(tmp, os.path.join(V02, "token_pools.fgb"))

    # ---- persist tokens + adjacency
    json.dump({
        "features": FEATS,
        "feature_zscore_mu": [round(float(v), 4) for v in mu],
        "feature_zscore_sd": [round(float(v), 4) for v in sd],
        "empty_rule": "coverage < 0.02 -> T0 EMPTY (reserved, not clustered)",
        "kmeans": {"k": 6, "random_state": 0, "ordering": "by mean GFA ascending"},
        "labels": tok_labels, "stats": tok_stats,
        "raw_centroids": raw_cent.round(3).to_dict("records"),
        "pool_sizes": {f"T{t}": int((pool['token'] == t).sum()) for t in range(1, 7)},
    }, open(os.path.join(V02, "tokens.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"tokens": [f"T{t}" for t in range(7)],
               "counts_pooled_4neighbour": A.astype(int).tolist(),
               "row_normalized": np.round(A_norm, 4).tolist()},
              open(os.path.join(V02, "adjacency.json"), "w"), indent=2)

    res_path = os.path.join(V02, "results_v02.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    results["step2"] = {
        "grid": "50 m cells, 20x20 per patch, 1200 cells pooled",
        "token_labels": tok_labels,
        "token_cell_counts": {f"T{t}": int((cells['token'] == t).sum()) for t in range(7)},
        "pool_sizes_buildings": {f"T{t}": int((pool['token'] == t).sum()) for t in range(1, 7)},
    }
    json.dump(results, open(res_path, "w"), ensure_ascii=False, indent=2)

    fig2(grids, A_norm, tok_labels, tok_stats, raw_cent)
    print(json.dumps(results["step2"], ensure_ascii=False, indent=1))


TOKEN_COLORS = ["#f0f0f0", "#e9c46a", "#f4a261", "#e76f51", "#2a9d8f", "#457b9d", "#6a4c93"]


def fig2(grids, A_norm, tok_labels, tok_stats, raw_cent):
    fig = plt.figure(figsize=(16.5, 6.4))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.0, 1.15, 1.35], hspace=0.32, wspace=0.25)
    cmap = ListedColormap(TOKEN_COLORS)
    for k, (n, g) in enumerate(grids.items()):
        ax = fig.add_subplot(gs[k, 0])
        ax.imshow(g, cmap=cmap, vmin=0, vmax=6, origin="lower")
        ax.set_title(n, fontsize=10, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    # adjacency heatmap
    ax = fig.add_subplot(gs[:, 1])
    im = ax.imshow(A_norm, cmap="YlGnBu", vmin=0, vmax=A_norm.max())
    ax.set_xticks(range(7), [f"T{t}" for t in range(7)], fontsize=9)
    ax.set_yticks(range(7), [f"T{t}" for t in range(7)], fontsize=9)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{A_norm[i,j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if A_norm[i, j] > 0.5 * A_norm.max() else "black")
    ax.set_title("Token adjacency (4-neighbour, pooled,\nrow-normalized)", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.75)
    # per-token mean profiles
    ax = fig.add_subplot(gs[:, 2])
    x = np.arange(len(FEATS)); w = 0.13
    # normalize gfa & mean_h to [0,1] for display only
    disp = raw_cent.copy()
    disp["gfa"] = disp["gfa"] / disp["gfa"].max()
    disp["mean_h"] = disp["mean_h"] / disp["mean_h"].max()
    for t in range(1, 7):
        vals = disp.iloc[t - 1][FEATS].values.astype(float)
        ax.bar(x + (t - 3.5) * w, vals, w, color=TOKEN_COLORS[t], label=f"T{t}")
    ax.set_xticks(x, ["GFA\n(/max)", "cov", "h̄\n(/max)", "house", "slab", "tower",
                      "large", "mid", "dom-sem\nshare"], fontsize=8.5)
    ax.set_title("Per-token mean profile (measured stats)", fontsize=11)
    ax.legend(fontsize=8.5, ncol=3, frameon=False); ax.grid(axis="y", alpha=0.3)
    handles = [Patch(facecolor=TOKEN_COLORS[t], label=tok_labels[t]) for t in range(7)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.10),
               ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("v0.2 STEP 2 — 50 m grid token grammar (KMeans k=6 + reserved EMPTY; labels = measured stats only)",
                 fontsize=13)
    fig.savefig(os.path.join(V02, "fig2_grammar.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "extract":
        extract(sys.argv[2])
    elif cmd == "step1":
        step1()
    elif cmd == "step2":
        step2()
    elif cmd == "all":
        for n in PATCHES:
            extract(n)
        step1()
        step2()
