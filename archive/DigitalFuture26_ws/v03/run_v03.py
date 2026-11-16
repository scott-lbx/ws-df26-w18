#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v03.py — v0.3 Learn at scale: 24 one-km windows, multi-source fusion cascade.

OWNER DIRECTIVE (latest): semantic recognition uses the multi-source fusion
cascade from the original probe (EULUC -> AOI -> OSM-type/Function -> POI 50 m
majority -> morphology fallback) — NOT v0.2's POI-primary rule (that stays a
teaching variant). Fusion + family rules reused verbatim from ../probe/run_learn.py.

v0.3 adds: 24-window library, TRAIN(18)/HELDOUT(6) split, TRAIN-only grammar
(k=8 token vocabulary + EMPTY, 4-neighbour adjacency, instantiation pools),
per-window signatures and the continuous signature cloud (PCA fit TRAIN,
project all).

Usage (stepwise; each step is cached / idempotent):
    python3 run_v03.py select                 # 24 windows + 18/6 split -> split.json
    python3 run_v03.py extract A01 A02 ...    # per-window cut+join+fuse -> cache/<id>_fused.parquet
    python3 run_v03.py library                # library.parquet + library.fgb + coverage table
    python3 run_v03.py learn                  # 50m grid, k=8 vocab (TRAIN), adjacency, pools, grids x24
    python3 run_v03.py signatures             # signatures.json + PCA cloud (fit TRAIN, project all)
    python3 run_v03.py fig1
    python3 run_v03.py fig2

Seed 0 everywhere. Vectors as FlatGeobuf via /tmp (NEVER GPKG on this mount).
"""
import json, os, sys, shutil, math
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

ROOT = "/sessions/epic-peaceful-feynman/mnt/Urban-Stakeholder-Workshop"
if not os.path.isdir(ROOT):  # host fallback
    ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF = os.path.join(ROOT, "typology_foundation")
V03 = os.path.join(TF, "v03")
CACHE = os.path.join(V03, "cache")
os.makedirs(CACHE, exist_ok=True)

DC = os.path.join(ROOT, "data_collection", "上海城市数据集")
SRC = {
    "bld":   os.path.join(DC, "01-建筑轮廓", "②AI解析", "上海市_建筑-AI影像解译-带高度.shp"),
    "euluc": os.path.join(DC, "09-开源土地利用", "09-开源土地利用", "开源建设用地分类", "Data", "上海市-开源建设用地.shp"),
    "aoi":   os.path.join(DC, "02-POI&AOI", "2-AOI", "AOI-baiduapi", "SHP", "上海市_AOI.shp"),
    "osm":   os.path.join(DC, "01-建筑轮廓", "③其它", "上海市_建筑-带type.shp"),
    "year":  os.path.join(DC, "01-建筑轮廓", "③其它", "上海市_建筑-带年份-内测版.shp"),
    "poi":   os.path.join(DC, "02-POI&AOI", "1-POI", "24", "shp格式", "上海市-POI.shp"),
}
AREAS = os.path.join(ROOT, "data_collection", "atlas_500m", "areas.csv")
PAD = 0.0078          # deg, bbox half-width for source reads
HALF = 500.0          # m, half window
CRS_M = "EPSG:32651"
FORCED = ["A01", "A10", "A13"]   # Lujiazui / Laoximen / Caoyang, forced TRAIN

SEMANTIC = ["residential_estate", "residential_lowrise", "commercial_retail",
            "office_finance", "industrial", "public_admin", "education", "healthcare"]
FAMILIES = ["house", "slab", "tower", "large_footprint", "midrise_other"]

SEM_COLORS = {
    "residential_estate": "#f4a261", "residential_lowrise": "#e9c46a",
    "commercial_retail": "#e76f51", "office_finance": "#6a4c93",
    "industrial": "#8d99ae", "public_admin": "#1d3557",
    "education": "#2a9d8f", "healthcare": "#d62828", "unlabeled": "#d9d9d9",
}

# ---------------------------------------------------------------- mappings (verbatim from probe/run_learn.py)
EULUC_MAP = {
    "居住用地": "residential_generic", "商务办公用地": "office_finance",
    "行政办公用地": "public_admin", "教育科研用地": "education",
    "医疗卫生用地": "healthcare", "商业服务用地": "commercial_retail",
    "工业用地": "industrial",
}

def aoi_to_sem(t):  # AOI 'type' = "type1;type2;type3"
    if not isinstance(t, str):
        return None
    p = t.split(";") + ["", ""]
    t1, t2 = p[0], p[1]
    if t1 == "商务住宅":
        if "住宅" in t2 or "别墅" in t2:
            return "residential_estate"
        if "楼宇" in t2:
            return "office_finance"
        if "产业园" in t2:
            return "industrial"
        return None
    if t1 == "政府机构及社会团体":
        return "public_admin"
    if t1 == "科教文化服务" and ("学校" in t2 or "科研" in t2):
        return "education"
    if t1 == "医疗保健服务":
        return "healthcare"
    if t1 in ("购物服务", "餐饮服务", "住宿服务"):
        return "commercial_retail"
    if t1 == "金融保险服务":
        return "office_finance"
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

def poi_to_sem(big, mid):
    if big in ("购物服务", "餐饮服务", "生活服务", "住宿服务", "体育休闲服务"):
        return "commercial_retail"
    if big in ("公司企业", "金融保险服务"):
        return "office_finance"
    if big == "商务住宅":
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


def results_io(update=None):
    p = os.path.join(V03, "results_v03.json")
    res = json.load(open(p)) if os.path.exists(p) else {}
    if update:
        res.update(update)
        json.dump(res, open(p, "w"), ensure_ascii=False, indent=2)
    return res


def load_split():
    return json.load(open(os.path.join(V03, "split.json")))


# ================================================================ SELECT
def overlap_frac(a, b):
    """Overlap fraction of two 1 km axis-aligned windows given centers (m)."""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if dx >= 2 * HALF or dy >= 2 * HALF:
        return 0.0
    return (2 * HALF - dx) * (2 * HALF - dy) / (2 * HALF) ** 2


def select():
    df = pd.read_csv(AREAS)
    pts = gpd.GeoSeries(gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326").to_crs(CRS_M)
    df["cx"], df["cy"] = pts.x.round(1), pts.y.round(1)
    rng = np.random.RandomState(0)
    groups = list(dict.fromkeys(df["type"]))           # csv order
    counts = df["type"].value_counts()
    quota = {g: 2 for g in groups}                     # >=2 per group
    for g in sorted(groups, key=lambda g: (-counts[g], groups.index(g)))[:4]:
        quota[g] += 1                                  # 4 extras to largest groups -> 24
    picked = [df[df["id"] == f].iloc[0] for f in FORCED]

    def ok(row):
        return all(overlap_frac((row["cx"], row["cy"]), (p["cx"], p["cy"])) <= 0.30
                   for p in picked)

    for g in groups:
        cand = df[(df["type"] == g) & (~df["id"].isin([p["id"] for p in picked]))]
        need = quota[g] - sum(1 for p in picked if p["type"] == g)
        for i in rng.permutation(len(cand)):
            if need <= 0:
                break
            row = cand.iloc[i]
            if ok(row):
                picked.append(row)
                need -= 1
    while len(picked) < 24:                            # backfill on overlap rejections
        rest = df[~df["id"].isin([p["id"] for p in picked])]
        added = False
        for i in rng.permutation(len(rest)):
            if ok(rest.iloc[i]):
                picked.append(rest.iloc[i]); added = True
                break
        if not added:
            break
    sel = pd.DataFrame(picked).reset_index(drop=True)

    # ---- split 18 TRAIN / 6 HELDOUT, stratified by type group, forced -> TRAIN
    held = []
    three_g = [g for g in groups if quota[g] == 3]
    two_g = [g for g in groups if quota[g] == 2]
    chosen2 = [two_g[i] for i in rng.permutation(len(two_g))[:2]]
    for g in three_g + chosen2:
        cand = sel[(sel["type"] == g) & (~sel["id"].isin(FORCED)) & (~sel["id"].isin(held))]
        if len(cand):
            held.append(cand.iloc[rng.randint(len(cand))]["id"])
    while len(held) < 6:                               # safety fill, group-diverse
        cand = sel[(~sel["id"].isin(FORCED)) & (~sel["id"].isin(held))]
        cand = cand.sort_values("id")
        held.append(cand.iloc[rng.randint(len(cand))]["id"])
    sel["role"] = np.where(sel["id"].isin(held), "heldout", "train")

    windows = sel[["id", "name", "type", "lat", "lon", "cx", "cy", "role"]].rename(
        columns={"type": "type_group"}).to_dict("records")
    out = {
        "seed": 0,
        "note": "type_group = curated sampling-frame metadata ONLY (audit); never used in figure labels.",
        "rule": "stratified >=2/group; +1 to 4 largest groups; forced A01/A10/A13 -> TRAIN; "
                "candidates overlapping >30% with an already-picked window dropped; "
                "split 18 TRAIN / 6 HELDOUT stratified by group (1 heldout per 3-quota group + 2 random 2-quota groups)",
        "n_train": int((sel["role"] == "train").sum()),
        "n_heldout": int((sel["role"] == "heldout").sum()),
        "windows": windows,
    }
    json.dump(out, open(os.path.join(V03, "split.json"), "w"), ensure_ascii=False, indent=2)
    print(sel.groupby(["type", "role"]).size())
    print("picked:", ", ".join(f"{w['id']}({w['role'][0]})" for w in windows))


# ================================================================ EXTRACT (per window, cached)
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


def extract_one(w):
    wid = w["id"]
    out_path = os.path.join(CACHE, f"{wid}_fused.parquet")
    if os.path.exists(out_path):
        print(f"[extract:{wid}] cached, skip")
        return
    bbox = (w["lon"] - PAD, w["lat"] - PAD, w["lon"] + PAD, w["lat"] + PAD)
    square = shapely.box(w["cx"] - HALF, w["cy"] - HALF, w["cx"] + HALF, w["cy"] + HALF)

    # ---- cut 1 km window from AI building-height layer
    b = gpd.read_file(SRC["bld"], bbox=bbox).to_crs(CRS_M)
    b["geometry"] = b.geometry.buffer(0)
    b = gpd.clip(b, square)
    b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    b["fp_area"] = b.geometry.area
    b = b[b["fp_area"] > 5].reset_index(drop=True)
    b["height"] = pd.to_numeric(b["Height"], errors="coerce").fillna(2).clip(lower=2)
    b["floors"] = np.maximum(1, np.round(b["height"] / 3.5)).astype(int)
    b["bid"] = [f"{wid}_{i}" for i in range(len(b))]
    b = b[["bid", "height", "fp_area", "floors", "geometry"]]
    if len(b) == 0:
        gpd.GeoDataFrame(columns=["bid", "window", "semantic", "judge", "family", "height",
                                  "fp_area", "floors", "age", "elong", "gfa", "geometry"],
                         geometry="geometry", crs=CRS_M).to_parquet(out_path)
        print(f"[extract:{wid}] EMPTY window")
        return
    rep = b[["bid", "geometry"]].copy()
    rep["geometry"] = b.representative_point()

    # ---- source joins (probe pattern)
    e = gpd.read_file(SRC["euluc"], bbox=bbox).to_crs(CRS_M)
    j = gpd.sjoin(rep, e[["class2", "geometry"]], predicate="within", how="left")
    b["euluc_class2"] = j.drop_duplicates("bid").set_index("bid")["class2"].reindex(b["bid"]).values

    a = gpd.read_file(SRC["aoi"], bbox=bbox).to_crs(CRS_M)
    a["aoi_area"] = a.geometry.area
    j = gpd.sjoin(rep, a[["type", "aoi_area", "geometry"]], predicate="within", how="left")
    j = j.sort_values("aoi_area").drop_duplicates("bid").set_index("bid")
    b["aoi_type"] = j["type"].reindex(b["bid"]).values

    o = gpd.read_file(SRC["osm"], bbox=bbox).to_crs(CRS_M)
    j = gpd.sjoin(rep, o[["type", "name", "geometry"]], predicate="within", how="left")
    j = j.drop_duplicates("bid").set_index("bid")
    b["osm_type"] = j["type"].reindex(b["bid"]).values
    b["osm_name"] = j["name"].reindex(b["bid"]).values

    y = gpd.read_file(SRC["year"], bbox=bbox).to_crs(CRS_M)
    j = gpd.sjoin(rep, y[["Function", "Age", "geometry"]], predicate="within", how="left")
    j = j.drop_duplicates("bid").set_index("bid")
    b["func"] = j["Function"].reindex(b["bid"]).values
    b["age"] = pd.to_numeric(j["Age"].reindex(b["bid"]), errors="coerce").values

    p = gpd.read_file(SRC["poi"], bbox=bbox, columns=["bigType", "midType"]).to_crs(CRS_M)
    p = p[p["bigType"] != "地名地址信息"].copy()
    p["poi_sem"] = [poi_to_sem(bt, mt) for bt, mt in zip(p["bigType"], p["midType"])]
    buf = b[["bid", "geometry"]].copy()
    buf["geometry"] = b.geometry.buffer(50)
    j = gpd.sjoin(buf, p[["poi_sem", "geometry"]], predicate="intersects", how="left")
    cnt = j.groupby("bid")["index_right"].count()
    maj = (j.dropna(subset=["poi_sem"]).groupby(["bid", "poi_sem"]).size()
             .rename("n").reset_index().sort_values("n", ascending=False)
             .drop_duplicates("bid").set_index("bid"))
    b["poi_n"] = cnt.reindex(b["bid"]).fillna(0).values
    b["poi_major_sem"] = maj["poi_sem"].reindex(b["bid"]).values
    b["poi_major_n"] = maj["n"].reindex(b["bid"]).fillna(0).values

    # ---- fusion cascade (verbatim probe: EULUC -> AOI -> OSM/Function -> POI -> morphology)
    b["elong"] = [minimum_rotated_rect_aspect(g) for g in b.geometry]
    b["gfa"] = b["fp_area"] * b["floors"]
    b["sem_euluc"] = b["euluc_class2"].map(EULUC_MAP)
    b["sem_aoi"] = b["aoi_type"].map(aoi_to_sem)
    b["sem_osm"] = b["osm_type"].map(OSM_MAP)
    name_admin = b["osm_name"].astype(str).str.contains("政府|法院|派出所|海关|委员会", na=False)
    name_bank = b["osm_name"].astype(str).str.contains("银行|证券|保险", na=False)
    b.loc[b["sem_osm"].isna() & name_admin, "sem_osm"] = "public_admin"
    b.loc[b["sem_osm"].isna() & name_bank, "sem_osm"] = "office_finance"
    b["sem_func"] = b["func"].map(FUNC_MAP)

    sem = pd.Series([None] * len(b), index=b.index, dtype=object)
    src = pd.Series([None] * len(b), index=b.index, dtype=object)
    for col, tag in [("sem_euluc", "euluc"), ("sem_aoi", "aoi"),
                     ("sem_osm", "osm"), ("sem_func", "function")]:
        m = b[col].notna()
        sem[m] = b.loc[m, col]
        src[m] = tag
    poi_fill = sem.isna() & b["poi_major_sem"].notna() & (b["poi_major_n"] >= 2)
    sem[poi_fill] = b.loc[poi_fill, "poi_major_sem"]
    src[poi_fill] = "poi"
    poi_pow = (sem == "residential_generic") & b["poi_major_sem"].isin(
        ["public_admin", "education", "healthcare"]) & (b["poi_major_n"] >= 3)
    sem[poi_pow] = b.loc[poi_pow, "poi_major_sem"]
    src[poi_pow] = "poi"
    is_low = (b["floors"] <= 3) & (b["fp_area"] < 300)
    gen = sem == "residential_generic"
    sem[gen & is_low] = "residential_lowrise"
    sem[gen & ~is_low] = "residential_estate"
    src[gen] = src[gen] + "+morpho"
    morpho_fb = sem.isna() & is_low & (b["euluc_class2"] == "居住用地")
    sem[morpho_fb] = "residential_lowrise"
    src[morpho_fb] = "morpho"
    sem[sem.isna()] = "unlabeled"
    b["semantic"] = sem
    b["judge"] = src.fillna("none")

    # ---- morphological family (verbatim probe cascade & thresholds)
    fam = pd.Series(["midrise_other"] * len(b), index=b.index, dtype=object)
    cond_house = (b["floors"] <= 3) & (b["fp_area"] < 400)
    cond_slab = (b["elong"] >= 2) & (b["floors"] >= 4) & (b["floors"] <= 12)
    cond_tower = (b["height"] >= 35) & (b["elong"] < 2)
    cond_large = b["fp_area"] >= 2000
    fam[cond_large] = "large_footprint"
    fam[cond_tower & ~cond_large] = "tower"
    fam[cond_slab & ~cond_tower & ~cond_large] = "slab"
    fam[cond_house] = "house"
    b["family"] = fam
    b["window"] = wid

    # per-source presence flags for the coverage table
    b["has_euluc"] = b["euluc_class2"].notna()
    b["has_aoi"] = b["sem_aoi"].notna()
    b["has_osm"] = b["sem_osm"].notna()
    b["has_func"] = b["func"].notna()
    b["has_poi"] = b["poi_n"] > 0
    keep = ["bid", "window", "semantic", "judge", "family", "height", "fp_area",
            "floors", "age", "elong", "gfa",
            "has_euluc", "has_aoi", "has_osm", "has_func", "has_poi", "geometry"]
    gpd.GeoDataFrame(b[keep], geometry="geometry", crs=CRS_M).to_parquet(out_path)
    lab = (b["semantic"] != "unlabeled").mean() * 100
    print(f"[extract:{wid}] n={len(b)} labeled={lab:.1f}% poi_pts={len(p)}")


def extract(ids):
    split = load_split()
    wmap = {w["id"]: w for w in split["windows"]}
    for wid in ids:
        extract_one(wmap[wid])


# ================================================================ LIBRARY
def library():
    split = load_split()
    parts, coverage = [], {}
    for w in split["windows"]:
        g = gpd.read_parquet(os.path.join(CACHE, f"{w['id']}_fused.parquet"))
        parts.append(g)
        coverage[w["id"]] = {
            "name": w["name"], "role": w["role"], "n_buildings": int(len(g)),
            "euluc": round(float(g["has_euluc"].mean()) * 100, 1) if len(g) else 0.0,
            "aoi": round(float(g["has_aoi"].mean()) * 100, 1) if len(g) else 0.0,
            "osm_type": round(float(g["has_osm"].mean()) * 100, 1) if len(g) else 0.0,
            "function_age": round(float(g["has_func"].mean()) * 100, 1) if len(g) else 0.0,
            "poi_50m_any": round(float(g["has_poi"].mean()) * 100, 1) if len(g) else 0.0,
            "age_filled": round(float(g["age"].notna().mean()) * 100, 1) if len(g) else 0.0,
            "fused_labeled": round(float((g["semantic"] != "unlabeled").mean()) * 100, 1) if len(g) else 0.0,
        }
    lib = pd.concat(parts, ignore_index=True)
    lib["bidx"] = np.arange(len(lib))
    lib = gpd.GeoDataFrame(lib, geometry="geometry", crs=CRS_M)
    lib.to_parquet(os.path.join(V03, "library.parquet"))
    tmp = "/tmp/library.fgb"
    lib.drop(columns=["has_euluc", "has_aoi", "has_osm", "has_func", "has_poi"]).to_file(
        tmp, driver="FlatGeobuf")
    shutil.copy(tmp, os.path.join(V03, "library.fgb"))
    judges = lib.groupby("window")["judge"].value_counts(normalize=True).round(3)
    results_io({"library": {
        "n_total": int(len(lib)),
        "coverage_pct": coverage,
        "judge_share_pooled": lib["judge"].value_counts(normalize=True).round(3).to_dict(),
    }})
    lab = [c["fused_labeled"] for c in coverage.values()]
    print(f"library n={len(lib)}  fused_labeled range {min(lab):.1f}-{max(lab):.1f}%")
    print(judges.head(20))


# ================================================================ LEARN (TRAIN only)
CELL = 50.0
NCELL = 20
FEATS = ["gfa", "coverage", "mean_h", "sh_house", "sh_slab", "sh_tower",
         "sh_large", "sh_mid", "dom_sem_share"]
K = 8
NTOK = K + 1  # T0 EMPTY + T1..T8


def build_cells():
    """50 m cell features for all 24 windows (cached)."""
    cpath = os.path.join(CACHE, "cells.parquet")
    if os.path.exists(cpath):
        return pd.read_parquet(cpath)
    split = load_split()
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))
    cent = lib.geometry.centroid
    lib["_cx"], lib["_cy"] = cent.x, cent.y
    rows = []
    for w in split["windows"]:
        g = lib[lib["window"] == w["id"]].copy()
        ox, oy = w["cx"] - HALF, w["cy"] - HALF
        g["col"] = np.clip(((g["_cx"] - ox) // CELL).astype(int), 0, NCELL - 1)
        g["row"] = np.clip(((g["_cy"] - oy) // CELL).astype(int), 0, NCELL - 1)
        grp = dict(tuple(g.groupby(["row", "col"])))
        for r in range(NCELL):
            for c in range(NCELL):
                rec = {"window": w["id"], "role": w["role"], "row": r, "col": c}
                sub = grp.get((r, c))
                if sub is None or len(sub) == 0:
                    rec.update({f: 0.0 for f in FEATS}); rec["n_bld"] = 0
                else:
                    fp = sub["fp_area"].sum()
                    rec["gfa"] = float(sub["gfa"].sum())
                    rec["coverage"] = float(min(fp / CELL ** 2, 1.0))
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
    cells.to_parquet(cpath)
    return cells


def learn():
    from sklearn.cluster import KMeans
    split = load_split()
    cells = build_cells()
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))

    empty = cells["coverage"] < 0.02
    train = cells["role"] == "train"
    fit_mask = train & ~empty
    X = cells.loc[fit_mask, FEATS].values
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    km = KMeans(n_clusters=K, random_state=0, n_init=10).fit((X - mu) / sd)
    raw_cent = pd.DataFrame(km.cluster_centers_ * sd + mu, columns=FEATS)
    order = np.argsort(raw_cent["gfa"].values)
    remap = {int(old): i + 1 for i, old in enumerate(order)}
    raw_cent = raw_cent.iloc[order].reset_index(drop=True)
    cells["token"] = 0
    cells.loc[fit_mask, "token"] = [remap[l] for l in km.labels_]
    # HELDOUT cells -> nearest TRAIN centroid (for skeletons/signatures only;
    # excluded from vocabulary fitting, adjacency and pools)
    ho_mask = (~train) & ~empty
    if ho_mask.sum():
        Zh = (cells.loc[ho_mask, FEATS].values - mu) / sd
        cells.loc[ho_mask, "token"] = [remap[l] for l in km.predict(Zh)]
    cells.to_parquet(os.path.join(CACHE, "cells.parquet"))  # now with tokens

    # measured-stat labels, computed on TRAIN cells only
    tok_labels, tok_stats = {}, {}
    fam_short = {"sh_house": "house", "sh_slab": "slab", "sh_tower": "tower",
                 "sh_large": "large", "sh_mid": "mid"}
    tr_cells = cells[train]
    tok_labels[0] = f"T0 EMPTY: cov<.02 (train n={int((tr_cells['token'] == 0).sum())})"
    tok_stats[0] = {"n_cells_train": int((tr_cells["token"] == 0).sum()),
                    "n_cells_heldout": int((cells[~train]["token"] == 0).sum())}
    for t in range(1, NTOK):
        sub = tr_cells[tr_cells["token"] == t]
        st = sub[FEATS].mean()
        shares = {k: st[k] for k in fam_short}
        top_f = max(shares, key=shares.get)
        tok_labels[t] = (f"T{t}: cov {st['coverage']:.2f} h̄ {st['mean_h']:.0f}m "
                         f"{fam_short[top_f]} {shares[top_f]:.2f}")
        tok_stats[t] = {**{k: round(float(st[k]), 3) for k in FEATS},
                        "n_cells_train": int(len(sub)),
                        "n_cells_heldout": int((cells[~train]["token"] == t).sum()),
                        "top_family": fam_short[top_f]}

    # ---- token grids x24 (saved for all; adjacency from TRAIN only)
    A = np.zeros((NTOK, NTOK))
    for w in split["windows"]:
        sub = cells[cells["window"] == w["id"]]
        gpatch = np.zeros((NCELL, NCELL), dtype=int)
        gpatch[sub["row"].values, sub["col"].values] = sub["token"].values
        np.save(os.path.join(V03, f"token_grid_{w['id']}.npy"), gpatch)
        if w["role"] != "train":
            continue
        for r in range(NCELL):
            for c in range(NCELL):
                for dr, dc in ((0, 1), (1, 0)):
                    rr, cc = r + dr, c + dc
                    if rr < NCELL and cc < NCELL:
                        a_, b_ = gpatch[r, c], gpatch[rr, cc]
                        A[a_, b_] += 1; A[b_, a_] += 1
    A_norm = A / A.sum(axis=1, keepdims=True).clip(min=1)

    # ---- instantiation pools (TRAIN cells only, cell-local coords)
    from shapely.affinity import translate
    origins = {w["id"]: (w["cx"] - HALF, w["cy"] - HALF) for w in split["windows"]}
    train_ids = {w["id"] for w in split["windows"] if w["role"] == "train"}
    cent = lib.geometry.centroid
    lib["col"] = 0; lib["row"] = 0
    for wid, (ox, oy) in origins.items():
        m = lib["window"] == wid
        lib.loc[m, "col"] = np.clip(((cent[m].x - ox) // CELL).astype(int), 0, NCELL - 1)
        lib.loc[m, "row"] = np.clip(((cent[m].y - oy) // CELL).astype(int), 0, NCELL - 1)
    cell_tok = cells.set_index(["window", "row", "col"])["token"]
    lib["token"] = cell_tok.loc[list(zip(lib["window"], lib["row"], lib["col"]))].values
    pool = lib[(lib["token"] > 0) & lib["window"].isin(train_ids)].copy()
    ox = pool["window"].map({k: v[0] for k, v in origins.items()})
    oy = pool["window"].map({k: v[1] for k, v in origins.items()})
    cx = pool.geometry.centroid.x; cy = pool.geometry.centroid.y
    pool["dx"] = (cx - ox - pool["col"] * CELL).round(2)
    pool["dy"] = (cy - oy - pool["row"] * CELL).round(2)
    shift_x = -(ox + pool["col"] * CELL); shift_y = -(oy + pool["row"] * CELL)
    pool["geometry"] = [translate(g, xoff=sx, yoff=sy)
                        for g, sx, sy in zip(pool.geometry, shift_x, shift_y)]
    cols = ["bidx", "bid", "window", "token", "row", "col", "dx", "dy",
            "height", "floors", "fp_area", "family", "semantic"]
    pool[cols].to_parquet(os.path.join(V03, "token_pools.parquet"))
    tmp = "/tmp/token_pools_v03.fgb"
    gpd.GeoDataFrame(pool[cols + ["geometry"]], geometry="geometry", crs=CRS_M).to_file(
        tmp, driver="FlatGeobuf")
    shutil.copy(tmp, os.path.join(V03, "token_pools.fgb"))

    json.dump({
        "features": FEATS,
        "feature_zscore_mu": [round(float(v), 4) for v in mu],
        "feature_zscore_sd": [round(float(v), 4) for v in sd],
        "empty_rule": "coverage < 0.02 -> T0 EMPTY (reserved, not clustered)",
        "kmeans": {"k": K, "random_state": 0, "ordering": "by mean GFA ascending",
                   "fit_on": "TRAIN windows' non-empty cells only "
                             f"({int(fit_mask.sum())} of 18x400=7200)"},
        "heldout_handling": "heldout cells assigned to nearest TRAIN centroid; "
                            "EXCLUDED from vocabulary fit, adjacency counts and pools",
        "labels": tok_labels, "stats": tok_stats,
        "raw_centroids": raw_cent.round(3).to_dict("records"),
        "pool_sizes": {f"T{t}": int((pool["token"] == t).sum()) for t in range(1, NTOK)},
    }, open(os.path.join(V03, "tokens.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"tokens": [f"T{t}" for t in range(NTOK)],
               "source": "TRAIN windows only (18 grids), 4-neighbour, pooled",
               "counts_pooled_4neighbour": A.astype(int).tolist(),
               "row_normalized": np.round(A_norm, 4).tolist()},
              open(os.path.join(V03, "adjacency.json"), "w"), indent=2)
    results_io({"learn": {
        "k": K, "fit_cells": int(fit_mask.sum()),
        "token_labels": tok_labels,
        "token_cell_counts_train": {f"T{t}": int((tr_cells["token"] == t).sum()) for t in range(NTOK)},
        "token_cell_counts_heldout": {f"T{t}": int((cells[~train]["token"] == t).sum()) for t in range(NTOK)},
        "pool_sizes_buildings": {f"T{t}": int((pool["token"] == t).sum()) for t in range(1, NTOK)},
    }})
    print(json.dumps(tok_labels, ensure_ascii=False, indent=1))


# ================================================================ SIGNATURES + PCA cloud
IND = ["FAR", "coverage", "h_p50", "h_p90", "median_fp",
       "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]


def signatures():
    from sklearn.decomposition import PCA
    split = load_split()
    cells = pd.read_parquet(os.path.join(CACHE, "cells.parquet"))
    lib = gpd.read_parquet(os.path.join(V03, "library.parquet"))
    AREA = (2 * HALF) ** 2
    rows = []
    for w in split["windows"]:
        g = lib[lib["window"] == w["id"]]
        sub = cells[cells["window"] == w["id"]]
        comp = [float((sub["token"] == t).mean()) for t in range(NTOK)]
        fp = g["fp_area"].sum()
        fam_fp = g.groupby("family")["fp_area"].sum()
        ind = {
            "FAR": float(g["gfa"].sum() / AREA),
            "coverage": float(fp / AREA),
            "h_p50": float(np.percentile(g["height"], 50)) if len(g) else 0.0,
            "h_p90": float(np.percentile(g["height"], 90)) if len(g) else 0.0,
            "median_fp": float(g["fp_area"].median()) if len(g) else 0.0,
        }
        for f, key in zip(["sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"],
                          ["house", "slab", "tower", "large_footprint", "midrise_other"]):
            ind[f] = float(fam_fp.get(key, 0.0) / fp) if fp > 0 else 0.0
        rows.append({"id": w["id"], "name": w["name"], "role": w["role"],
                     "type_group": w["type_group"],
                     "token_comp": [round(v, 4) for v in comp], **ind})
    sig = pd.DataFrame(rows)

    feat_names = [f"tok_T{t}" for t in range(NTOK)] + IND
    M = np.hstack([np.array(sig["token_comp"].tolist()),
                   sig[IND].values])
    tr = (sig["role"] == "train").values
    mu, sd = M[tr].mean(0), M[tr].std(0); sd[sd == 0] = 1
    Z = (M - mu) / sd
    pca = PCA(n_components=2, random_state=0).fit(Z[tr])
    comps = pca.components_.copy()
    XY = Z @ comps.T
    for k in range(2):  # orient: top-|loading| feature positive
        j = int(np.argmax(np.abs(comps[k])))
        if comps[k, j] < 0:
            comps[k] *= -1; XY[:, k] *= -1
    loadings = {f"PC{k+1}": {feat_names[i]: round(float(comps[k, i]), 3)
                             for i in np.argsort(-np.abs(comps[k]))[:6]}
                for k in range(2)}
    out = {
        "features": feat_names,
        "zscore": "mu/sd from TRAIN windows only",
        "pca": {"fit_on": "TRAIN (18)", "explained_variance_ratio":
                [round(float(v), 3) for v in pca.explained_variance_ratio_],
                "top_loadings": loadings},
        "windows": [],
    }
    for i, r in sig.iterrows():
        out["windows"].append({
            "id": r["id"], "name": r["name"], "role": r["role"],
            "type_group": r["type_group"],
            "token_comp_T0_T8": r["token_comp"],
            "indicators": {k: round(float(r[k]), 4) for k in IND},
            "pca_xy": [round(float(XY[i, 0]), 4), round(float(XY[i, 1]), 4)],
        })
    json.dump(out, open(os.path.join(V03, "signatures.json"), "w"), ensure_ascii=False, indent=2)
    results_io({"signatures": {"explained_variance_ratio": out["pca"]["explained_variance_ratio"],
                               "top_loadings": loadings}})
    print(json.dumps(loadings, indent=1))
    print(sig[["id", "name", "role", "FAR", "coverage", "h_p90"]].round(2).to_string(index=False))


# ================================================================ FIG 1 — library map + signature cloud
def _smart_label(ax, x, y, text, placed, xr, yr, fontsize=6.4, fontweight="normal"):
    """Greedy collision-avoiding annotation: try 4 offsets, keep first free slot."""
    cands = [((5, 4), "left"), ((5, -9), "left"), ((-5, 4), "right"), ((-5, -9), "right")]
    for (dx, dy), ha in cands:
        ax_, ay_ = x + dx * xr / 400, y + dy * yr / 400
        if all((abs(ax_ - px) / xr) ** 2 + (abs(ay_ - py) / yr) ** 2 > 0.0022
               for px, py in placed):
            placed.append((ax_, ay_))
            ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, fontsize=fontsize, fontweight=fontweight)
            return
    placed.append((x, y))
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(5, 4),
                fontsize=fontsize, fontweight=fontweight)


def fig1():
    sigs = json.load(open(os.path.join(V03, "signatures.json")))
    split = load_split()
    smap = {s["id"]: s for s in split["windows"]}
    W = sigs["windows"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # left: window locations — label notable ones (forced + heldout + peripheral)
    lons = [smap[r["id"]]["lon"] for r in W]; lats = [smap[r["id"]]["lat"] for r in W]
    xr, yr = max(lons) - min(lons), max(lats) - min(lats)
    placed = []
    for r in W:
        s = smap[r["id"]]
        mk = "o" if r["role"] == "train" else "*"
        sz = 70 if r["role"] == "train" else 150
        col = "#2a9d8f" if r["role"] == "train" else "#e76f51"
        axL.scatter(s["lon"], s["lat"], marker=mk, s=sz, color=col,
                    edgecolor="black", linewidth=0.6, zorder=3)
        peripheral = abs(s["lon"] - 121.46) > 0.10 or abs(s["lat"] - 31.23) > 0.07
        if r["id"] in FORCED or r["role"] == "heldout" or peripheral:
            _smart_label(axL, s["lon"], s["lat"], r["name"], placed, xr, yr,
                         fontsize=6.6, fontweight="bold" if r["id"] in FORCED else "normal")
    axL.set_aspect(1 / np.cos(np.radians(31.2)))
    axL.set_xlabel("longitude"); axL.set_ylabel("latitude")
    axL.grid(alpha=0.3)
    axL.set_title("24 one-km windows across Shanghai\n"
                  "(18 TRAIN circles, 6 HELDOUT stars; labels: notable windows)", fontsize=11)
    axL.legend(handles=[
        Line2D([], [], marker="o", ls="", color="#2a9d8f", markeredgecolor="black", label="TRAIN (18)"),
        Line2D([], [], marker="*", ls="", color="#e76f51", markersize=12, markeredgecolor="black", label="HELDOUT (6)")],
        fontsize=9, loc="lower left")

    # right: signature cloud — label all 24
    ld = sigs["pca"]["top_loadings"]
    evr = sigs["pca"]["explained_variance_ratio"]
    xs = [r["pca_xy"][0] for r in W]; ys = [r["pca_xy"][1] for r in W]
    xr, yr = max(xs) - min(xs), max(ys) - min(ys)
    placed = []
    for r in W:
        x, y = r["pca_xy"]
        mk = "o" if r["role"] == "train" else "*"
        sz = 75 if r["role"] == "train" else 170
        col = "#2a9d8f" if r["role"] == "train" else "#e76f51"
        axR.scatter(x, y, marker=mk, s=sz, color=col, edgecolor="black", linewidth=0.6, zorder=3)
        _smart_label(axR, x, y, r["name"], placed, xr, yr, fontsize=6.6,
                     fontweight="bold" if r["id"] in FORCED else "normal")

    def axis_lab(pc):
        items = list(ld[pc].items())
        pos = [f"{k} {v:+.2f}" for k, v in items if v > 0][:3]
        neg = [f"{k} {v:+.2f}" for k, v in items if v < 0][:3]
        return f"{pc} ({evr[int(pc[-1])-1]*100:.0f}% var)  +: {', '.join(pos)}   -: {', '.join(neg)}"
    axR.set_xlabel(axis_lab("PC1"), fontsize=8.5)
    axR.set_ylabel(axis_lab("PC2"), fontsize=8.5)
    axR.axhline(0, color="gray", lw=0.5, alpha=0.5); axR.axvline(0, color="gray", lw=0.5, alpha=0.5)
    axR.grid(alpha=0.3)
    axR.set_title("Signature cloud — PCA of z-scored window signatures\n"
                  "(9-dim token composition + 10 indicators; fit on TRAIN, all projected)",
                  fontsize=11)
    fig.suptitle("v0.3 FIG 1 — Patch library (24 windows) and the continuous signature cloud",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(V03, "fig1_library.png"), dpi=130)
    plt.close(fig)
    print("fig1_library.png written")


# ================================================================ FIG 2 — grammar
TOKEN_COLORS = ["#f2f2f2", "#e9c46a", "#f4a261", "#e76f51", "#d62828",
                "#2a9d8f", "#457b9d", "#6a4c93", "#1d3557"]


def fig2():
    split = load_split()
    toks = json.load(open(os.path.join(V03, "tokens.json")))
    adj = json.load(open(os.path.join(V03, "adjacency.json")))
    A_norm = np.array(adj["row_normalized"])
    tok_labels = {int(k): v for k, v in toks["labels"].items()}
    raw_cent = pd.DataFrame(toks["raw_centroids"])

    train_ws = [w for w in split["windows"] if w["role"] == "train"]
    reps = [w for w in train_ws if w["id"] in FORCED]
    for grp in ["industrial", "new_town", "heritage_watertown", "urban_village",
                "developer_resid", "midrise_resid"]:
        if len(reps) >= 6:
            break
        cand = [w for w in train_ws if w["type_group"] == grp and w not in reps]
        if cand:
            reps.append(cand[0])
    reps = reps[:6]

    fig = plt.figure(figsize=(17, 7.2))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.85, 0.85, 0.85, 1.25, 1.35],
                          hspace=0.25, wspace=0.3)
    cmap = ListedColormap(TOKEN_COLORS)
    for k, w in enumerate(reps):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        g = np.load(os.path.join(V03, f"token_grid_{w['id']}.npy"))
        ax.imshow(g, cmap=cmap, vmin=0, vmax=NTOK - 1, origin="lower")
        ax.set_title(w["name"], fontsize=9, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[:, 3])
    im = ax.imshow(A_norm, cmap="YlGnBu", vmin=0, vmax=A_norm.max())
    ax.set_xticks(range(NTOK), [f"T{t}" for t in range(NTOK)], fontsize=8.5)
    ax.set_yticks(range(NTOK), [f"T{t}" for t in range(NTOK)], fontsize=8.5)
    for i in range(NTOK):
        for j in range(NTOK):
            ax.text(j, i, f"{A_norm[i, j]:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if A_norm[i, j] > 0.5 * A_norm.max() else "black")
    ax.set_title("Token adjacency (4-neighbour,\nTRAIN windows only, row-normalized)", fontsize=10.5)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax = fig.add_subplot(gs[:, 4])
    x = np.arange(len(FEATS)); wdt = 0.105
    disp = raw_cent.copy()
    disp["gfa"] = disp["gfa"] / disp["gfa"].max()
    disp["mean_h"] = disp["mean_h"] / disp["mean_h"].max()
    for t in range(1, NTOK):
        vals = disp.iloc[t - 1][FEATS].values.astype(float)
        ax.bar(x + (t - 4.5) * wdt, vals, wdt, color=TOKEN_COLORS[t], label=f"T{t}")
    ax.set_xticks(x, ["GFA/max", "cov", "h̄/max", "house", "slab", "tower",
                      "large", "mid", "domsem"], fontsize=7.5, rotation=35, ha="right")
    ax.set_title("Per-token mean profile (TRAIN cells,\nmeasured stats)", fontsize=10.5)
    ax.legend(fontsize=8, ncol=4, frameon=False); ax.grid(axis="y", alpha=0.3)
    handles = [Patch(facecolor=TOKEN_COLORS[t], label=tok_labels[t]) for t in range(NTOK)]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.085),
               ncol=3, fontsize=8.2, frameon=False)
    fig.suptitle("v0.3 FIG 2 — Token grammar learned from 18 TRAIN windows "
                 "(KMeans k=8 + reserved EMPTY; labels = measured stats only)", fontsize=13)
    fig.savefig(os.path.join(V03, "fig2_grammar.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("fig2_grammar.png written")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "select":
        select()
    elif cmd == "extract":
        extract(sys.argv[2:])
    elif cmd == "library":
        library()
    elif cmd == "learn":
        learn()
    elif cmd == "signatures":
        signatures()
    elif cmd == "fig1":
        fig1()
    elif cmd == "fig2":
        fig2()
    else:
        raise SystemExit(f"unknown cmd {cmd}")
