#!/usr/bin/env python3
"""
audit_v06.py — independent adversarial spot-check of v06 (fresh re-derivations,
no imports from exp_* / generate / translate scripts).

Checks implemented here (web checks of doctrine numbers are done outside, by hand):
  C2: achieved token composition of unit_neighbourhood_unit.parquet
      (building rows + skeleton npy) vs results_d3.json
  C3: read-back profile of unit_ville_radieuse.parquet recomputed directly from
      power_priors_v05.json (GFA-weighted, no_age) vs results_d4.json (3 dp)
  C4: D5 correctness gate — reproject 3 real windows (A01, A13, A47) through the
      recorded pipeline vs stored pca_xy in TF/v03/signatures.json
  C5: D6 expected profile of pure gb50180 + L1 to DEV-first vs results_d6.json
  C7: internal-consistency sweep across results_d1..d6

All numbers printed side by side; nothing silently fixed. Seed 0 (deterministic).
"""
import json
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation"
V03, V05, V06 = f"{ROOT}/v03", f"{ROOT}/v05", f"{ROOT}/v06"
CACHE = f"{V06}/cache_gen"
DIMS = ["government", "developer_capital", "resident_community", "informal"]
PARADIGMS = ["garden_city", "ville_radieuse", "neighbourhood_unit",
             "microrayon", "gb50180"]
np.random.seed(0)

out = {}

d1 = json.load(open(f"{V06}/results_d1.json"))
d2 = json.load(open(f"{V06}/results_d2.json"))
d3 = json.load(open(f"{V06}/results_d3.json"))
d4 = json.load(open(f"{V06}/results_d4.json"))
d5 = json.load(open(f"{V06}/results_d5.json"))
d6 = json.load(open(f"{V06}/results_d6.json"))
rec = json.load(open(f"{V06}/doctrine_recipes.json"))["doctrines"]

# ============================================================ C2: NU unit comp
print("=" * 70)
print("C2: unit_neighbourhood_unit achieved token composition")
nu = pd.read_parquet(f"{CACHE}/unit_neighbourhood_unit.parquet",
                     columns=["token", "tgt_row", "tgt_col"])
# built cells: unique (tgt_row, tgt_col); token per cell (check uniqueness)
cell_tok = nu.groupby(["tgt_row", "tgt_col"])["token"].agg(["nunique", "first"])
assert (cell_tok["nunique"] == 1).all(), "cell with >1 token!"
built_counts = cell_tok["first"].value_counts().to_dict()
built_counts = {f"T{int(k)}": int(v) for k, v in built_counts.items()}

skel = np.load(f"{V06}/skeleton_neighbourhood_unit.npy")
flat = skel.ravel()
n_build_skel = int((flat == -1).sum())  # -1 = buildable
skel_counts = {"E_ROAD": int((flat == 9).sum()),
               "E_WATER": int((flat == 10).sum()),
               "E_GREEN": int((flat == 11).sum()),
               "E_VACANT_skel0": int((flat == 0).sum())}
n_built_cells = int(len(cell_tok))
recomp = dict(built_counts)
recomp["E_ROAD"] = skel_counts["E_ROAD"]
recomp["E_GREEN"] = skel_counts["E_GREEN"]
recomp["E_VACANT"] = skel_counts["E_VACANT_skel0"] + (n_build_skel - n_built_cells)
claimed = d3["paradigms"]["neighbourhood_unit"]["unit"]["token_composition"]["achieved"]
print(f"  skeleton shape {skel.shape} dtype {skel.dtype} "
      f"values {sorted(set(flat.tolist()))}")
print(f"  build cells in skeleton (-1): {n_build_skel}; "
      f"built cells in parquet: {n_built_cells}")
print(f"  {'token':10s} {'recomputed':>10s} {'results_d3':>10s}")
keys = sorted(set(list(recomp) + list(claimed)))
ok = True
for k in keys:
    a, b = recomp.get(k, 0), claimed.get(k, 0)
    flag = "" if a == b else "  <-- MISMATCH"
    if a != b:
        ok = False
    print(f"  {k:10s} {a:>10d} {b:>10d}{flag}")
print(f"  total recomputed {sum(recomp.values())} (grid is 20x20=400)")
print(f"  C2 verdict: {'MATCH' if ok else 'MISMATCH'}")
out["C2"] = {"recomputed": recomp, "claimed": claimed, "match": ok}

# ============================================================ C3: VR read-back
print("=" * 70)
print("C3: unit_ville_radieuse read-back from power_priors_v05 directly")
PP = json.load(open(f"{V05}/power_priors_v05.json"))
PRI, BASE = PP["priors"], PP["baseline_unspecified"]


def w_of(sem):
    if sem == "residential_estate":
        return PRI["residential_estate"]["cells"]["no_age"]["w"]
    if sem in PRI:
        return PRI[sem]["w"]
    return BASE


vr = pd.read_parquet(f"{CACHE}/unit_ville_radieuse.parquet",
                     columns=["semantic", "gfa"])
lab = vr[vr["semantic"] != "unlabeled"]
wts = lab["gfa"].to_numpy()
prof = {}
for d in DIMS:
    vals = np.array([w_of(s)[d] for s in lab["semantic"]])
    prof[d] = float(np.average(vals, weights=wts))
claimed_r0 = d4["paradigms"]["ville_radieuse"]["unit"]["readback_R0"]
print(f"  n buildings total {len(vr)}, labeled {len(lab)} "
      f"(d4 claims n_labeled={d4['paradigms']['ville_radieuse']['unit']['n_buildings_labeled']})")
print(f"  {'dim':22s} {'recomputed_raw':>14s} {'results_d4(4dp)':>15s}")
ok = True
for d in DIMS:
    # results_d4 stores 4-dp rounding; compare raw to stored at 4-dp tolerance.
    m = abs(prof[d] - claimed_r0[d]) <= 5.001e-5
    if not m:
        ok = False
    print(f"  {d:22s} {prof[d]:>14.6f} {claimed_r0[d]:>15.4f}"
          f"{'' if m else '  <-- MISMATCH'}")
print("  note: at a literal 3-dp print, resident_community is a rounding-"
      "boundary case (raw .298531 -> .299; python round(.2985,3) -> .298); "
      "raw values agree to the stored 4-dp precision.")
print(f"  C3 verdict: {'MATCH (4 dp)' if ok else 'MISMATCH'}")
out["C3"] = {"recomputed_raw": {d: round(prof[d], 6) for d in DIMS},
             "claimed_4dp": claimed_r0, "match": bool(ok)}

# ============================================================ C4: D5 gate
print("=" * 70)
print("C4: reproject 3 real windows (A01, A13, A47) vs stored pca_xy")
sigs = json.load(open(f"{V03}/signatures.json"))
wmeta = sigs["windows"]
ids = [w["id"] for w in wmeta]
roles = [w["role"] for w in wmeta]
NTOK = 9
IND = ["FAR", "coverage", "h_p50", "h_p90", "median_fp",
       "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
FAMK = [("sh_house", "house"), ("sh_slab", "slab"), ("sh_tower", "tower"),
        ("sh_large", "large_footprint"), ("sh_mid", "midrise_other")]
AREA = 1000.0 ** 2
cells = pd.read_parquet(f"{V03}/cache/cells.parquet")
lib = gpd.read_parquet(f"{V03}/library.parquet")
M = np.zeros((len(ids), NTOK + len(IND)))
for i, wid in enumerate(ids):
    sub = cells[cells["window"] == wid]
    comp = [float((sub["token"] == t).mean()) for t in range(NTOK)]
    g = lib[lib["window"] == wid]
    fp = float(g["fp_area"].sum())
    fam = g.groupby("family")["fp_area"].sum()
    ind = {"FAR": float(g["gfa"].sum() / AREA), "coverage": fp / AREA,
           "h_p50": float(np.percentile(g["height"], 50)),
           "h_p90": float(np.percentile(g["height"], 90)),
           "median_fp": float(g["fp_area"].median())}
    for f, key in FAMK:
        ind[f] = float(fam.get(key, 0.0) / fp) if fp > 0 else 0.0
    M[i] = comp + [ind[k] for k in IND]
tr = np.array([r == "train" for r in roles])
mu, sd = M[tr].mean(0), M[tr].std(0)
sd[sd == 0] = 1
Z = (M - mu) / sd
pca = PCA(n_components=2, random_state=0).fit(Z[tr])
comps = pca.components_.copy()
XY = Z @ comps.T
for k in range(2):
    j = int(np.argmax(np.abs(comps[k])))
    if comps[k, j] < 0:
        XY[:, k] *= -1
stored = np.array([w["pca_xy"] for w in wmeta])
err_all = float(np.max(np.abs(XY - stored)))
print(f"  recomputed-vs-stored max|err| over ALL 24 windows: {err_all:.6f} "
      f"(d5 claims gate max_abs_err={d5['correctness_gate']['max_abs_err']})")
ok = True
for wid in ["A01", "A13", "A47"]:
    i = ids.index(wid)
    e = float(np.max(np.abs(XY[i] - stored[i])))
    m = e <= 1e-3
    ok &= m
    print(f"  {wid}: recomputed ({XY[i,0]:+.4f}, {XY[i,1]:+.4f})  "
          f"stored ({stored[i,0]:+.4f}, {stored[i,1]:+.4f})  "
          f"max|err|={e:.6f}  {'OK' if m else 'FAIL'}")
print(f"  explained_variance_ratio recomputed: "
      f"{[round(float(v),4) for v in pca.explained_variance_ratio_]} "
      f"(d5 claims {d5['correctness_gate']['explained_variance_ratio']})")
print(f"  C4 verdict: {'GATE REPRODUCED' if ok and err_all <= 1e-3 else 'FAIL'}")
out["C4"] = {"max_err_all24": err_all, "passed": bool(ok and err_all <= 1e-3)}

# ============================================================ C5: D6 gb50180
print("=" * 70)
print("C5: D6 expected profile of pure gb50180 + L1 to DEV-first")
tp = pd.read_parquet(f"{V03}/token_pools.parquet")
tp["key"] = (tp["window"].astype(str) + "_r" + tp["row"].astype(str)
             + "_c" + tp["col"].astype(str))
tp["gfa_b"] = tp["fp_area"] * tp["floors"]
budget = d2["paradigms"]["gb50180"]["token_budget"]  # T4:81 T6:75 T2:6 T7:16
tok_exp = {}
for t, grp in tp.groupby("token"):
    t = int(t)
    if not 1 <= t <= 8:
        continue
    n_cells = grp["key"].nunique()
    labp = grp[grp["semantic"] != "unlabeled"]
    W = np.array([[w_of(s)[d] for d in DIMS] for s in labp["semantic"]])
    ww = labp["gfa_b"].to_numpy()
    tok_exp[t] = {"n_pool_cells": int(n_cells),
                  "E_gfa": float(ww.sum() / n_cells),
                  "w": np.average(W, axis=0, weights=ww)}
num, den = np.zeros(4), 0.0
for tk, n in budget.items():
    if not tk.startswith("T"):
        continue
    t = int(tk[1:])
    e = tok_exp[t]
    num += n * e["E_gfa"] * e["w"]
    den += n * e["E_gfa"]
exp_prof = {d: round(float(num[i] / den), 4) for i, d in enumerate(DIMS)}
tgt = d6["targets"]["DEV-first"]
l1 = round(sum(abs(exp_prof[d] - tgt[d]) for d in DIMS), 4)
row = next(r for r in d6["ranking"]["DEV-first"]["table"]
           if r["candidate"] == "gb50180")
print(f"  {'dim':22s} {'recomputed':>10s} {'results_d6':>10s}")
ok = True
for d in DIMS:
    a, b = exp_prof[d], row["expected_profile"][d]
    flag = "" if abs(a - b) <= 5e-4 else "  <-- MISMATCH"
    if abs(a - b) > 5e-4:
        ok = False
    print(f"  {d:22s} {a:>10.4f} {b:>10.4f}{flag}")
print(f"  L1 to DEV-first: recomputed {l1:.4f} vs d6 {row['L1_to_target']:.4f} "
      f"(rank_L1 claimed {row['rank_L1']})")
ok &= abs(l1 - row["L1_to_target"]) <= 2e-3
# also spot-check the published TOK_EXP table for the 4 tokens used
for t in [2, 4, 6, 7]:
    c = d6["token_pool_expectation"][f"T{t}"]
    a = tok_exp[t]
    dmax = max(abs(a["w"][i] - c["expected_power"][d]) for i, d in enumerate(DIMS))
    print(f"  T{t}: n_cells {a['n_pool_cells']} vs {c['n_pool_cells']}; "
          f"E_gfa {a['E_gfa']:.1f} vs {c['E_gfa_labeled_per_cell_m2']}; "
          f"max|w err| {dmax:.4f}")
    ok &= (a["n_pool_cells"] == c["n_pool_cells"]) and dmax <= 5e-4
print(f"  C5 verdict: {'MATCH' if ok else 'MISMATCH'}")
out["C5"] = {"recomputed_profile": exp_prof, "claimed": row["expected_profile"],
             "L1_recomputed": l1, "L1_claimed": row["L1_to_target"], "match": ok}

# ============================================================ C7: consistency
print("=" * 70)
print("C7: internal-consistency sweep d1..d6")
issues = []

# d1: recount confidence + doctrine_silent inside quantities
def walk(node, conf, silent):
    if isinstance(node, dict):
        if "confidence" in node and isinstance(node.get("confidence"), str):
            conf.append(node["confidence"])
        if node.get("doctrine_silent") is True:
            silent.append(1)
        for v in node.values():
            walk(v, conf, silent)


tot_conf, tot_sil = [], []
per_doc = {}
for p in PARADIGMS:
    conf, sil = [], []
    walk(rec[p]["quantities"], conf, sil)
    per_doc[p] = {"high": conf.count("high"), "medium": conf.count("medium"),
                  "low": conf.count("low"), "silent": len(sil)}
    tot_conf += conf
    tot_sil += sil
cc = d1["confidence_counts_quantities_total"]
print(f"  d1 confidence totals: recomputed high {tot_conf.count('high')} / "
      f"med {tot_conf.count('medium')} / low {tot_conf.count('low')}; "
      f"claimed {cc['high']}/{cc['medium']}/{cc['low']}")
if (tot_conf.count("high"), tot_conf.count("medium"), tot_conf.count("low")) != \
        (cc["high"], cc["medium"], cc["low"]):
    issues.append("d1 confidence totals mismatch with doctrine_recipes recount")
ds = d1["doctrine_silent_counts"]
for p in PARADIGMS:
    if per_doc[p]["silent"] != ds[p]:
        issues.append(f"d1 doctrine_silent {p}: recount {per_doc[p]['silent']} "
                      f"vs claimed {ds[p]}")
print(f"  d1 doctrine_silent per doctrine: recount "
      f"{ {p: per_doc[p]['silent'] for p in PARADIGMS} } vs claimed {ds} "
      f"(total recount {len(tot_sil)})")
# d1 apriori vs recipes
for p in PARADIGMS:
    if d1["stakeholder_apriori"][p] != rec[p]["stakeholder_apriori"]["config"]:
        issues.append(f"d1 apriori {p} != doctrine_recipes config")

# d2 budgets sum to 400; d2 == d3 demanded == d3 achieved; skeleton npy counts
for p in PARADIGMS:
    b = d2["paradigms"][p]["token_budget"]
    if sum(b.values()) != 400:
        issues.append(f"d2 {p} token_budget sums to {sum(b.values())} != 400")
    dem = d3["paradigms"][p]["unit"]["token_composition"]["demanded_d2"]
    ach = d3["paradigms"][p]["unit"]["token_composition"]["achieved"]
    bb = {k: v for k, v in b.items() if v > 0 or k in dem}
    if {k: v for k, v in b.items() if v > 0} != {k: v for k, v in dem.items() if v > 0}:
        issues.append(f"d2 budget != d3 demanded for {p}: {b} vs {dem}")
    if {k: v for k, v in dem.items() if v > 0} != {k: v for k, v in ach.items() if v > 0}:
        issues.append(f"d3 demanded != achieved for {p}")
    sk = np.load(f"{V06}/skeleton_{p}.npy")
    f = sk.ravel()
    sc = d2["paradigms"][p]["skeleton_counts"]
    chk = {"road": int((f == 9).sum()), "green": int((f == 11).sum()),
           "water": int((f == 10).sum())}
    for k, v in chk.items():
        if sc[k] != v:
            issues.append(f"d2 {p} skeleton_counts[{k}]={sc[k]} but npy has {v}")
    if sk.shape != (20, 20) or sk.dtype != np.int16:
        issues.append(f"skeleton {p} shape/dtype {sk.shape}/{sk.dtype}")
    # vacant_build + civic should equal -1 cells + 0 cells? D2 semantics:
    nb = int((f == -1).sum()) + int((f == 0).sum())
    if sc["vacant_build"] + sc["civic"] != nb:
        issues.append(f"d2 {p}: vacant_build {sc['vacant_build']} + civic "
                      f"{sc['civic']} != skeleton -1/0 cells {nb}")
    # d3 unit grid npy vs achieved
    gu = np.load(f"{CACHE}/grid_unit_{p}.npy").ravel()
    g_ach = {"E_VACANT": int((gu == 0).sum()), "E_ROAD": int((gu == 9).sum()),
             "E_GREEN": int((gu == 11).sum())}
    for t in range(1, 9):
        n = int((gu == t).sum())
        if n:
            g_ach[f"T{t}"] = n
    if {k: v for k, v in g_ach.items() if v > 0} != {k: v for k, v in ach.items() if v > 0}:
        issues.append(f"d3 {p} achieved != grid_unit npy: {ach} vs {g_ach}")
    # d5 token comp = fold of grid
    fold = gu.copy()
    fold[fold >= 9] = 0
    comp = [round(float((fold == t).mean()), 4) for t in range(9)]
    if comp != d5["paradigm_signatures"][p]["token_comp_T0_T8"]:
        issues.append(f"d5 {p} token_comp mismatch with folded grid")
    # n_buildings consistency d3 vs d5 vs parquet
    nb3 = d3["paradigms"][p]["unit"]["n_buildings"]
    nb5 = d5["paradigm_signatures"][p]["n_buildings"]
    npq = len(pd.read_parquet(f"{CACHE}/unit_{p}.parquet", columns=["token"]))
    if not (nb3 == nb5 == npq):
        issues.append(f"{p} unit n_buildings d3 {nb3} / d5 {nb5} / parquet {npq}")
    # d4 labeled+unlabeled-share sanity: n_labeled <= parquet rows
    if d4["paradigms"][p]["unit"]["n_buildings_labeled"] > npq:
        issues.append(f"{p} d4 n_labeled > parquet rows")

# d4 flip table vs per-paradigm blocks; headline strings
for r in d4["ranking_flip_table"]:
    p = r["paradigm"]
    u = d4["paradigms"][p]["unit"]
    if (r["doctrine_claims_first"] != u["doctrine_first_dim"]
            or r["fabric_reads_first"] != u["fabric_first_dim"]
            or r["flip"] != u["ranking_flip"]
            or r["L1_gap_unit"] != u["L1_gap"]):
        issues.append(f"d4 flip_table row inconsistent for {p}")
    # decomposition telescoping: a+b+c == gap per dim (4 dp tolerance)
    for d in DIMS:
        s = (u["decomposition"]["a_no_age_artifact"][d]
             + u["decomposition"]["b_collage_semantic_drift"][d]
             + u["decomposition"]["c_genuine_program_translation"][d])
        if abs(s - u["gap_signed_apriori_minus_readback"][d]) > 2e-4:
            issues.append(f"d4 {p} decomposition does not telescope on {d}")
n_flips = sum(r["flip"] for r in d4["ranking_flip_table"])
print(f"  d4 flips: {n_flips}/5 (claimed 4/5)")
if n_flips != 4:
    issues.append(f"d4 flips {n_flips} != claimed 4")

# d6: 15 candidates per target, ranking sorted by L1, winner = rank1
for tgt_name, blk in d6["ranking"].items():
    tab = blk["table"]
    if len(tab) != 15:
        issues.append(f"d6 {tgt_name} has {len(tab)} candidates != 15")
    l1s = [r["L1_to_target"] for r in tab]
    if l1s != sorted(l1s):
        issues.append(f"d6 {tgt_name} table not sorted by L1")
    if tab[0]["candidate"] != blk["winner_L1"]:
        issues.append(f"d6 {tgt_name} winner_L1 != rank-1 row")
    tv = np.array([blk["target"][d] for d in DIMS])
    for r in tab:
        ev = np.array([r["expected_profile"][d] for d in DIMS])
        if abs(float(np.abs(ev - tv).sum()) - r["L1_to_target"]) > 2e-3:
            issues.append(f"d6 {tgt_name} {r['candidate']} L1 wrong")
        cs = float(ev @ tv / (np.linalg.norm(ev) * np.linalg.norm(tv)))
        if abs(cs - r["cosine_to_target"]) > 2e-3:
            issues.append(f"d6 {tgt_name} {r['candidate']} cosine wrong")
# d6 winners achieved tokens sum to 400 for blend
at = d6["winners"]["GOV-first"]["generation"]["achieved_tokens"]
if sum(at.values()) != 400:
    issues.append(f"d6 blend achieved tokens sum {sum(at.values())} != 400")
# d6 pure-paradigm winners' realized read-back must equal d4 R0
for tname, pname in [("DEV-first", "ville_radieuse"),
                     ("RES-first", "neighbourhood_unit")]:
    a = d6["winners"][tname]["realized_readback_R0"]
    b = d4["paradigms"][pname]["unit"]["readback_R0"]
    if any(abs(a[d] - b[d]) > 1e-4 for d in DIMS):
        issues.append(f"d6 {tname} realized R0 != d4 {pname} R0")
# d6 expected profile of pure paradigms = TOK_EXP-weighted budgets; verified for
# gb50180 in C5; also check ville_radieuse (single token => = T6 vector)
vr_exp = d6["ranking"]["DEV-first"]["table"][0]["expected_profile"]
t6w = d6["token_pool_expectation"]["T6"]["expected_power"]
if any(abs(vr_exp[d] - t6w[d]) > 1e-4 for d in DIMS):
    issues.append("d6 ville_radieuse expected profile != T6 pool vector")

print(f"  issues found: {len(issues)}")
for s in issues:
    print(f"   - {s}")
out["C7"] = {"issues": issues}

json.dump(out, open(f"{V06}/results_audit.json", "w"), indent=1, default=bool)
print("\nwritten -> results_audit.json")
