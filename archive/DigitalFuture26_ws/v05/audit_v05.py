#!/usr/bin/env python3
"""
audit_v05.py — independent adversarial spot-checks of the v05 results
BEFORE they enter the report. Re-derives numbers from raw inputs
(dataset.parquet, power_priors_v05.json, signatures.json, cf_DEV.parquet)
WITHOUT importing any exp script.

Checks
  1. A13 (Caoyang) power profile (GFA-weighted, labeled only) vs results_e2
  2. E1 GroupKFold balanced accuracy, FORM-only, RF(n_estimators=120)  [exp used 400]
  3. Bridge residential_lowrise -> house (count, TRAIN) vs results_e2
  4. LOO R^2 developer_capital, FORM features, Ridge alpha=10 FIXED   [exp used nested-LOO alpha in {0.1,1,10,100}]
  5. FAR of cache_gen/cf_DEV.parquet from geometry vs results_e4
  6. internal-consistency sanity sweeps over results_e0..e4.json

Output: V05/results_audit.json  (own file; never a shared one)
"""
import json
import numpy as np
import pandas as pd

SEED = 0
ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation"
V05 = f"{ROOT}/v05"

out = {"meta": {"script": "v05/audit_v05.py", "seed": SEED,
                "independent": "re-derived from raw inputs; exp scripts NOT imported"}}

df = pd.read_parquet(f"{V05}/dataset.parquet",
                     columns=["bid", "window", "semantic", "family", "height",
                              "fp_area", "floors", "age", "elong", "gfa", "role"])
e0 = json.load(open(f"{V05}/results_e0.json"))
e1 = json.load(open(f"{V05}/results_e1.json"))
e2 = json.load(open(f"{V05}/results_e2.json"))
e3 = json.load(open(f"{V05}/results_e3.json"))
e4 = json.load(open(f"{V05}/results_e4.json"))
priors = json.load(open(f"{V05}/power_priors_v05.json"))
DIMS = priors["dims"]

# ----------------------------------------------------------------- check 1
# A13 Caoyang power profile: GFA-weighted mean of per-building prior weights,
# labeled buildings only. Age regimes per results_e2 meta:
#   pre1985: age<=1985 ; 1985_1997: 1986..1997 ; post1998: age>=1998 ; NaN -> no_age
def prior_w(sem, age):
    p = priors["priors"][sem]
    if p.get("time_conditioned"):
        if pd.isna(age):
            reg = "no_age"
        elif age <= 1985:
            reg = "pre1985"
        elif age <= 1997:
            reg = "1985_1997"
        else:
            reg = "post1998"
        return p["cells"][reg]["w"]
    return p["w"]

def window_profile(wid):
    g = df[(df.window == wid) & (df.semantic != "unlabeled")]
    W = np.array([[prior_w(s, a)[d] for d in DIMS]
                  for s, a in zip(g.semantic, g.age)])
    return {d: float(np.average(W[:, i], weights=g.gfa.values))
            for i, d in enumerate(DIMS)}

a13_mine = window_profile("A13")
a13_e2 = e2["window_profiles"]["A13"]["profile"]
c1 = {d: {"audit": round(a13_mine[d], 4), "e2": a13_e2[d],
          "match_3dp": round(a13_mine[d], 3) == round(a13_e2[d], 3)}
      for d in DIMS}
out["check1_A13_profile"] = c1
print("check1 A13:", c1)

# also recompute ALL 24 profiles -> reuse as independent targets in check 4
all_prof = {wid: window_profile(wid) for wid in sorted(df.window.unique())}
prof_max_absdiff = max(abs(all_prof[w][d] - e2["window_profiles"][w]["profile"][d])
                       for w in all_prof for d in DIMS)
out["check1b_all24_profiles_max_absdiff_vs_e2"] = round(float(prof_max_absdiff), 6)
dom_mine = {w: max(all_prof[w], key=all_prof[w].get) for w in all_prof}
gov_dom = sorted([w for w, d in dom_mine.items() if d == "government"])
dev_dom = sorted([w for w, d in dom_mine.items() if d == "developer_capital"])
out["check1c_dominance"] = {
    "gov_dominant_audit": gov_dom, "n_gov": len(gov_dom),
    "dev_dominant_audit": dev_dom, "n_dev": len(dev_dom),
    "matches_e2": (gov_dom == sorted(e2["findings"]["gov_dominant_windows"]) and
                   dev_dom == sorted(e2["findings"]["dev_dominant_windows"]))}

# ----------------------------------------------------------------- check 2
# E1 GroupKFold(24, by window) balanced accuracy, FORM features only.
# DELIBERATE DIFFERENCE: n_estimators=120 (exp1 used 400) -> expect within ~3pp.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, accuracy_score

lab = df[df.semantic != "unlabeled"].reset_index(drop=True)
X = lab[["fp_area", "height", "floors", "elong"]].values
y = lab["semantic"].values
groups = lab["window"].values
rf = RandomForestClassifier(n_estimators=120, class_weight="balanced",
                            random_state=SEED, n_jobs=-1)
pred = cross_val_predict(rf, X, y, groups=groups, cv=GroupKFold(n_splits=24),
                         n_jobs=1)
bal = balanced_accuracy_score(y, pred)
acc = accuracy_score(y, pred)
out["check2_e1_group24_FORM"] = {
    "audit_balanced_acc_rf120": round(float(bal), 4),
    "e1_balanced_acc_rf400": e1["runs"]["group24_FORM"]["balanced_accuracy"],
    "audit_acc_rf120": round(float(acc), 4),
    "e1_acc_rf400": e1["runs"]["group24_FORM"]["accuracy"],
    "n_labeled_audit": int(len(lab)), "n_labeled_e1": e1["n_buildings_labeled"],
    "within_3pp": abs(bal - e1["runs"]["group24_FORM"]["balanced_accuracy"]) < 0.03,
    "note": "audit uses 120 trees vs exp1's 400 (documented difference)"}
print("check2:", out["check2_e1_group24_FORM"])

# ----------------------------------------------------------------- check 3
tr = lab[lab.role == "train"]
ct = pd.crosstab(tr.semantic, tr.family)
n_house = int(ct.loc["residential_lowrise", "house"])
n_row = int(ct.loc["residential_lowrise"].sum())
mine3 = n_house / n_row
out["check3_bridge_lowrise_house_count_train"] = {
    "audit": round(mine3, 4), "audit_counts": f"{n_house}/{n_row}",
    "e2": e2["bridge_train"]["counts_row_normalized"]["residential_lowrise"]["house"],
    "e2_counts": f'{e2["bridge_train"]["counts"]["residential_lowrise"]["house"]}/'
                 f'{e2["bridge_train"]["row_totals_count"]["residential_lowrise"]}',
    "match": round(mine3, 4) == e2["bridge_train"]["counts_row_normalized"]["residential_lowrise"]["house"]}
print("check3:", out["check3_bridge_lowrise_house_count_train"])

# ----------------------------------------------------------------- check 4
# LOO R^2 for developer_capital from FORM (PC1,PC2 + 10 v03 indicators),
# Ridge alpha=10 FIXED (exp3 used nested-LOO alpha in {0.1,1,10,100}).
# Targets = my OWN recomputed profiles (check 1b), not results_e2.
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sig = json.load(open(f"{ROOT}/v03/signatures.json"))
sig_by_id = {w["id"]: w for w in sig["windows"]}
ids = sorted(sig_by_id.keys())
IND10 = ["FAR", "coverage", "h_p50", "h_p90", "median_fp",
         "sh_house", "sh_slab", "sh_tower", "sh_large", "sh_mid"]
Xf = np.array([list(sig_by_id[w]["pca_xy"]) +
               [sig_by_id[w]["indicators"][k] for k in IND10] for w in ids])
ydev = np.array([all_prof[w]["developer_capital"] for w in ids])
pred_loo = np.zeros(24)
for i in range(24):
    tr_idx = np.array([j for j in range(24) if j != i])
    sc = StandardScaler().fit(Xf[tr_idx])
    m = Ridge(alpha=10.0, random_state=SEED).fit(sc.transform(Xf[tr_idx]), ydev[tr_idx])
    pred_loo[i] = m.predict(sc.transform(Xf[i:i + 1]))[0]
r2 = r2_score(ydev, pred_loo)
e3_dev = e3["loo"]["FORM"]["developer_capital"]
out["check4_loo_r2_dev_form"] = {
    "audit_r2_alpha10_fixed": round(float(r2), 4),
    "e3_r2_nested_alpha": e3_dev["loo_r2"],
    "e3_alphas_actually_used": e3_dev["alphas_used"],
    "same_sign": bool(np.sign(r2) == np.sign(e3_dev["loo_r2"])),
    "note": "audit fixes alpha=10; exp3 selected alpha by nested LOO (documented difference); audit targets are independently recomputed profiles"}
print("check4:", out["check4_loo_r2_dev_form"])

# ----------------------------------------------------------------- check 5
import geopandas as gpd
cf = gpd.read_parquet(f"{V05}/cache_gen/cf_DEV.parquet")
far_geom = float((cf.geometry.area * cf.floors).sum() / 1e6)
far_gfacol = float(cf.gfa.sum() / 1e6)
out["check5_cf_DEV_FAR"] = {
    "audit_FAR_from_geometry": round(far_geom, 4),
    "audit_FAR_from_gfa_col": round(far_gfacol, 4),
    "e4_FAR": e4["per_config"]["DEV"]["indicators"]["FAR"],
    "n_buildings_audit": int(len(cf)),
    "n_buildings_e4": e4["per_config"]["DEV"]["n_buildings"]}
print("check5:", out["check5_cf_DEV_FAR"])

# ----------------------------------------------------------------- check 6 (json sanity)
s = {}
# e0
s["e0_n_rows_eq_dataset"] = (e0["n_rows"] == len(df) == 14527)
s["e0_per_window_n_sum"] = int(sum(e0["per_window_n"].values()))
s["e0_per_window_match_dataset"] = bool(
    all(e0["per_window_n"][w] == int((df.window == w).sum()) for w in e0["per_window_n"]))
s["e0_semantic_counts_sum"] = int(sum(e0["semantic_counts"].values()))
s["e0_family_counts_sum"] = int(sum(e0["family_counts"].values()))
# e1
s["e1_n_labeled_eq_total_minus_unlabeled"] = (
    e1["n_buildings_labeled"] == 14527 - e0["semantic_counts"]["unlabeled"])
s["e1_majority_baseline_recompute"] = round(
    e0["semantic_counts"]["residential_estate"] / e1["n_buildings_labeled"], 4)
s["e1_per_window_n_sum"] = int(sum(v["n"] for v in e1["per_window_group24"].values()))
s["e1_per_window_n_match_labeled"] = bool(
    all(v["n"] == int(((df.window == w) & (df.semantic != "unlabeled")).sum())
        for w, v in e1["per_window_group24"].items()))
# confusion-matrix internal consistency for every run
conf_ok = {}
for run, r in e1["runs"].items():
    cm = pd.DataFrame(r["confusion"]).T  # rows true, cols pred
    cm = cm[list(r["confusion"].keys())]
    row_sums_ok = all(int(cm.loc[c].sum()) == int(r["per_class"][c]["support"])
                      for c in cm.index)
    acc_cm = np.trace(cm.values) / cm.values.sum()
    recalls = [cm.loc[c, c] / cm.loc[c].sum() for c in cm.index]
    conf_ok[run] = {"row_sums_match_support": bool(row_sums_ok),
                    "acc_from_confusion": round(float(acc_cm), 4),
                    "acc_reported": r["accuracy"],
                    "bal_from_confusion": round(float(np.mean(recalls)), 4),
                    "bal_reported": r["balanced_accuracy"],
                    "total_n": int(cm.values.sum())}
s["e1_confusion_consistency"] = conf_ok
# pooled power recall vs confusion (group24 FORM+CTX headline 0.0007)
pw = ["public_admin", "education", "healthcare"]
cm = pd.DataFrame(e1["runs"]["group24_FORM+CTX"]["confusion"]).T
s["e1_power_pooled_recall_g24ctx_recompute"] = round(
    float(sum(cm.loc[c, c] for c in pw) / sum(cm.loc[c].sum() for c in pw)), 4)
s["e1_n_power_buildings_recompute"] = int(sum(cm.loc[c].sum() for c in pw))
# nan-share note: e0 reports dist_water nan on ALL buildings, e1 on labeled subset
s["note_dist_water_nan"] = {"e0_all": e0["nan_share"]["dist_water"],
                            "e1_labeled": e1["nan_handling"]["dist_water"]["nan_share"]}
# e2
s["e2_bridge_row_totals_sum"] = int(sum(e2["bridge_train"]["row_totals_count"].values()))
s["e2_n_train_labeled"] = e2["meta"]["n_train_labeled"]
s["e2_bridge_rows_sum_to_1"] = bool(all(
    abs(sum(v.values()) - 1) < 0.002
    for v in e2["bridge_train"]["counts_row_normalized"].values()))
s["e2_age_regime_shares_sum_to_1"] = bool(all(
    abs(sum(p["res_estate_age_regime_gfa_share"].values()) - 1) < 0.002
    for p in e2["window_profiles"].values()
    if p["res_estate_age_regime_gfa_share"]))
s["e2_dominant_dim_consistent"] = bool(all(
    max(p["profile"], key=p["profile"].get) == p["dominant_dim"]
    for p in e2["window_profiles"].values()))
s["e2_n_dom"] = {"gov": len(e2["findings"]["gov_dominant_windows"]),
                 "dev": len(e2["findings"]["dev_dominant_windows"]),
                 "other": len(e2["findings"]["other_dominant_windows"])}
# e3 screen count + BH survivors
n_tests = sum(len(v) for v in e3["correlation_screen"].values())
n_bh = sum(1 for dim in e3["correlation_screen"].values()
           for v in dim.values() if v["bh_fdr10"])
s["e3_n_tests"] = n_tests
s["e3_bh_survivors"] = n_bh
# e4
s["e4_real_FAR_recompute"] = round(e4["meta"]["site_real_total_gfa_m2"] / 1e6, 4)
s["e4_token_cells_sum_293"] = {k: int(sum(v["token_cells_T1_T8"]))
                               for k, v in e4["per_config"].items()}
s["e4_semantic_mix_sums"] = {k: round(sum(v.values()), 4)
                             for k, v in e4["semantic_mixes"].items()}
out["check6_sanity"] = s

# headline dataset claims from the build agent
full = pd.read_parquet(f"{V05}/dataset.parquet",
                       columns=["dist_water", "dist_green", "age", "dist_road_all",
                                "dist_road_major", "density_100m", "semantic",
                                "gfa", "window"])
claims = {
    "nan_share_dist_water": round(float(full.dist_water.isna().mean()), 4),
    "nan_share_dist_green": round(float(full.dist_green.isna().mean()), 4),
    "nan_share_age": round(float(full.age.isna().mean()), 4),
    "median_dist_road_all_m": round(float(full.dist_road_all.median()), 1),
    "median_dist_road_major_m": round(float(full.dist_road_major.median()), 1),
    "median_density_100m": float(full.density_100m.median()),
    "share_dated_le_1985": round(float((full.age.dropna() <= 1985).mean()), 4),
    "res_estate_no_age_gfa_share_pooled": round(float(
        full[(full.semantic == "residential_estate") & (full.age.isna())].gfa.sum() /
        full[full.semantic == "residential_estate"].gfa.sum()), 4),
    "A47_unlabeled_gfa_share": round(float(
        full[(full.window == "A47") & (full.semantic == "unlabeled")].gfa.sum() /
        full[full.window == "A47"].gfa.sum()), 4),
}
out["check7_dataset_claims_recompute"] = claims
print("claims:", claims)

def _np(o):
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(type(o))

print("check6 sanity:", json.dumps(s, indent=1, default=_np))
with open(f"{V05}/results_audit.json", "w") as f:
    json.dump(out, f, indent=1, default=_np)
print("\nwritten results_audit.json")
