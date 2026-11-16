#!/usr/bin/env python3
"""D1: comparative doctrine chart + results_d1.json, drawn ONLY from doctrine_recipes.json."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

V06 = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation/v06"
R = json.load(open(f"{V06}/doctrine_recipes.json"))["doctrines"]
np.random.seed(0)

ORDER = ["garden_city", "ville_radieuse", "neighbourhood_unit", "microrayon", "gb50180"]
LABEL = {"garden_city": "Garden City\n(Howard 1898)", "ville_radieuse": "Ville Radieuse\n(Corbusier 1933)",
         "neighbourhood_unit": "Neighbourhood Unit\n(Perry 1929)", "microrayon": "Microrayon\n(SNiP era)",
         "gb50180": "GB 50180-2018\n(life circles)"}
DENOM = {"garden_city": "town 405 ha", "ville_radieuse": "housing sector",
         "neighbourhood_unit": "unit 64.7 ha", "microrayon": "microdistrict 10-60 ha",
         "gb50180": "block 2-4 ha"}

def mid(q):
    if q.get("doctrine_silent"): return None
    if "value" in q: return q["value"] if isinstance(q["value"], (int, float)) else None
    if "range" in q: return 0.5 * (q["range"][0] + q["range"][1])
    return None

def rng(q):
    if q.get("doctrine_silent"): return None
    if "range" in q: return q["range"]
    if "value" in q and isinstance(q["value"], (int, float)): return [q["value"], q["value"]]
    return None

# ---------- figure ----------
fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), height_ratios=[1.25, 1.0, 0.95])
fig.subplots_adjust(hspace=0.52, top=0.92, bottom=0.06, left=0.125, right=0.95)
x = np.arange(len(ORDER))

# Panel A: land budget stacked bars
ax = axes[0]
CAT = [("residential", "#c08552", "Residential / built"), ("green_open", "#5a8f5a", "Green / open"),
       ("civic_public", "#7a6fb0", "Civic / public")]
for i, d in enumerate(ORDER):
    lb = R[d]["quantities"]["land_budget"]
    bottom = 0.0
    for key, col, _ in CAT:
        v = mid(lb.get(key, {"doctrine_silent": True}))
        if v:
            ax.bar(i, v, 0.62, bottom=bottom, color=col, edgecolor="white", lw=0.8)
            ax.text(i, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
            bottom += v
    rem = mid(lb.get("remainder_unspecified", {"doctrine_silent": True})) or 0.0
    if rem > 0.005:
        ax.bar(i, rem, 0.62, bottom=bottom, color="#cccccc", edgecolor="white", lw=0.8, hatch="//")
        ax.text(i, bottom + rem / 2, "unspecified\nby doctrine", ha="center", va="center",
                fontsize=7.5, color="#444444")
    ax.text(i, -0.105, DENOM[d], ha="center", va="top", fontsize=8, style="italic", color="#555555")
ax.set_xticks(x); ax.set_xticklabels([""] * len(ORDER))
ax.set_ylim(0, 1.02); ax.set_ylabel("Land-budget fraction")
ax.set_title("A.  Doctrinal land budgets (stated fractions only)",
             loc="left", fontsize=11, fontweight="bold")
hnd = [plt.Rectangle((0, 0), 1, 1, fc=c) for _, c, _ in CAT] + \
      [plt.Rectangle((0, 0), 1, 1, fc="#cccccc", hatch="//")]
ax.legend(hnd, [l for _, _, l in CAT] + ["Unspecified"], ncol=4, fontsize=8.5, frameon=False,
          loc="lower right", bbox_to_anchor=(1.0, 1.01))
for s in ["top", "right"]: ax.spines[s].set_visible(False)

# Panel B: density + height
ax = axes[1]
DKEY = {"garden_city": ("density_gross_town", "gross town"), "ville_radieuse": ("density_net_residential", "net resid."),
        "neighbourhood_unit": ("density_net", "gross unit"), "microrayon": ("density_gross", "gross"),
        "gb50180": ("density_block", "block")}
DV = {"garden_city": [74, 74], "neighbourhood_unit": [77, 93]}  # persons/ha derived in JSON text
for i, d in enumerate(ORDER):
    q = R[d]["quantities"][DKEY[d][0]]
    r = DV.get(d) or rng(q)
    ax.plot([i - 0.13] * 2, r, color="#2b6cb0", lw=3, solid_capstyle="round")
    ax.scatter([i - 0.13], [np.mean(r)], s=60, color="#2b6cb0", zorder=3)
    ax.text(i - 0.13, r[1] * 1.25, f"{int(r[0])}-{int(r[1])}" if r[0] != r[1] else f"{int(r[0])}",
            ha="center", fontsize=8, color="#2b6cb0")
ax.set_yscale("log"); ax.set_ylim(8, 4000)
ax.set_ylabel("Density (persons/ha)", color="#2b6cb0")
ax.tick_params(axis="y", labelcolor="#2b6cb0")
ax2 = ax.twinx()
FL = {"garden_city": None, "ville_radieuse": [12, 60], "neighbourhood_unit": [1, 2.5],
      "microrayon": [5, 16], "gb50180": [1, 26]}
IMPLIED = {"neighbourhood_unit"}
for i, d in enumerate(ORDER):
    r = FL[d]
    if r is None:
        ax2.text(i + 0.13, 1.25, "height\nn.s.", ha="center", fontsize=8, color="#b03a2e", style="italic")
        continue
    fill = "none" if d in IMPLIED else "#b03a2e"
    ax2.plot([i + 0.13] * 2, r, color="#b03a2e", lw=3, alpha=0.85, solid_capstyle="round",
             linestyle=":" if d in IMPLIED else "-")
    ax2.scatter([i + 0.13], [np.mean(r)], s=64, marker="^", facecolors=fill, edgecolors="#b03a2e", zorder=3)
    lab = f"{r[0]:g}-{r[1]:g}F" + (" (implied)" if d in IMPLIED else "")
    ax2.text(i + 0.13, r[1] * 1.3, lab, ha="center", fontsize=8, color="#b03a2e")
ax2.set_yscale("log"); ax2.set_ylim(0.7, 150)
ax2.set_ylabel("Storeys (floors)", color="#b03a2e")
ax2.tick_params(axis="y", labelcolor="#b03a2e")
ax.set_xticks(x); ax.set_xticklabels([""] * len(ORDER)); ax.set_xlim(-0.6, 4.6)
ax.set_title("B.  Density (blue, left, log) and height regime (red, right, log); n.s. = doctrine silent",
             loc="left", fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)

# Panel C: stakeholder a-priori dot matrix
ax = axes[2]
DIMS = ["government", "developer_capital", "resident_community", "informal"]
DLAB = ["Government", "Developer capital", "Resident community", "Informal"]
for j, dim in enumerate(DIMS):
    for i, d in enumerate(ORDER):
        v = R[d]["stakeholder_apriori"]["config"][dim]
        ax.scatter(i, j, s=80 + 2200 * v, color=plt.cm.viridis(0.15 + 0.75 * v),
                   edgecolors="#333333", lw=0.7, zorder=3)
        ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.45 else "#222222", zorder=4, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([LABEL[d] for d in ORDER], fontsize=9.5)
ax.set_yticks(range(4)); ax.set_yticklabels(DLAB, fontsize=9.5)
ax.set_xlim(-0.6, 4.6); ax.set_ylim(-0.7, 3.7); ax.invert_yaxis()
ax.set_title("C.  Stakeholder a-priori: WHO THE DOCTRINE EMPOWERS (asserted, additive dims in [0,1])",
             loc="left", fontsize=11, fontweight="bold")
for s in ["top", "right", "left", "bottom"]: ax.spines[s].set_visible(False)
ax.grid(True, axis="x", alpha=0.18)

fig.suptitle("D1 — Five classical planning doctrines as machine-readable recipes (every value sourced; silences marked)",
             fontsize=12.5, fontweight="bold")
out = f"{V06}/fig/fig_d1_doctrines.png"
fig.savefig(out, dpi=140)
print("wrote", out)

# ---------- results_d1.json ----------
conf = {"high": 0, "medium": 0, "low": 0}
silent = {}
def walk(node, dname):
    if isinstance(node, dict):
        if node.get("doctrine_silent") is True:
            silent[dname] = silent.get(dname, 0) + 1
        c = node.get("confidence")
        if c in conf and "config" not in node:  # quantity-level confidence only
            conf[c] += 1
        for v in node.values(): walk(v, dname)
    elif isinstance(node, list):
        for v in node: walk(v, dname)

per_doc_conf = {}
for d in ORDER:
    snap = dict(conf)
    walk(R[d]["quantities"], d)
    per_doc_conf[d] = {k: conf[k] - snap[k] for k in conf}

results = {
    "task": "D1 doctrine distillation",
    "n_doctrines": 5,
    "confidence_counts_quantities_total": conf,
    "confidence_counts_per_doctrine": per_doc_conf,
    "doctrine_silent_counts": {d: silent.get(d, 0) for d in ORDER},
    "explicit_silence_lists": {d: len(R[d]["silences"]) for d in ORDER},
    "stakeholder_apriori": {d: R[d]["stakeholder_apriori"]["config"] for d in ORDER},
    "stakeholder_confidence": {d: R[d]["stakeholder_apriori"]["confidence"] for d in ORDER},
    "caoyang_anchor": {"neighbourhood_unit": "verified high", "microrayon": "lineage note, medium"},
    "artifacts": [f"{V06}/doctrine_recipes.json", f"{V06}/doctrine_sources.md",
                  f"{V06}/fig/fig_d1_doctrines.png"],
}
json.dump(results, open(f"{V06}/results_d1.json", "w"), indent=2, ensure_ascii=False)
print("wrote results_d1.json")
print(json.dumps(results["confidence_counts_quantities_total"]), json.dumps(results["doctrine_silent_counts"]))
