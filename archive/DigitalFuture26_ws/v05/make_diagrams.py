#!/usr/bin/env python3
"""
v05 methodology diagrams (matplotlib only, English labels).

Produces three publication-style diagrams into V05/fig/:
  fig_d1_triangle.png  - research design triangle (CLASSIFICATION / POWER / FORM, E1-E4)
  fig_d2_pipeline.png  - data-to-label fusion pipeline + parallel geometry track
  fig_d3_twolayer.png  - two-layer type system (semantic x morphological family)

All numbers shown are quoted from upstream v03/v04/probe results; nothing is computed here.
Rerunnable; seed-free (pure drawing). dpi=140.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

V05 = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation/v05"
FIG = os.path.join(V05, "fig")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- palette
C = dict(
    cls_fill="#dce8f5", cls_edge="#33618f",   # classification = blue
    pow_fill="#f6e1e1", pow_edge="#9c3a3a",   # power = crimson
    form_fill="#e0eee2", form_edge="#3c7a4d", # form = green
    neut_fill="#f2f2f2", neut_edge="#555555", # neutral grey
    fuse_fill="#fdf2dd", fuse_edge="#a87f2f", # fusion cascade = amber
    warn_fill="#fdecea", warn_edge="#b03a2e", # warning red
    arrow="#4a4a4a", text="#222222",
)

# ---------------------------------------------------------------- helpers
def draw_box(ax, cx, cy, w, h, fc, ec, lw=1.5, rounding=0.12, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        fc=fc, ec=ec, lw=lw, zorder=zorder))

def box_text(ax, cx, top_y, lines, ha="center"):
    """lines: list of dicts {text, size, weight, color, style, dy} stacked downward."""
    y = top_y
    for ln in lines:
        ax.text(cx, y, ln["text"], ha=ha, va="top",
                fontsize=ln.get("size", 8.5),
                fontweight=ln.get("weight", "normal"),
                color=ln.get("color", C["text"]),
                style=ln.get("style", "normal"), zorder=4)
        y -= ln.get("dy", 0.34)

def arrow(ax, p0, p1, color=None, style="-|>", rad=0.0, lw=1.6, ls="-", ms=15,
          shrinkA=3, shrinkB=3, zorder=1):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        connectionstyle=f"arc3,rad={rad}",
                        color=color or C["arrow"], lw=lw, linestyle=ls,
                        shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder)
    ax.add_patch(a)

def elabel(ax, x, y, text, color="#333333", size=8.5, ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, style=style,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92),
            zorder=5)

def new_ax(figsize, xmax, ymax):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.set_aspect("auto"); ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


# ================================================================ D1: triangle
def fig_d1():
    fig, ax = new_ax((9.8, 7.3), 10, 10)

    ax.text(5, 9.75, "Research design: CLASSIFICATION – POWER – FORM",
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=C["text"])
    ax.text(5, 9.22, "“A building’s power semantics ride on its metadata type, "
                     "not its shape alone” (A/B/C 2026-06-11)",
            ha="center", va="top", fontsize=9.5, style="italic", color="#555555")

    # --- nodes
    # CLASSIFICATION (top)
    draw_box(ax, 5.0, 7.30, 4.6, 1.45, C["cls_fill"], C["cls_edge"], lw=1.8)
    box_text(ax, 5.0, 7.78, [
        dict(text="CLASSIFICATION", size=12, weight="bold", color=C["cls_edge"], dy=0.47),
        dict(text="semantic type · 8 classes", size=9),
    ])
    # FORM (bottom left)
    draw_box(ax, 2.15, 2.60, 3.95, 1.95, C["form_fill"], C["form_edge"], lw=1.8)
    box_text(ax, 2.15, 3.38, [
        dict(text="FORM", size=12, weight="bold", color=C["form_edge"], dy=0.45),
        dict(text="morphological family", size=8.7, dy=0.33),
        dict(text="+ 50 m token grammar", size=8.7, dy=0.33),
        dict(text="+ window signature", size=8.7),
    ])
    # POWER (bottom right)
    draw_box(ax, 7.85, 2.60, 3.95, 1.95, C["pow_fill"], C["pow_edge"], lw=1.8)
    box_text(ax, 7.85, 3.38, [
        dict(text="POWER", size=12, weight="bold", color=C["pow_edge"], dy=0.45),
        dict(text="4 additive actor weights:", size=8.7, dy=0.33),
        dict(text="government / developer capital", size=8.7, dy=0.33),
        dict(text="resident community / informal", size=8.7),
    ])

    # --- edges
    # E1: form -> classification
    arrow(ax, (2.85, 3.60), (3.85, 6.55), lw=1.8)
    elabel(ax, 1.70, 5.25, "E1: form → classification\n(recognition test)", size=8.8)
    # E2: classification -> power
    arrow(ax, (6.15, 6.55), (7.15, 3.60), lw=1.8)
    elabel(ax, 8.42, 5.25, "E2: classification → power\n(literature priors,\nage-conditioned)", size=8.8)
    # E3: form <-> power (upper of the two bottom edges)
    arrow(ax, (4.15, 3.02), (5.85, 3.02), style="<|-|>", lw=1.8)
    elabel(ax, 5.0, 3.93, "E3: form ↔ power\n(window-level triangle test)", size=8.4)
    # E4: power -> form (lower, straight, parallel to E3)
    arrow(ax, (5.85, 2.18), (4.15, 2.18), lw=1.8)
    elabel(ax, 5.0, 1.38, "E4: power → form\n(counterfactual generation)", size=8.4)

    out = os.path.join(FIG, "fig_d1_triangle.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


# ================================================================ D2: pipeline
def fig_d2():
    fig, ax = new_ax((12.6, 8.3), 14, 10)

    ax.text(7, 9.78, "Data-to-label pipeline: source layers → fusion cascade → semantic classes",
            ha="center", va="top", fontsize=13, fontweight="bold", color=C["text"])

    # --- left column: source layers
    ax.text(2.3, 9.18, "Source layers", ha="center", va="top",
            fontsize=10, fontweight="bold", color=C["cls_edge"])
    ax.text(2.3, 8.84, "(per-building joinability, 3-patch audit)", ha="center", va="top",
            fontsize=8, style="italic", color="#666666")

    sources = [
        ("EULUC land-use parcels", "86–94%", 8.10),
        ("AOI compounds", "23–78%", 7.08),
        ("OSM building type", "10–19%", 6.06),
        ("Function + Age layer", "22–45%", 5.04),
        ("POI 2024 within 50 m", "79–99%", 4.02),
    ]
    for name, pct, y in sources:
        draw_box(ax, 2.3, y, 3.9, 0.86, C["cls_fill"], C["cls_edge"], lw=1.3, rounding=0.09)
        box_text(ax, 2.3, y + 0.30, [
            dict(text=name, size=9, weight="bold", dy=0.36),
            dict(text="joinability " + pct, size=8.2, color="#444444"),
        ])

    # --- fusion cascade
    fx, fy, fw, fh = 7.3, 6.06, 4.5, 3.55
    draw_box(ax, fx, fy, fw, fh, C["fuse_fill"], C["fuse_edge"], lw=1.8)
    box_text(ax, fx, fy + fh / 2 - 0.22, [
        dict(text="Fusion cascade — priority order", size=10.5, weight="bold",
             color="#7a5a1e", dy=0.62),
    ])
    steps = [
        "1   EULUC parcel prior",
        "2   AOI override",
        "3   OSM-type / Function override",
        "4   POI 50 m majority (≥2)",
        "5   morphology fallback",
    ]
    yy = fy + fh / 2 - 0.95
    for s in steps:
        ax.text(fx - fw / 2 + 0.45, yy, s, ha="left", va="top", fontsize=9,
                color=C["text"], zorder=4)
        yy -= 0.52

    # arrows: sources -> fusion (fan-in)
    tgt_ys = [7.45, 6.78, 6.06, 5.34, 4.67]
    for (name, pct, y), ty in zip(sources, tgt_ys):
        arrow(ax, (4.27, y), (fx - fw / 2, ty), lw=1.4, ms=12)

    # --- output box
    ox, oy, ow, oh = 11.9, 6.06, 3.7, 2.0
    draw_box(ax, ox, oy, ow, oh, C["cls_fill"], C["cls_edge"], lw=1.8)
    box_text(ax, ox, oy + oh / 2 - 0.18, [
        dict(text="Output", size=10, weight="bold", color=C["cls_edge"], dy=0.42),
        dict(text="8 semantic classes + unlabeled", size=8.8, dy=0.36),
        dict(text="fused coverage 31–99.8%", size=8.4, dy=0.32),
        dict(text="across 24 windows (median ~96%)", size=8.4),
    ])
    arrow(ax, (fx + fw / 2, fy), (ox - ow / 2, oy), lw=1.8)

    # --- geometry track (parallel, below)
    ax.text(0.35, 3.12, "Geometry track (morphology)", ha="left", va="top",
            fontsize=10, fontweight="bold", color=C["form_edge"])
    g_y = 2.25
    draw_box(ax, 2.3, g_y, 3.9, 0.95, C["form_fill"], C["form_edge"], lw=1.3, rounding=0.09)
    box_text(ax, 2.3, g_y + 0.20, [dict(text="AI footprints + heights", size=9, weight="bold")])
    draw_box(ax, 7.3, g_y, 4.5, 0.95, C["form_fill"], C["form_edge"], lw=1.3, rounding=0.09)
    box_text(ax, 7.3, g_y + 0.18, [
        dict(text="fp_area · height · floors = h/3.5 · elongation", size=8.6),
    ])
    draw_box(ax, 11.9, g_y, 3.7, 1.15, C["form_fill"], C["form_edge"], lw=1.3, rounding=0.09)
    box_text(ax, 11.9, g_y + 0.44, [
        dict(text="family rules", size=9, weight="bold", dy=0.36),
        dict(text="house / slab / tower /", size=8.2, dy=0.30),
        dict(text="large_footprint / midrise_other", size=8.2),
    ])
    arrow(ax, (4.27, g_y), (5.03, g_y), lw=1.5, color=C["form_edge"])
    arrow(ax, (9.57, g_y), (10.03, g_y), lw=1.5, color=C["form_edge"])
    # dashed link: family rules feed cascade step 5
    arrow(ax, (11.0, g_y + 0.60), (8.9, fy - fh / 2), lw=1.3, ls=(0, (4, 3)),
          color=C["form_edge"], rad=0.18, ms=11)
    elabel(ax, 10.55, 3.65, "feeds step 5", size=7.8, color=C["form_edge"], style="italic")

    # --- red side note
    draw_box(ax, 7.0, 0.78, 11.4, 0.85, C["warn_fill"], C["warn_edge"], lw=1.8)
    ax.text(7.0, 0.78, "No cadastre / ownership layer exists  →  power is NEVER read "
                       "from form; documentary path only",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=C["warn_edge"], zorder=4)

    out = os.path.join(FIG, "fig_d2_pipeline.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


# ================================================================ D3: two layers
def fig_d3():
    fig, ax = new_ax((9.8, 7.4), 10, 10)

    ax.text(5, 9.75, "Two-layer type system: semantic × morphological",
            ha="center", va="top", fontsize=13.5, fontweight="bold", color=C["text"])

    bx = 3.8   # center x of the layer boxes

    # --- upper layer: SEMANTIC
    sy, sh = 7.45, 2.55
    draw_box(ax, bx, sy, 6.6, sh, C["cls_fill"], C["cls_edge"], lw=1.8)
    box_text(ax, bx, sy + sh / 2 - 0.20, [
        dict(text="SEMANTIC layer — 8 classes", size=11, weight="bold",
             color=C["cls_edge"], dy=0.50),
        dict(text="residential_estate · residential_lowrise · commercial_retail · office_finance",
             size=8.3, dy=0.34),
        dict(text="industrial · public_admin · education · healthcare",
             size=8.3, dy=0.50),
        dict(text="power priors P(w | type, age regime) attach HERE,",
             size=8.5, style="italic", color=C["pow_edge"], dy=0.33),
        dict(text="with provenance + confidence",
             size=8.5, style="italic", color=C["pow_edge"]),
    ])

    # --- bridge
    by, bh = 4.85, 1.05
    draw_box(ax, bx, by, 4.8, bh, C["neut_fill"], C["neut_edge"], lw=1.5)
    box_text(ax, bx, by + bh / 2 - 0.16, [
        dict(text="co-occurrence counts", size=9.5, weight="bold", dy=0.38),
        dict(text="(learned from 18 TRAIN windows)", size=8.3, color="#555555"),
    ])
    arrow(ax, (bx, sy - sh / 2), (bx, by + bh / 2), style="<|-|>", lw=1.8)
    arrow(ax, (bx, by - bh / 2), (bx, 3.42), style="<|-|>", lw=1.8)

    # --- lower layer: MORPHOLOGICAL FAMILY
    my, mh = 2.35, 2.15
    draw_box(ax, bx, my, 6.6, mh, C["form_fill"], C["form_edge"], lw=1.8)
    box_text(ax, bx, my + mh / 2 - 0.20, [
        dict(text="MORPHOLOGICAL FAMILY layer — 5 families", size=11, weight="bold",
             color=C["form_edge"], dy=0.50),
        dict(text="house · slab · tower · large_footprint · midrise_other",
             size=8.5, dy=0.50),
        dict(text="generation instantiates HERE from real-cell pools",
             size=8.5, style="italic", color=C["form_edge"]),
    ])

    # --- right margin: age-regime axis
    axx = 9.05
    arrow(ax, (axx, 1.45), (axx, 8.85), style="-|>", lw=1.5, color="#666666",
          shrinkA=0, shrinkB=0, ms=12)
    ax.text(9.62, 5.15, "age regime", ha="center", va="center", fontsize=9.5,
            rotation=90, color="#555555", fontweight="bold")
    ticks = [
        (2.45, "≤1985 danwei"),
        (5.15, "1986–97 dual track"),
        (7.85, "≥1998 commodity"),
    ]
    for ty, lab in ticks:
        ax.plot([axx - 0.09, axx + 0.09], [ty, ty], color="#666666", lw=1.3, zorder=3)
        ax.text(axx - 0.20, ty, lab, ha="right", va="center", fontsize=8.2,
                color="#555555", zorder=4)

    out = os.path.join(FIG, "fig_d3_twolayer.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    return out


if __name__ == "__main__":
    outs = [fig_d1(), fig_d2(), fig_d3()]
    res = {
        "task": "v05 methodology diagrams",
        "figures": outs,
        "note": "pure drawing; all quoted numbers come from v03/v04/probe upstream results, "
                "none computed here",
    }
    with open(os.path.join(V05, "results_diagrams.json"), "w") as f:
        json.dump(res, f, indent=2)
    for o in outs:
        print("wrote", o)
