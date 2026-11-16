#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 — translation layer: doctrine recipes -> Shanghai token vocabulary + programmatic skeletons.

Inputs : V06/doctrine_recipes.json (D1), TF/v03/tokens.json (measured token stats).
Outputs: V06/translation.json, V06/skeleton_<paradigm>.npy (20x20 int grids),
         V06/fig/fig_d2_translation.png, V06/results_d2.json.

Skeleton vocabulary: 0=E_VACANT (build-allowed), 9=E_ROAD, 10=E_WATER (unused, flat site),
11=E_GREEN, -1=RESERVED-CIVIC (built cell whose semantic is demanded civic/education).
Grid: 20x20 cells of 50 m = 1 km x 1 km window (400 cells). Seed 0; fully deterministic.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

np.random.seed(0)

ROOT = "/Users/jinbao/Projects/Urban-Stakeholder-Workshop"
TF   = os.path.join(ROOT, "typology_foundation")
V06  = os.path.join(TF, "v06")
FIG  = os.path.join(V06, "fig")
os.makedirs(FIG, exist_ok=True)

E_VAC, E_ROAD, E_WATER, E_GREEN, CIVIC = 0, 9, 10, 11, -1
N = 20                       # 20 x 20 cells of 50 m
CELL_M2 = 2500.0
CTR = (N - 1) / 2.0          # 9.5 : window centre in cell coords

recipes = json.load(open(os.path.join(V06, "doctrine_recipes.json")))["doctrines"]
tok     = json.load(open(os.path.join(TF, "v03", "tokens.json")))["stats"]
GFA = {t: tok[str(t)]["gfa"]      for t in range(1, 9)}   # mean GFA per 50m cell, m2
COV = {t: tok[str(t)]["coverage"] for t in range(1, 9)}   # mean cell ground coverage
HGT = {t: tok[str(t)]["mean_h"]   for t in range(1, 9)}   # mean height, m

SEM8 = ["residential_estate", "residential_lowrise", "commercial_retail",
        "office_finance", "industrial", "public_admin", "education", "healthcare"]


def dist_grid():
    rr, cc = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    return np.sqrt((rr - CTR) ** 2 + (cc - CTR) ** 2)


def counts(grid):
    return {"road": int((grid == E_ROAD).sum()), "green": int((grid == E_GREEN).sum()),
            "civic": int((grid == CIVIC).sum()), "vacant_build": int((grid == E_VAC).sum()),
            "water": int((grid == E_WATER).sum())}


def window_checks(budget, persons_assumption=None, res_tokens=None):
    """Window-level indicators implied by a token budget (counts over 400 cells).
    Implied persons use RESIDENTIAL GFA only (res_tokens = {token:int residential cells})."""
    built = {t: budget.get(f"T{t}", 0) for t in range(1, 9)}
    gfa = sum(n * GFA[t] for t, n in built.items())
    cov = sum(n * COV[t] for t, n in built.items()) / (N * N)
    far = gfa / (N * N * CELL_M2)
    out = {"window_FAR": round(far, 3), "window_ground_coverage": round(cov, 3),
           "built_cells": int(sum(built.values())),
           "green_frac": round(budget.get("E_GREEN", 0) / 400, 3),
           "road_frac": round(budget.get("E_ROAD", 0) / 400, 3),
           "vacant_frac": round(budget.get("E_VACANT", 0) / 400, 3)}
    if persons_assumption and res_tokens:
        m2pp, label = persons_assumption
        res_gfa = sum(n * GFA[t] for t, n in res_tokens.items())
        out["residential_GFA_m2"] = int(round(res_gfa))
        out["implied_persons"] = int(round(res_gfa / m2pp))
        out["implied_persons_per_ha"] = round(res_gfa / m2pp / 100.0, 1)
        out["persons_assumption"] = label
    return out


# ----------------------------------------------------------------- garden city
def build_garden_city():
    g = np.zeros((N, N), dtype=np.int16)
    d = dist_grid()
    notes = []
    # 1. perimeter circular railway + ring road -> border E_ROAD (Howard Diagram 2)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = E_ROAD
    # 2. industry strip: ring just inside the railway (factories front the circular rail)
    ring1 = np.zeros((N, N), bool)
    ring1[1, 1:-1] = ring1[-2, 1:-1] = ring1[1:-1, 1] = ring1[1:-1, -2] = True
    # 3. Central Park disk (145/1000 acres = 14.5% of town) at half-scale window
    park = d < 4.5
    # 4. civic cluster (town hall, library, museum, hospital) inside Central Park
    civ = np.zeros((N, N), bool)
    civ[9:11, 9:11] = True
    # 5. Crystal Palace arcade ring at park edge: 12 commercial cells picked by angle
    ring_cp = (d >= 4.5) & (d < 5.5)
    # 6. Grand Avenue green annulus, 420 ft = 128 m wide; half-scale -> 1.5 cells
    grand = (d >= 6.25) & (d < 7.75)
    # 7. radial boulevards: Howard's six 120-ft boulevards -> 4 orthogonal radials
    rad = np.zeros((N, N), bool)
    rad[9, 1:-1] = True
    rad[1:-1, 9] = True
    rad &= (d >= 4.5)          # boulevards stop at the park edge (pedestrian centre)
    # paint in priority order: border > radial > industry > park/civic > grand avenue
    g[grand] = E_GREEN
    g[park] = E_GREEN
    g[rad] = E_ROAD
    g[civ] = CIVIC
    # schools live inside Grand Avenue (Howard: 'in this splendid avenue ... schools')
    grand_free = [(r, c) for r in range(N) for c in range(N)
                  if grand[r, c] and g[r, c] == E_GREEN]
    ang = {p: np.arctan2(p[0] - CTR, p[1] - CTR) for p in grand_free}
    grand_sorted = sorted(grand_free, key=lambda p: ang[p])
    school_cells = grand_sorted[::max(1, len(grand_sorted) // 6)][:6]
    for p in school_cells:
        g[p] = CIVIC
    # Crystal Palace commercial cells: 12 by angle on the park-edge ring
    cp_free = [(r, c) for r in range(N) for c in range(N)
               if ring_cp[r, c] and g[r, c] == E_VAC]
    cp_sorted = sorted(cp_free, key=lambda p: np.arctan2(p[0] - CTR, p[1] - CTR))
    arcade = cp_sorted[::max(1, len(cp_sorted) // 12)][:12]
    # industry ring cells (still E_VAC after roads/park): alternate shed / yard,
    # alternation by angle around the centre so sheds spread evenly along the rail
    ind_cells = sorted([(r, c) for r in range(N) for c in range(N)
                        if ring1[r, c] and g[r, c] == E_VAC],
                       key=lambda p: np.arctan2(p[0] - CTR, p[1] - CTR))
    ind_built = ind_cells[::2]                      # T7 sheds
    ind_yard = [p for p in ind_cells if p not in ind_built]   # E_VACANT yards
    cn = counts(g)
    n_res = 400 - cn["road"] - cn["green"] - cn["civic"] - len(arcade) - len(ind_cells)
    budget = {"T1": n_res, "T2": cn["civic"], "T7": len(arcade) + len(ind_built),
              "E_VACANT": len(ind_yard), "E_ROAD": cn["road"], "E_GREEN": cn["green"]}
    civic_sem = {str(p): s for p, s in zip(school_cells, ["education"] * 6)}
    civic_sem.update({"(9, 9)": "public_admin", "(9, 10)": "public_admin",
                      "(10, 9)": "healthcare", "(10, 10)": "public_admin"})
    sem_cells = {"commercial_retail": [list(p) for p in arcade],
                 "industrial": [list(p) for p in ind_built]}
    geom = [
        {"name": "perimeter rail+ring road", "value": "1-cell border (76 cells)",
         "source": "Howard 1902 Diagram 2: circular railway around the town"},
        {"name": "industry strip", "value": "1-cell ring inside rail (alternating shed/yard)",
         "source": "Howard 1902 ch.1: factories/warehouses front the circular railway; acreage doctrine-silent"},
        {"name": "Central Park disk", "value": "radius 4.5 cells (=225 m, half-scale)",
         "source": "Central Park 145 acres = 14.5% of town; window compresses Howard's 2.27-km town to 1 km"},
        {"name": "Grand Avenue annulus", "value": "r in [6.25,7.75) cells, 1.5 cells wide",
         "source": "Grand Avenue 420 ft = 128 m wide = 2.56 cells; halved at window half-scale"},
        {"name": "radial boulevards", "value": "4 orthogonal 1-cell radials (Howard: 6 x 120 ft)",
         "source": "Howard 1902 Diagram 2; 6->4 adaptation to the square grid; 120 ft = 37 m < 50-m cell"},
        {"name": "civic cluster", "value": "2x2 cells (1 ha) at centre",
         "source": "Howard 1902 ch.1: town hall/library/museum/hospital in Central Park; site sizes silent"},
        {"name": "schools", "value": "6 cells spaced by angle inside Grand Avenue",
         "source": "Howard 1902 ch.1: schools sited in Grand Avenue, one cluster per ward (6 wards)"},
    ]
    notes.append("Window = half-scale town (1 km vs Howard's ~2.27 km diameter); "
                 "ring radii halved, area FRACTIONS conserved (green 26% target).")
    notes.append("Agricultural belt (5/6 of the estate) is entirely out of frame.")
    return g, budget, civic_sem, sem_cells, geom, notes, {1: n_res}


# --------------------------------------------------------------- ville radieuse
def build_ville_radieuse():
    g = np.full((N, N), E_GREEN, dtype=np.int16)
    notes = []
    # 400-m orthogonal arterial grid; offset so window edges are mid-block
    for k in (4, 12):
        g[k, :] = E_ROAD
        g[:, k] = E_ROAD
    # tower sockets: 100-m lattice of build-allowed cells in the green carpet
    sockets = [(r, c) for r in range(1, N, 2) for c in range(1, N, 2)
               if g[r, c] == E_GREEN]
    for p in sockets:
        g[p] = E_VAC
    cn = counts(g)
    n6 = cn["vacant_build"]                # all sockets get T6 towers
    budget = {"T6": n6, "E_ROAD": cn["road"], "E_GREEN": cn["green"], "E_VACANT": 0}
    geom = [
        {"name": "arterial grid", "value": "1-cell E_ROAD rows/cols at 8-cell (400 m) spacing, "
         "offset to rows/cols {4,12} so window edges fall mid-module",
         "source": "Ville Contemporaine 1922 400-m grid retained in VR (doctrine_recipes block_grain)"},
        {"name": "tower sockets", "value": f"{n6} cells on a 2-cell (100 m) lattice in the green carpet",
         "source": "count solves LC's own numbers: 1,000 p/ha at 14 m2/person dwelling cell "
                   "(La Ville radieuse 1935) with measured T6 GFA; coverage lands inside doctrinal 5-12%"},
    ]
    notes.append("Business district (24 cruciform 200-m towers at ~5% coverage) is NOT in the window: "
                 "the land-budget denominator in D1 is the housing sector; the vocabulary also has no "
                 "5%-coverage tower utterance (see expressiveness_gaps).")
    notes.append("Doctrine wants few colossal towers per 400-m module; Shanghai T6 (60 m) is 3.3x "
                 "shorter, so doctrinal density forces a tower every 100 m: 'many short towers' "
                 "where the text says 'few tall ones'.")
    return g, budget, {}, {}, geom, notes, {6: n6}


# ----------------------------------------------------------- neighbourhood unit
def build_neighbourhood_unit():
    g = np.zeros((N, N), dtype=np.int16)
    notes = []
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = E_ROAD     # bounding arterials (principle 2)
    g[7:13, 7:13] = E_GREEN                              # central common / playground
    g[9:11, 9:11] = CIVIC                                # school + community site (1 ha)
    parks = [(4, 4), (4, 5), (4, 14), (4, 15), (15, 4), (15, 5), (15, 14), (15, 15)]
    for p in parks:
        g[p] = E_GREEN                                   # distributed small parks
    corners = [(1, 1), (1, 2), (1, 18), (2, 18), (18, 1), (17, 1), (18, 18), (18, 17)]
    cn = counts(g)
    n_res = 400 - cn["road"] - cn["green"] - cn["civic"] - len(corners)
    n_t2_apartment = int(round(0.10 * n_res))
    budget = {"T1": n_res - n_t2_apartment, "T2": n_t2_apartment + len(corners) + cn["civic"],
              "E_ROAD": cn["road"], "E_GREEN": cn["green"], "E_VACANT": 0}
    civic_sem = {"(9, 9)": "education", "(9, 10)": "education", "(10, 9)": "education",
                 "(10, 10)": "public_admin"}
    sem_cells = {"commercial_retail": [list(p) for p in corners]}
    geom = [
        {"name": "bounding arterials", "value": "1-cell E_ROAD border (window = unit)",
         "source": "Perry 1929 principle 2: arterials on all sides; unit is 800 m across, "
                   "stretched to our 1,000-m window"},
        {"name": "central common + school", "value": "6x6 green block, 2x2 civic core",
         "source": "Perry 1929 principle 4: school at centre within 1/4-mile (400 m) walk"},
        {"name": "small parks", "value": "4 x 2-cell parks at quadrant centres; green total = 10%",
         "source": "Perry 1929 principle 3: 'about 10 per cent' recreation space"},
        {"name": "corner shops", "value": "2 cells at each of 4 arterial corners",
         "source": "Perry 1929 principle 6: shops at periphery traffic junctions; land fraction silent"},
        {"name": "interior local streets", "value": "NOT marked (sub-cell, carried by pools)",
         "source": "Perry 1929 principle 5: internal hierarchy discourages through traffic"},
    ]
    notes.append("Window stretches the 800-m unit to 1,000 m (+25% linear): corner cells lie "
                 ">400 m from the school, violating principle 4 at the corners (recorded as gap).")
    notes.append("10% apartment share within housing is a translation choice: Perry's worked "
                 "examples admit apartment variants; doctrine fixes no share.")
    return g, budget, civic_sem, sem_cells, geom, notes, \
        {1: n_res - n_t2_apartment, 2: n_t2_apartment}


# --------------------------------------------------------------------- microrayon
def build_microrayon():
    g = np.zeros((N, N), dtype=np.int16)
    notes = []
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = E_ROAD     # uncrossable bounding arterials
    # green wedges: 2-cell cross from each edge to centre + flares + central pocket
    g[9:11, 1:-1] = E_GREEN
    g[1:-1, 9:11] = E_GREEN
    g[8:12, 8:12] = E_GREEN
    for rr, cc in [(slice(1, 3), (7, 12)), (slice(17, 19), (7, 12))]:
        for c in cc:
            g[rr, c] = E_GREEN
    for cc_, rr_ in [(slice(1, 3), (7, 12)), (slice(17, 19), (7, 12))]:
        for r in rr_:
            g[r, cc_] = E_GREEN
    # school (2x2) tangent to central green; kindergartens at 300-m service spacing
    school = [(6, 6), (6, 7), (7, 6), (7, 7)]
    kgarten = [(4, 4), (4, 15), (15, 4), (15, 15)]
    for p in school + kgarten:
        g[p] = CIVIC
    svc = [(12, 12), (12, 13), (13, 12), (13, 13)]       # daily-service block (torgovy tsentr)
    cn = counts(g)
    # residential count anchored on density: 330 p/ha (canonical 10,000/30 ha) over 100 ha
    target_persons = 330 * 100
    m2pp = 15.0   # ~9 m2 living space/person (SNiP-era sanitary norm) -> ~15 m2 GFA/person
    sh3 = 0.55
    gfa_mix = sh3 * GFA[3] + (1 - sh3) * GFA[4]
    n_res = int(round(target_persons * m2pp / gfa_mix))
    n3 = int(round(sh3 * n_res)); n4 = n_res - n3
    n_vac = 400 - cn["road"] - cn["green"] - cn["civic"] - len(svc) - n_res
    budget = {"T3": n3, "T4": n4, "T2": cn["civic"], "T7": len(svc),
              "E_VACANT": n_vac, "E_ROAD": cn["road"], "E_GREEN": cn["green"]}
    civic_sem = {str(p): "education" for p in school + kgarten}
    sem_cells = {"commercial_retail": [list(p) for p in svc]}
    geom = [
        {"name": "bounding arterials", "value": "1-cell E_ROAD border",
         "source": "SNiP 2.07.01-89: major roads must not cross the microdistrict"},
        {"name": "green wedges", "value": "2-cell cross + edge flares + centre pocket "
         f"= {cn['green']} cells ({cn['green']/4:.0f}% > 20% target)",
         "source": "SNiP 2.07.01-89 sec 2.11: >=6 m2/person greenery at ~330 p/ha => >=20%"},
        {"name": "school site", "value": "2x2 cells (1 ha) tangent to central green",
         "source": "school radius ~750 m (SNiP table value) -> one site serves the window"},
        {"name": "kindergartens", "value": "4 x 1-cell sites at (4,4),(4,15),(15,4),(15,15)",
         "source": "kindergarten radius ~300 m -> 4 sites cover the 1-km window"},
        {"name": "residential cell count", "value": f"{n_res} cells (T3 {n3} / T4 {n4})",
         "source": "anchored on 330 p/ha x 100 ha at 15 m2 GFA/person (9 m2 living-space "
                   "sanitary norm grossed up; translation-layer assumption, labeled)"},
    ]
    notes.append("Window (100 ha) exceeds the SNiP microrayon maximum (60-80 ha) by 25%+ (gap).")
    notes.append(f"Density-anchored translation leaves {n_vac} E_VACANT cells "
                 "(drives/parking/yards): a faithful rendering of SNiP's land-balance silence "
                 "(remainder_unspecified = 0.80).")
    return g, budget, civic_sem, sem_cells, geom, notes, {3: n3, 4: n4}


# ----------------------------------------------------------------------- gb50180
def build_gb50180():
    g = np.zeros((N, N), dtype=np.int16)
    notes = []
    lines = [0, 6, 13, 19]                                # 250/300/250-m road grid
    for k in lines:
        g[k, :] = E_ROAD
        g[:, k] = E_ROAD
    g[7:13, 7:13] = E_GREEN                               # central garden block (36 cells)
    # green strips: middle row of every other block
    strips = []
    bands = [(1, 6), (7, 13), (14, 19)]
    for (r0, r1) in bands:
        for (c0, c1) in bands:
            if (r0, c0) == (7, 7):
                continue
            mid = (r0 + r1 - 1) // 2
            strips += [(mid, c) for c in range(c0, c1)]
    for p in strips:
        g[p] = E_GREEN
    # perimeter commercial podium row (T7) along the northern arterial
    podium = [(1, c) for c in range(1, 19) if c not in lines]
    school = [(10, 2), (10, 3), (11, 2), (11, 3)]
    svc_admin = [(10, 16)]; clinic = [(11, 16)]
    for p in school + svc_admin + clinic:
        g[p] = CIVIC
    cn = counts(g)
    n_res = 400 - cn["road"] - cn["green"] - cn["civic"] - len(podium)
    # T4/T6 split solved so residential-block FAR = 2.5 (mid of high-rise I band 2.2-2.8)
    block_land = 400 - cn["road"] - len(podium)           # 街坊 land excl. city roads & podium
    gfa_target = 2.5 * block_land * CELL_M2 / n_res
    s6 = (gfa_target - GFA[4]) / (GFA[6] - GFA[4])
    n6 = int(round(s6 * n_res)); n4 = n_res - n6
    budget = {"T4": n4, "T6": n6, "T2": cn["civic"], "T7": len(podium),
              "E_VACANT": 0, "E_ROAD": cn["road"], "E_GREEN": cn["green"]}
    civic_sem = {str(p): "education" for p in school}
    civic_sem[str(svc_admin[0])] = "public_admin"
    civic_sem[str(clinic[0])] = "healthcare"
    sem_cells = {"commercial_retail": [list(p) for p in podium]}
    far_blk = (n4 * GFA[4] + n6 * GFA[6]) / (block_land * CELL_M2)
    cov_blk = (n4 * COV[4] + n6 * COV[6]) / block_land
    grn_blk = cn["green"] / block_land
    geom = [
        {"name": "road grid", "value": "1-cell E_ROAD rows/cols {0,6,13,19}: spacings 250/300/250 m; "
         "network density 8 km/km2", "source": "GB 50180-2018 6.0.2: spacing 150-250 m pref., <=300 m; "
         "density >=8 km/km2 (met exactly)"},
        {"name": "central garden", "value": "full 6x6 centre block",
         "source": "GB 50180-2018 4.0.4/4.0.7: concentrated public green per circle"},
        {"name": "green strips", "value": f"mid-row strip per block; total green {cn['green']} cells "
         f"= {grn_blk:.2f} of block land", "source": "GB 50180-2018 Table 4.0.2: green rate 30% (multi) "
         "/ 35% (high-rise) minimums"},
        {"name": "commercial podium row", "value": f"{len(podium)} T7 cells along northern arterial",
         "source": "street-wall podium practice along arterials; GB itself lists facilities, "
                   "not commercial land (doctrine-silent)"},
        {"name": "civic sites", "value": "school 2x2 + service centre 1 + clinic 1",
         "source": "GB 50180-2018 sec 5: 5-min-circle facility checklist (school, community service, "
                   "health station), radius <=300-500 m"},
        {"name": "T4/T6 split", "value": f"T4 {n4} / T6 {n6} solves block FAR = 2.5",
         "source": "GB 50180-2018 Table 4.0.2 high-rise I FAR 2.2-2.8 (climate III, Shanghai)"},
    ]
    notes.append(f"All three GB caps satisfied simultaneously at block level: FAR {far_blk:.2f} "
                 f"(cap band 2.2-2.8), coverage {cov_blk:.2f} (<=0.22), green {grn_blk:.2f} (>=0.30).")
    notes.append("50-m cells quantize every road to 50 m width: 36% road land vs ~15-20% real; "
                 "the 8 km/km2 norm is hit exactly, the land take is a resolution artifact.")
    return g, budget, civic_sem, sem_cells, geom, notes, {4: n4, 6: n6}


# -------------------------------------------------------------- claims + semantics
def semantic_mix(paradigm, budget, civic_sem, sem_cells):
    """Demanded semantic mix over BUILT cells (fractions sum to 1). Rule: residential split
    by height regime (>=18 m doctrine height -> residential_estate, else residential_lowrise);
    civic split per the doctrine's NAMED institutions; commercial/industry from designated cells."""
    built = sum(v for k, v in budget.items() if k.startswith("T"))
    mix = {s: 0.0 for s in SEM8}
    n_civ = {s: sum(1 for v in civic_sem.values() if v == s) for s in SEM8}
    n_com = len(sem_cells.get("commercial_retail", []))
    n_ind = len(sem_cells.get("industrial", []))
    if paradigm == "garden_city":
        mix["residential_lowrise"] = budget["T1"]
    elif paradigm == "ville_radieuse":
        mix["residential_estate"] = budget["T6"]
    elif paradigm == "neighbourhood_unit":
        mix["residential_lowrise"] = budget["T1"] + (budget["T2"] - n_com -
                                                     sum(n_civ.values()))
    elif paradigm == "microrayon":
        mix["residential_estate"] = budget["T3"] + budget["T4"]
    elif paradigm == "gb50180":
        mix["residential_estate"] = budget["T4"] + budget["T6"]
    mix["commercial_retail"] = n_com
    mix["industrial"] = n_ind
    for s in ("public_admin", "education", "healthcare"):
        mix[s] += n_civ[s]
    tot = sum(mix.values())
    assert abs(tot - built) < 1e-6, (paradigm, tot, built)
    return {s: round(v / built, 4) for s, v in mix.items()}


GAPS = {
    "garden_city": [
        {"demand": "1-2 storey cottages on 20x130 ft garden lots (built practice; doctrine height-silent)",
         "nearest_shanghai_utterance": "T1 (cov .17, h 10.0 m, house .95)",
         "distortion": "height inflated ~1.5-2x; Shanghai T1 is attached village/lilong fabric, not detached cottages"},
        {"demand": "Crystal Palace: single-storey glass ring arcade around Central Park",
         "nearest_shanghai_utterance": "T7 big-box (cov .98, h 29.4 m)",
         "distortion": "a thin glass ring becomes 12 opaque podium cells ~3x too tall"},
        {"demand": "factory/warehouse sheds (~10 m) fronting the circular railway",
         "nearest_shanghai_utterance": "T7 (h 29.4 m) alternating with E_VACANT yards",
         "distortion": "shed height ~3x inflated; rail siding grain lost at 50-m cells"},
        {"demand": "radial-concentric geometry: 6 boulevards, circular rings",
         "nearest_shanghai_utterance": "square grid: 4 orthogonal radials, square 'rings'",
         "distortion": "6-fold symmetry -> 4-fold; circles -> squares; boulevards widened 37->50 m"},
        {"demand": "5,000-acre agricultural belt (5/6 of the estate)",
         "nearest_shanghai_utterance": "none (out of frame)",
         "distortion": "the doctrine's core anti-speculation instrument is invisible in a 1-km window"},
    ],
    "ville_radieuse": [
        {"demand": "cruciform business towers, 60 storeys / ~200 m, ~5% ground coverage",
         "nearest_shanghai_utterance": "T8 (h 94.5 m, cov .99) or T6 (h 60.4 m, cov .31)",
         "distortion": "NO token says 'very tall AND nearly no footprint': T8 inverts the open ground "
                       "(coverage .99), T6 is a third of the height; business district omitted from window"},
        {"demand": "a-redent continuous housing slabs, 12-15 storeys (~50 m), on pilotis",
         "nearest_shanghai_utterance": "T3 slab (h 21.5 m) OR T6 tower (h 60.4 m) — never slab-at-50m",
         "distortion": "vocabulary can say 'slab' or '50-60 m' but not both; we chose T6: height kept, "
                       "continuous redent form becomes point towers"},
        {"demand": "one colossal tower per 400-m module ('few, tall, far apart')",
         "nearest_shanghai_utterance": "T6 socket every 100 m",
         "distortion": "to carry 1,000 p/ha with 60-m towers the carpet needs 4x the tower count"},
        {"demand": "elevated motorways; ground 100% pedestrian",
         "nearest_shanghai_utterance": "E_ROAD at grade",
         "distortion": "grade separation unsayable; 19% of ground becomes road surface"},
    ],
    "neighbourhood_unit": [
        {"demand": "single-family houses, 10 families/acre (1-2.5 storeys, detached/row)",
         "nearest_shanghai_utterance": "T1 (cov .17, h 10.0 m, house .95)",
         "distortion": "T1 is denser attached Shanghai fabric; ~3 storeys vs 1-2.5; tenure image "
                       "(freehold lot) not carried"},
        {"demand": "800-m unit with all dwellings within 400 m of the school",
         "nearest_shanghai_utterance": "1,000-m window (unit stretched +25%)",
         "distortion": "corner cells sit 500-700 m from the school: principle 4 fails at corners"},
        {"demand": "curvilinear interior streets discouraging through traffic",
         "nearest_shanghai_utterance": "sub-cell street grain carried implicitly by pools",
         "distortion": "interior street pattern invisible at 50-m resolution (not marked, by design)"},
    ],
    "microrayon": [
        {"demand": "5-storey khrushchevka panel series (~15 m)",
         "nearest_shanghai_utterance": "T3 (h 21.5 m, slab .92)",
         "distortion": "height inflated ~40%; Shanghai pre-1985 estates are the same lineage, "
                       "so form family is faithful"},
        {"demand": "microrayon 10-60 ha (max 80 ha)",
         "nearest_shanghai_utterance": "100-ha window",
         "distortion": "frame oversizes the canonical unit by >=25%; one window ~ 1.5-3 microrayons"},
        {"demand": "free insolation-driven slab placement (no coverage norm)",
         "nearest_shanghai_utterance": "50-m cell tokens with fixed internal arrangements",
         "distortion": "insolation spacing is sub-cell; carried only as far as Shanghai pools "
                       "already embody it"},
    ],
    "gb50180": [
        {"demand": "high-rise I residential: coverage <=22%, height <=54 m (Table 4.0.2)",
         "nearest_shanghai_utterance": "T6 (cov .31, h 60.4 m)",
         "distortion": "the measured Shanghai high-rise token VIOLATES the standard's own cell-level "
                       "caps; compliance only re-emerges at block level via green embedding"},
        {"demand": "open small blocks, dense network ('xiao jiequ, mi luwang', 6.0.2)",
         "nearest_shanghai_utterance": "300-m gated-superblock grid (built practice)",
         "distortion": "we encode the gated practice the standard administers, not the open-block "
                       "reform it asks for: doctrine-vs-practice tension made explicit"},
        {"demand": "road land ~15-20% at 8 km/km2",
         "nearest_shanghai_utterance": "1-cell (50 m) road lines",
         "distortion": "road land overstated at 36% by 50-m quantization (network density exact)"},
    ],
}

BUILDERS = {
    "garden_city": build_garden_city,
    "ville_radieuse": build_ville_radieuse,
    "neighbourhood_unit": build_neighbourhood_unit,
    "microrayon": build_microrayon,
    "gb50180": build_gb50180,
}

PERSONS = {   # (m2 GFA per person, label) — translation-layer assumptions, labeled
    "garden_city": (22.0, "22 m2 GFA/p: Victorian cottage ~120 m2 / 5.45 p-household (Howard's 5,500 lots/30,000)"),
    "ville_radieuse": (14.0, "14 m2/p: Le Corbusier's dwelling-cell norm (La Ville radieuse, 1935)"),
    "neighbourhood_unit": (40.0, "40 m2 GFA/p: ~220 m2 house / 5.5-person 1929 family (indicative)"),
    "microrayon": (15.0, "15 m2 GFA/p: 9 m2 living-space sanitary norm grossed up (SNiP era)"),
    "gb50180": (35.0, "35 m2/p: current Chinese urban per-capita housing floorspace (indicative)"),
}

DOCTRINE_TARGETS = {  # window-level doctrinal anchors for the deviation report
    "garden_city": {"green_frac": 0.26, "residential_frac": 0.33, "density_p_ha": 74},
    "ville_radieuse": {"coverage": [0.05, 0.12], "density_p_ha": 1000},
    "neighbourhood_unit": {"green_frac": 0.10, "density_p_ha": [77, 93]},
    "microrayon": {"green_frac": 0.20, "density_p_ha": [200, 450]},
    "gb50180": {"far_block": [2.2, 2.8], "coverage_block_max": 0.22, "green_rate_min": 0.30},
}


def main():
    translation = {"_schema": {
        "skeleton": "20x20 int16 .npy; 0=E_VACANT(build-allowed) 9=E_ROAD 10=E_WATER(unused) "
                    "11=E_GREEN -1=RESERVED_CIVIC; cell=50 m; window=1 km",
        "token_budget": "400-cell composition; T-counts include RESERVED_CIVIC cells (their carrier "
                        "token is stated per paradigm); E_WATER unused (flat site)",
        "semantic_mix": "DEMANDED mix over built cells; rule: residential split by doctrine height "
                        "regime (>=18 m -> residential_estate else residential_lowrise); civic split "
                        "per the doctrine's NAMED institutions; commercial/industrial from designated "
                        "cells. office_finance=0 everywhere: no doctrine allocates office land inside "
                        "the residential unit (VR banishes it to the business band, out of window).",
        "provenance": "doctrine numbers from V06/doctrine_recipes.json (D1); token stats measured from "
                      "TF/v03/tokens.json; civic carrier T2 chosen because pools give T2 the highest "
                      "public_admin share (.075) and education present (.023) — measured, not asserted."},
        "paradigms": {}, "expressiveness_gaps": GAPS}
    results = {"paradigms": {}, "gap_counts": {}, "files": {}}
    skeletons = {}

    for name, fn in BUILDERS.items():
        g, budget, civic_sem, sem_cells, geom, notes, res_tokens = fn()
        assert sum(budget.values()) == 400, (name, budget, sum(budget.values()))
        np.save(os.path.join(V06, f"skeleton_{name}.npy"), g)
        checks = window_checks(budget, PERSONS[name], res_tokens)
        mix = semantic_mix(name, budget, civic_sem, sem_cells)
        translation["paradigms"][name] = {
            "token_budget": budget,
            "claims": build_claims(name, budget),
            "geometry_constants": geom,
            "reserved_civic_semantics": civic_sem,
            "designated_semantic_cells": sem_cells,
            "semantic_mix_demanded": mix,
            "window_checks": checks,
            "doctrine_targets": DOCTRINE_TARGETS[name],
            "notes": notes,
        }
        results["paradigms"][name] = {
            "token_budget": budget, "skeleton_counts": counts(g),
            "window_checks": checks, "semantic_mix": mix,
            "n_gaps": len(GAPS[name]),
        }
        results["gap_counts"][name] = len(GAPS[name])
        skeletons[name] = g
        results["files"][name] = f"skeleton_{name}.npy"

    json.dump(translation, open(os.path.join(V06, "translation.json"), "w"), indent=1)
    json.dump(results, open(os.path.join(V06, "results_d2.json"), "w"), indent=1)
    draw_figure(skeletons, translation)
    print("written: translation.json, results_d2.json, 5 skeleton_*.npy, fig_d2_translation.png")
    for n, r in results["paradigms"].items():
        print(n, r["token_budget"], r["window_checks"]["window_FAR"],
              r["window_checks"]["implied_persons_per_ha"])


def build_claims(name, budget):
    """Transparent doctrine-claim -> token-allocation table."""
    C = []
    a = lambda **kw: C.append(kw)
    if name == "garden_city":
        a(claim="residential = 0.33 of town (5,500 lots of 20x130 ft)",
          source="doctrine_recipes garden_city.land_budget.residential (medium)",
          allocation={"T1": budget["T1"]},
          reason=f"T1 house-token cells = {budget['T1']}/400 = {budget['T1']/400:.2f} "
                 "(short of .33: road/industry quantization eats lots; deviation reported)")
        a(claim="green = 0.26 (Central Park 145 + Grand Avenue 115 acres)",
          source="garden_city.land_budget.green_open (high)",
          allocation={"E_GREEN": budget["E_GREEN"]},
          reason=f"park disk + Grand-Avenue annulus = {budget['E_GREEN']}/400 = "
                 f"{budget['E_GREEN']/400:.2f}")
        a(claim="factory ring on circular railway (acreage silent)",
          source="garden_city.land_budget.work_industry (doctrine_silent)",
          allocation={"T7": budget["T7"] - 12, "E_VACANT": budget["E_VACANT"]},
          reason="1-cell ring inside rail; alternate T7 shed / E_VACANT yard (derived resolution "
                 "of remainder_unspecified)")
        a(claim="Crystal Palace ring arcade (acreage silent)",
          source="garden_city.land_budget.commercial_shops (doctrine_silent)",
          allocation={"T7": 12}, reason="12 cells by angle on the park-edge ring")
        a(claim="civic in park + schools in Grand Avenue",
          source="garden_city.land_budget.civic_public (doctrine_silent)",
          allocation={"T2_as_RESERVED_CIVIC": budget["T2"]},
          reason="2x2 centre cluster + 6 ward schools; T2 = measured civic carrier")
        a(claim="6 boulevards + rail ring (acreage silent)",
          source="garden_city.street_doctrine (high)",
          allocation={"E_ROAD": budget["E_ROAD"]},
          reason="geometry generates the road budget the text never states")
    elif name == "ville_radieuse":
        a(claim="housing 1,000 p/ha; ground coverage 5-12%; 12-60 storeys",
          source="vr.density_net_residential (high), ground_coverage (medium), height_regime (medium)",
          allocation={"T6": budget["T6"]},
          reason=f"{budget['T6']} T6 cells solve 1,000 p/ha at LC's 14 m2/p cell with measured "
                 f"T6 GFA ({GFA[6]:.0f} m2); block coverage = {budget['T6']*COV[6]/400:.3f} in [.05,.12]")
        a(claim="ground is a continuous park ('88-95% open')",
          source="vr.land_budget.green_open (medium)",
          allocation={"E_GREEN": budget["E_GREEN"]},
          reason=f"green carpet = {budget['E_GREEN']}/400; open ground incl. roads = "
                 f"{1-budget['T6']/400:.2f} of window")
        a(claim="400-m superblock arterial grid, circulation separated",
          source="vr.block_grain (medium), street_doctrine (high)",
          allocation={"E_ROAD": budget["E_ROAD"]},
          reason="rows/cols {4,12}: window edges mid-module (carpet is extensible, unit has no edge)")
    elif name == "neighbourhood_unit":
        a(claim="open space 'about 10 per cent'",
          source="nu.land_budget.green_open (high)",
          allocation={"E_GREEN": budget["E_GREEN"]},
          reason=f"central common + 4 quadrant parks = {budget['E_GREEN']}/400 = "
                 f"{budget['E_GREEN']/400:.2f} exactly")
        a(claim="school at centre, 400-m walk (principle 4)",
          source="nu.service_radii (high)",
          allocation={"T2_as_RESERVED_CIVIC": 4},
          reason="2x2 civic core in the common: 3 education + 1 public_admin (community house)")
        a(claim="shops at arterial corner junctions (principle 6, land silent)",
          source="nu.land_budget.commercial_shops (doctrine_silent)",
          allocation={"T2_corner_shops": 8}, reason="2 cells per corner; T2 = lowrise retail row")
        a(claim="residential = residual at 10 families/acre",
          source="nu.density_net (high); residential fraction doctrine_silent",
          allocation={"T1": budget["T1"], "T2_apartments": budget["T2"] - 12},
          reason="residual fabric; 10% apartment variant share (Perry's worked examples)")
        a(claim="bounding arterials, no through traffic",
          source="nu.street_doctrine (high)",
          allocation={"E_ROAD": budget["E_ROAD"]}, reason="window border = the four arterials")
    elif name == "microrayon":
        a(claim="density 200-450 p/ha (canonical 330)",
          source="mr.density_gross (medium)",
          allocation={"T3": budget["T3"], "T4": budget["T4"]},
          reason="cell count anchored on 330 p/ha at 15 m2 GFA/p; T3/T4 = 5-16-storey panel "
                 "series rendered by the same-lineage Shanghai estate tokens")
        a(claim=">=6 m2/person green (>=20% at 330 p/ha)",
          source="mr.land_budget.green_open (medium)",
          allocation={"E_GREEN": budget["E_GREEN"]},
          reason=f"wedge cross = {budget['E_GREEN']}/400 = {budget['E_GREEN']/400:.2f} >= .20")
        a(claim="school 750 m + kindergarten 300 m service radii",
          source="mr.service_radii (medium)",
          allocation={"T2_as_RESERVED_CIVIC": 8},
          reason="1 school (2x2) + 4 kindergartens at 300-m coverage spacing")
        a(claim="daily services per-capita norms (land silent)",
          source="mr.land_budget.commercial_shops (doctrine_silent)",
          allocation={"T7": 4}, reason="one 4-cell service block (torgovy tsentr analogue)")
        a(claim="uncrossable superblock; land balance unstated",
          source="mr.street_doctrine (high); land_budget silences",
          allocation={"E_ROAD": budget["E_ROAD"], "E_VACANT": budget["E_VACANT"]},
          reason="border arterials; E_VACANT = the doctrine's unbudgeted drives/parking/yards")
    elif name == "gb50180":
        a(claim="block FAR 2.2-2.8, coverage <=22%, height class mix (Table 4.0.2, climate III)",
          source="gb.far_block_climate_III (high), ground_coverage (high)",
          allocation={"T4": budget["T4"], "T6": budget["T6"]},
          reason="T4/T6 split solved for block FAR 2.5; block coverage lands <= .22 via green embedding")
        a(claim="green rate >=30% + concentrated central green",
          source="gb.land_budget.green_open (high)",
          allocation={"E_GREEN": budget["E_GREEN"]},
          reason=f"centre block + per-block strips = {budget['E_GREEN']} cells = "
                 f"{budget['E_GREEN']/(400-budget['E_ROAD']-16):.2f} of block land")
        a(claim="dense road grid <=300 m, >=8 km/km2",
          source="gb.block_grain + street_doctrine (high)",
          allocation={"E_ROAD": budget["E_ROAD"]},
          reason="rows/cols {0,6,13,19}: 250/300/250-m spacing, 8 km/km2 exact")
        a(claim="5-min-circle facilities (school, service centre, clinic)",
          source="gb.service_radii (medium); facility checklists",
          allocation={"T2_as_RESERVED_CIVIC": 6}, reason="school 2x2 + admin 1 + health 1")
        a(claim="arterial commercial podium (GB land-silent; practice)",
          source="gb.land_budget.commercial_shops (doctrine_silent)",
          allocation={"T7": budget["T7"]}, reason="16-cell podium row along northern arterial")
    return C


# ------------------------------------------------------------------------ figure
def draw_figure(skeletons, translation):
    # value -> color: CIVIC red, VAC light, ROAD near-black, WATER blue, GREEN green
    cmap = ListedColormap(["#d62828", "#f2efe9", "#1a1a1a", "#b8cbe0", "#74c69d"])
    norm = BoundaryNorm([-1.5, -0.5, 4.5, 9.5, 10.5, 11.5], cmap.N)
    titles = {"garden_city": "Garden City (Howard 1898)",
              "ville_radieuse": "Ville Radieuse (Corbusier 1935)",
              "neighbourhood_unit": "Neighbourhood Unit (Perry 1929)",
              "microrayon": "Microrayon (SNiP 2.07.01-89)",
              "gb50180": "GB 50180-2018"}
    # canvas taller than v1 so the audit table can carry a readable font
    fig = plt.figure(figsize=(16.5, 13.5))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.75, 2.0], hspace=0.13, wspace=0.12,
                          top=0.93, bottom=0.05, left=0.02, right=0.98)
    for i, (name, g) in enumerate(skeletons.items()):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(g, cmap=cmap, norm=norm, interpolation="nearest")
        # mark designated commercial / industrial cells
        sem = translation["paradigms"][name]["designated_semantic_cells"]
        for (r, c) in sem.get("commercial_retail", []):
            ax.plot(c, r, marker="s", mfc="none", mec="#e76f51", mew=1.4, ms=5.5)
        for (r, c) in sem.get("industrial", []):
            ax.plot(c, r, marker="x", color="#6c584c", mew=1.4, ms=5)
        ax.set_title(titles[name], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        bud = translation["paradigms"][name]["token_budget"]
        btxt = "  ".join(f"{k}:{v}" for k, v in bud.items() if v and k.startswith("T"))
        vtxt = (f"ROAD:{bud['E_ROAD']}  GREEN:{bud['E_GREEN']}  "
                f"VAC:{bud.get('E_VACANT',0)}  CIVIC:{int((g==CIVIC).sum())}")
        ax.set_xlabel(btxt + "\n" + vtxt, fontsize=7.6)
    # legend row
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
               ["#f2efe9", "#1a1a1a", "#74c69d", "#d62828"]]
    handles += [plt.Line2D([], [], marker="s", mfc="none", mec="#e76f51", ls="", ms=6),
                plt.Line2D([], [], marker="x", color="#6c584c", ls="", ms=6)]
    fig.legend(handles, ["E_VACANT (build-allowed)", "E_ROAD", "E_GREEN",
                         "RESERVED-CIVIC", "designated commercial", "designated industrial"],
               loc="center", bbox_to_anchor=(0.5, 0.678), ncol=6, fontsize=8.6, frameon=False)
    # expressiveness-gap table
    axt = fig.add_subplot(gs[1, :]); axt.axis("off")
    rows = []
    for p, gaps in GAPS.items():
        for gp in gaps:
            rows.append([titles[p].split(" (")[0], gp["demand"],
                         gp["nearest_shanghai_utterance"], gp["distortion"]])
    import textwrap
    wrap = lambda s, w: "\n".join(textwrap.wrap(s, w))
    cell_text = [[wrap(r[0], 16), wrap(r[1], 52), wrap(r[2], 38), wrap(r[3], 62)] for r in rows]
    tbl = axt.table(cellText=cell_text,
                    colLabels=["Paradigm", "Doctrine demand", "Nearest Shanghai utterance",
                               "Distortion"],
                    colWidths=[0.09, 0.30, 0.235, 0.375], loc="upper center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    # row height: per-row max line count so every row fits inside the axes
    nrows = len(cell_text) + 1
    row_lines = {0: 1}
    for ri, row in enumerate(cell_text, start=1):
        row_lines[ri] = max(s.count("\n") + 1 for s in row)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        cell.set_height(0.009 + 0.0185 * row_lines[r] if r > 0 else 0.034)
        if r == 0:
            cell.set_text_props(weight="bold"); cell.set_facecolor("#e9ecef")
        cell.PAD = 0.02
    fig.suptitle("D2 — Doctrine recipes translated into the Shanghai 50-m token vocabulary: "
                 "typed-void skeletons (top) and the expressiveness audit (bottom)",
                 fontsize=12.5, y=0.975)
    fig.text(0.02, 0.012, "Skeletons: 20x20 cells x 50 m = 1 km. Token budgets under each panel "
             "(counts / 400 cells). Gap table: what each doctrine demands that the measured "
             "Shanghai vocabulary cannot say.", fontsize=7.8, color="#555555")
    fig.savefig(os.path.join(FIG, "fig_d2_translation.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
