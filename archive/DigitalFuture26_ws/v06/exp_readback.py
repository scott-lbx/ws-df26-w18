#!/usr/bin/env python3
"""
D4 — stakeholder read-back gap (v06 core experiment).

For each generated canvas (5 doctrine units + 5 3x3km composites) run the
v05 E2 power-attribution chain on the generated buildings:
    building semantic -> power_priors_v05 cell -> 4-dim weight vector,
    GFA-weighted mean over labeled buildings (unlabeled excluded, share reported).
residential_estate uses the *no_age* cell: generated buildings have no
construction year, so EVERY estate falls into the v05-documented low-confidence
fallback (gov .4 / dev .5 / res .3 / inf .05). This is disclosed and exploited:
"Chinese no-age priors reading Howard/Corbusier/microrayon".

Gap analysis: doctrine stakeholder_apriori (D1, asserted) vs read-back profile
(observed-through-priors). Signed per-dim gap, L1 gap, ranking-flip table.

Decomposition (sequential counterfactual, order disclosed):
    R0 = actual read-back (donor-inherited semantics, estate->no_age)
    R1 = read-back with demanded_semantic (the doctrine program as translated
         by D2/D3) instead of donor semantics, estate->no_age
         => R1 - R0 = (b) collage semantic drift (within-donor-cell semantic
            heterogeneity + unlabeled donors; D3 cell-level fallback rate = 0)
    R2 = demanded_semantic AND estate prior cell no_age -> pre1985
         (the historically apt cell for pre-market doctrines: danwei/public-
         housing logic, gov .7) => R2 - R1 = (a) no-age fallback artifact bound
    residual apriori - R2 = (c) genuine program translation gap (what remains
         with perfect semantic fidelity and a state-era estate reading: prior-
         vocabulary compression + program shares the pools cannot express).
Per dim: gap = (a)+(b)+(c) exactly (telescoping). L1 magnitudes of components
may exceed |gap| via cancellation (disclosed).

Outputs: V06/results_d4.json, V06/fig/fig_d4_gap.png, V06/fig/fig_d4_decomp.png
Seed 0 (no stochastic steps; deterministic aggregation).
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/Users/jinbao/Projects/Urban-Stakeholder-Workshop/typology_foundation'
V06 = f'{ROOT}/v06'
DIMS = ['government', 'developer_capital', 'resident_community', 'informal']
DIM_SHORT = {'government': 'gov', 'developer_capital': 'dev',
             'resident_community': 'res', 'informal': 'inf'}
PARADIGMS = ['garden_city', 'ville_radieuse', 'neighbourhood_unit',
             'microrayon', 'gb50180']
PNAMES = {'garden_city': 'Garden City (Howard 1898/1902)',
          'ville_radieuse': 'Ville Radieuse (Le Corbusier 1933)',
          'neighbourhood_unit': 'Neighbourhood Unit (Perry 1929)',
          'microrayon': 'Microrayon (SNiP 1958+)',
          'gb50180': 'GB50180 xiaoqu (1993/2018)'}
np.random.seed(0)

# ---------------------------------------------------------------- priors
P = json.load(open(f'{ROOT}/v05/power_priors_v05.json'))
PRIORS = P['priors']
BASELINE = P['baseline_unspecified']

LOW_CONF = {'low', 'medium-low'}  # v05 E2 definition


def prior_cell(sem, estate_cell='no_age'):
    """semantic -> (4-dim weight dict, confidence). v05 E2 chain verbatim;
    estate cell selectable for the counterfactual swap."""
    if sem == 'residential_estate':
        c = PRIORS['residential_estate']['cells'][estate_cell]
        return c['w'], c['confidence']
    if sem in PRIORS:
        return PRIORS[sem]['w'], PRIORS[sem]['confidence']
    return BASELINE, 'low'  # baseline fallback (should not trigger here)


def readback(df, sem_col='semantic', estate_cell='no_age'):
    """GFA-weighted mean power profile over labeled buildings."""
    g = df[df[sem_col] != 'unlabeled'].copy()
    W = np.array([prior_cell(s, estate_cell)[0][d]
                  for s in g[sem_col] for d in DIMS]).reshape(len(g), 4)
    wts = g['gfa'].to_numpy()
    prof = {d: float(np.average(W[:, i], weights=wts))
            for i, d in enumerate(DIMS)}
    conf = np.array([prior_cell(s, estate_cell)[1] in LOW_CONF
                     for s in g[sem_col]])
    low_share = float(wts[conf].sum() / wts.sum())
    unl_share = float(1 - wts.sum() / df['gfa'].sum())
    return prof, low_share, unl_share, int(len(g))


# ---------------------------------------------------------------- doctrines
REC = json.load(open(f'{V06}/doctrine_recipes.json'))['doctrines']
APRIORI = {p: REC[p]['stakeholder_apriori']['config'] for p in PARADIGMS}

# ---------------------------------------------------------------- run
results = {'_meta': {
    'script': 'v06/exp_readback.py', 'seed': 0,
    'chain': 'v05 E2 verbatim: semantic -> power_priors_v05 cell -> GFA-weighted mean; unlabeled excluded',
    'estate_cell': "no_age for ALL residential_estate (generated buildings carry no construction year) "
                   "= the v05-documented low-confidence fallback (gov .4 dev .5 res .3 inf .05), "
                   "deliberately retained: 'Chinese no-age priors reading the doctrines'",
    'decomposition': 'sequential counterfactuals R0->R1 (semantic: donor->demanded) ->R2 (estate cell: '
                     'no_age->pre1985); per-dim gap = (c: apriori-R2) + (a: R2-R1) + (b: R1-R0) exactly; '
                     'order-dependent (b before a) — a Shapley average was not computed (single path disclosed)',
    'apriori_source': 'V06/doctrine_recipes.json stakeholder_apriori (D1, ASSERTED doctrine-level claims)',
}, 'paradigms': {}}

profiles = {}   # (paradigm, scope) -> dict
for p in PARADIGMS:
    results['paradigms'][p] = {}
    for scope in ['unit', 'city']:
        df = gpd.read_parquet(f'{V06}/cache_gen/{scope}_{p}.parquet')
        r0, low0, unl0, n0 = readback(df, 'semantic', 'no_age')
        r1, _, _, n1 = readback(df, 'demanded_semantic', 'no_age')
        r2, _, _, _ = readback(df, 'demanded_semantic', 'pre1985')
        ap = APRIORI[p]
        gap = {d: round(ap[d] - r0[d], 4) for d in DIMS}
        comp_c = {d: round(ap[d] - r2[d], 4) for d in DIMS}
        comp_a = {d: round(r2[d] - r1[d], 4) for d in DIMS}
        comp_b = {d: round(r1[d] - r0[d], 4) for d in DIMS}
        # GFA shares for context
        lab = df[df['semantic'] != 'unlabeled']
        sem_gfa = (lab.groupby('semantic')['gfa'].sum() /
                   lab['gfa'].sum()).round(4).to_dict()
        estate_gfa = sem_gfa.get('residential_estate', 0.0)
        profiles[(p, scope)] = dict(r0=r0, r1=r1, r2=r2)
        results['paradigms'][p][scope] = {
            'readback_R0': {d: round(r0[d], 4) for d in DIMS},
            'cf_R1_demanded_semantic': {d: round(r1[d], 4) for d in DIMS},
            'cf_R2_demanded_plus_pre1985': {d: round(r2[d], 4) for d in DIMS},
            'apriori': ap,
            'gap_signed_apriori_minus_readback': gap,
            'L1_gap': round(sum(abs(v) for v in gap.values()), 4),
            'doctrine_first_dim': max(DIMS, key=lambda d: ap[d]),
            'fabric_first_dim': max(DIMS, key=lambda d: r0[d]),
            'ranking_flip': max(DIMS, key=lambda d: ap[d]) != max(DIMS, key=lambda d: r0[d]),
            'decomposition': {
                'a_no_age_artifact': comp_a,
                'b_collage_semantic_drift': comp_b,
                'c_genuine_program_translation': comp_c,
                'L1_components': {
                    'a': round(sum(abs(v) for v in comp_a.values()), 4),
                    'b': round(sum(abs(v) for v in comp_b.values()), 4),
                    'c': round(sum(abs(v) for v in comp_c.values()), 4)},
            },
            'low_confidence_gfa_share': round(low0, 4),
            'unlabeled_gfa_share': round(unl0, 4),
            'n_buildings_labeled': n0,
            'semantic_gfa_shares': sem_gfa,
            'res_estate_gfa_share': estate_gfa,
        }

# ranking flip table (unit scope = the canonical doctrine module)
flip_table = []
for p in PARADIGMS:
    u = results['paradigms'][p]['unit']
    flip_table.append({
        'paradigm': p,
        'doctrine_claims_first': u['doctrine_first_dim'],
        'fabric_reads_first': u['fabric_first_dim'],
        'flip': u['ranking_flip'],
        'L1_gap_unit': u['L1_gap'],
        'L1_gap_city': results['paradigms'][p]['city']['L1_gap'],
    })
results['ranking_flip_table'] = flip_table

# headline check (verify, not assume): do gov/community-claiming doctrines
# read back developer-first?
hl = {}
for p in PARADIGMS:
    u = results['paradigms'][p]['unit']
    hl[p] = (f"claims {u['doctrine_first_dim']} "
             f"({APRIORI[p][u['doctrine_first_dim']]:.2f}) -> fabric reads "
             f"{u['fabric_first_dim']} ({u['readback_R0'][u['fabric_first_dim']]:.3f})"
             + ('  [FLIP]' if u['ranking_flip'] else '  [held]'))
results['headline_per_paradigm'] = hl

# ---------------------------------------------------------------- fig 1: gap
DIM_COLORS = {'government': '#b2182b', 'developer_capital': '#2166ac',
              'resident_community': '#1b7837', 'informal': '#bf812d'}
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
axes = axes.ravel()
for i, p in enumerate(PARADIGMS):
    ax = axes[i]
    u = results['paradigms'][p]['unit']
    c = results['paradigms'][p]['city']
    ys = np.arange(len(DIMS))[::-1]
    for y, d in zip(ys, DIMS):
        a, r = u['apriori'][d], u['readback_R0'][d]
        ax.plot([a, r], [y, y], color=DIM_COLORS[d], lw=2, alpha=.55, zorder=2)
        ax.scatter([a], [y], s=90, facecolor='white', edgecolor=DIM_COLORS[d],
                   lw=2, zorder=3, label='doctrine a-priori' if y == ys[0] else None)
        ax.scatter([r], [y], s=90, color=DIM_COLORS[d], zorder=4,
                   label='fabric read-back (unit)' if y == ys[0] else None)
        ax.scatter([c['readback_R0'][d]], [y], marker='x', s=55, color='0.35',
                   zorder=5, label='read-back (3x3 km city)' if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels([DIM_SHORT[d] for d in DIMS])
    ax.set_xlim(-0.04, 1.0)
    ax.set_xlabel('stakeholder weight [0,1]')
    flip = ' — RANK FLIP' if u['ranking_flip'] else ''
    ax.set_title(f"{PNAMES[p]}\nL1 gap {u['L1_gap']:.2f} (unit){flip}", fontsize=10)
    ax.grid(axis='x', alpha=.3)
    if i == 0:
        ax.legend(loc='lower right', fontsize=7.5, framealpha=.9)
# panel 6: L1 gap bars
ax = axes[5]
x = np.arange(len(PARADIGMS))
l1u = [results['paradigms'][p]['unit']['L1_gap'] for p in PARADIGMS]
l1c = [results['paradigms'][p]['city']['L1_gap'] for p in PARADIGMS]
ax.bar(x - .19, l1u, .38, color='#4d4d4d', label='unit (1 km)')
ax.bar(x + .19, l1c, .38, color='#bababa', label='composite (3 km)')
for xi, (vu, vc) in enumerate(zip(l1u, l1c)):
    ax.text(xi - .19, vu + .02, f'{vu:.2f}', ha='center', fontsize=8)
    ax.text(xi + .19, vc + .02, f'{vc:.2f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(['garden\ncity', 'ville\nradieuse', 'nbhd\nunit',
                    'micro-\nrayon', 'GB\n50180'], fontsize=9)
ax.set_ylabel('L1 gap (a-priori vs read-back)')
ax.set_ylim(0, 1.62)
ax.set_title('Total claim-vs-fabric gap', fontsize=10)
ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(0.0, 1.0))
ax.grid(axis='y', alpha=.3)
fig.suptitle('D4 — Who the doctrine claims to serve vs whose city the fabric reads as '
             '(v05 power priors, GFA-weighted; estates forced to no-age cell)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{V06}/fig/fig_d4_gap.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- fig 2: decomposition
COMP_COLORS = {'c': '#762a83', 'a': '#d6604d', 'b': '#92c5de'}
COMP_LABEL = {'c': '(c) genuine program translation (a-priori - R2)',
              'a': '(a) no-age estate artifact (R2 - R1)',
              'b': '(b) collage semantic drift (R1 - R0)'}
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
axes = axes.ravel()
for i, p in enumerate(PARADIGMS):
    ax = axes[i]
    u = results['paradigms'][p]['unit']
    dec = u['decomposition']
    x = np.arange(len(DIMS))
    pos = np.zeros(len(DIMS))
    neg = np.zeros(len(DIMS))
    for comp, key in [('c', 'c_genuine_program_translation'),
                      ('a', 'a_no_age_artifact'),
                      ('b', 'b_collage_semantic_drift')]:
        vals = np.array([dec[key][d] for d in DIMS])
        bottom = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, .62, bottom=bottom, color=COMP_COLORS[comp],
               label=COMP_LABEL[comp] if i == 0 else None)
        pos += np.where(vals >= 0, vals, 0)
        neg += np.where(vals < 0, vals, 0)
    tot = np.array([u['gap_signed_apriori_minus_readback'][d] for d in DIMS])
    ax.scatter(x, tot, marker='D', s=42, color='black', zorder=5,
               label='total signed gap' if i == 0 else None)
    ax.axhline(0, color='0.4', lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT[d] for d in DIMS])
    ax.set_ylim(-0.75, 0.95)
    ax.set_ylabel('signed gap contribution')
    ax.set_title(PNAMES[p], fontsize=10)
    ax.grid(axis='y', alpha=.3)
# panel 6: legend + L1 magnitudes
ax = axes[5]
x = np.arange(len(PARADIGMS))
bot = np.zeros(len(PARADIGMS))
for comp in ['c', 'a', 'b']:
    vals = np.array([results['paradigms'][p]['unit']['decomposition']
                     ['L1_components'][comp] for p in PARADIGMS])
    ax.bar(x, vals, .6, bottom=bot, color=COMP_COLORS[comp])
    bot += vals
l1 = np.array([results['paradigms'][p]['unit']['L1_gap'] for p in PARADIGMS])
ax.scatter(x, l1, marker='D', s=42, color='black', zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(['garden\ncity', 'ville\nradieuse', 'nbhd\nunit',
                    'micro-\nrayon', 'GB\n50180'], fontsize=9)
ax.set_ylabel('L1 magnitude of components')
ax.set_title('Component L1 magnitudes (stack) vs net L1 gap (diamond);\n'
             'stack > diamond = sign cancellation between components', fontsize=9)
ax.grid(axis='y', alpha=.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.5, 0.0))
fig.suptitle('D4 — Decomposing the claim-vs-fabric gap (unit canvases): '
             'artifact vs drift vs genuine translation loss', fontsize=12)
fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(f'{V06}/fig/fig_d4_decomp.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- save
with open(f'{V06}/results_d4.json', 'w') as f:
    json.dump(results, f, indent=1)

for p in PARADIGMS:
    print(p, hl[p])
print('\nL1 unit:', {p: results['paradigms'][p]['unit']['L1_gap'] for p in PARADIGMS})
print('written: results_d4.json, fig_d4_gap.png, fig_d4_decomp.png')
