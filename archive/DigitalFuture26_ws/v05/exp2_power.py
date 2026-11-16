#!/usr/bin/env python3
"""
v05 Experiment 2 — power-attribution chain at 24-window scale.

Steps:
  1. Reuse TF/probe/power_priors_v01.json verbatim -> V05/power_priors_v05.json
     (all 8 semantic classes; baseline fallback if any missing; top-level note).
  2. Per-building power weight vector via semantic class (+ age regime for
     residential_estate). unlabeled excluded from aggregation.
  3. Window power profiles = GFA-weighted mean per dim over labeled buildings,
     plus low-confidence GFA share and res-estate age-regime GFA shares.
  4. Bridge: semantic x family co-occurrence on 18 TRAIN windows (counts and
     GFA-weighted, row-normalized semantic->family); flagship-cell comparison
     against the 3-patch probe.
  5. Probe-surprise checks (Caoyang A13 dev>gov?, gov-dominant windows,
     age_filled=0 suburbs).

Outputs: V05/results_e2.json, V05/power_priors_v05.json,
         V05/fig/fig_e2_profiles.png, V05/fig/fig_e2_bridge.png
Seed 0 (no stochastic steps, kept for convention).
"""
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path('/Users/jinbao/Projects/Urban-Stakeholder-Workshop')
TF = ROOT / 'typology_foundation'
V05 = TF / 'v05'
FIG = V05 / 'fig'
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)  # convention; no stochastic steps below

DIMS = ['government', 'developer_capital', 'resident_community', 'informal']
SEM8 = ['residential_estate', 'residential_lowrise', 'commercial_retail',
        'office_finance', 'industrial', 'public_admin', 'education', 'healthcare']
FAM5 = ['house', 'slab', 'tower', 'large_footprint', 'midrise_other']
LOWCONF = {'low', 'medium-low'}

# ---------------------------------------------------------------- 1) priors
v01 = json.load(open(TF / 'probe' / 'power_priors_v01.json'))
v05p = copy.deepcopy(v01)
added_fallback = []
for sem in SEM8:
    if sem not in v05p['priors']:
        v05p['priors'][sem] = {
            'w': dict(v05p['baseline_unspecified']),
            'provenance': 'baseline fallback — no B-report row',
            'confidence': 'low',
        }
        added_fallback.append(sem)
v05p['note_v05'] = ('v05 reuses power_priors_v01 unchanged (verbatim values). '
                    'These are literature priors, NOT calibrated against any '
                    'observed outcome; per-cell provenance and confidence kept. '
                    'pre1985 bin: the Age layer truncates at 1984/85 (spike), '
                    "so pre1985 reads as 'built 1985 or earlier'.")
json.dump(v05p, open(V05 / 'power_priors_v05.json', 'w'), indent=2, ensure_ascii=False)

# ------------------------------------------------- 2) per-building weights
df = pd.read_parquet(V05 / 'dataset.parquet',
                     columns=['bid', 'window', 'name', 'role', 'semantic',
                              'family', 'age', 'gfa'])

def age_regime(age):
    if pd.isna(age):
        return 'no_age'
    if age <= 1985:
        return 'pre1985'   # truncation: '1985 or earlier'
    if age <= 1997:
        return '1985_1997'
    return 'post1998'

def prior_cell(sem, age):
    """Return (w dict, confidence, cell_name) for a labeled building."""
    p = v05p['priors'][sem]
    if p.get('time_conditioned'):
        reg = age_regime(age)
        cell = p['cells'][reg]
        return cell['w'], cell['confidence'], reg
    return p['w'], p['confidence'], sem

lab = df[df['semantic'] != 'unlabeled'].copy()
cells = [prior_cell(s, a) for s, a in zip(lab['semantic'], lab['age'])]
for d in DIMS:
    lab[d] = [c[0][d] for c in cells]
lab['conf'] = [c[1] for c in cells]
lab['low_conf'] = lab['conf'].isin(LOWCONF)

# ------------------------------------------------- 3) window power profiles
wmeta = json.load(open(V05 / 'windows.json'))['windows']
profiles = {}
for wid, g in lab.groupby('window'):
    gfa_lab = g['gfa'].sum()
    prof = {d: float(np.average(g[d], weights=g['gfa'])) for d in DIMS}
    re = g[g['semantic'] == 'residential_estate']
    re_gfa = re['gfa'].sum()
    age_shares = {}
    if re_gfa > 0:
        reg = re['age'].map(age_regime)
        for r in ['pre1985', '1985_1997', 'post1998', 'no_age']:
            age_shares[r] = round(float(re.loc[reg == r, 'gfa'].sum() / re_gfa), 4)
    tot = df.loc[df['window'] == wid, 'gfa'].sum()
    profiles[wid] = {
        'name': wmeta[wid]['name'],
        'role': wmeta[wid]['role'],
        'profile': {d: round(prof[d], 4) for d in DIMS},
        'dominant_dim': max(prof, key=prof.get),
        'low_confidence_gfa_share': round(float(g.loc[g['low_conf'], 'gfa'].sum() / gfa_lab), 4),
        'unlabeled_gfa_share': round(float(1 - gfa_lab / tot), 4),
        'res_estate_age_regime_gfa_share': age_shares,
        'res_estate_gfa_share_of_labeled': round(float(re_gfa / gfa_lab), 4),
        'age_filled_pct': wmeta[wid]['age_filled_pct'],
    }

# ------------------------------------------------- 4) bridge (TRAIN windows)
tr = lab[lab['role'] == 'train']
cnt = pd.crosstab(tr['semantic'], tr['family']).reindex(index=SEM8, columns=FAM5).fillna(0)
gfa = pd.crosstab(tr['semantic'], tr['family'], values=tr['gfa'],
                  aggfunc='sum').reindex(index=SEM8, columns=FAM5).fillna(0.0)
cnt_rn = cnt.div(cnt.sum(axis=1), axis=0)
gfa_rn = gfa.div(gfa.sum(axis=1), axis=0)

# probe flagship cells (TF/probe/results.json cooccurrence_row_normalized,
# stored there as [family][semantic] = P(family|semantic))
probe_flag = {'residential_lowrise->house': 1.00,
              'residential_estate->slab': 0.34,
              'residential_estate->midrise_other': 0.504}
v05_flag_cnt = {
    'residential_lowrise->house': round(float(cnt_rn.loc['residential_lowrise', 'house']), 4),
    'residential_estate->slab': round(float(cnt_rn.loc['residential_estate', 'slab']), 4),
    'residential_estate->midrise_other': round(float(cnt_rn.loc['residential_estate', 'midrise_other']), 4),
}
v05_flag_gfa = {
    'residential_lowrise->house': round(float(gfa_rn.loc['residential_lowrise', 'house']), 4),
    'residential_estate->slab': round(float(gfa_rn.loc['residential_estate', 'slab']), 4),
    'residential_estate->midrise_other': round(float(gfa_rn.loc['residential_estate', 'midrise_other']), 4),
}

# ------------------------------------------------- 5) probe-surprise checks
a13 = profiles['A13']['profile']
caoyang_check = {
    'v05_government': a13['government'],
    'v05_developer_capital': a13['developer_capital'],
    'dev_gt_gov_v05': bool(a13['developer_capital'] > a13['government']),
    'probe_government': 0.416, 'probe_developer_capital': 0.524,
    'note': 'probe found dev>gov in Caoyang (worker-village heritage window)',
}
gov_dominant = sorted([w for w, p in profiles.items()
                       if p['dominant_dim'] == 'government'])
dev_dominant = sorted([w for w, p in profiles.items()
                       if p['dominant_dim'] == 'developer_capital'])
zero_age = sorted([w for w, p in profiles.items() if p['age_filled_pct'] == 0.0])
zero_age_summary = {w: {
    'name': profiles[w]['name'],
    'res_estate_no_age_share': profiles[w]['res_estate_age_regime_gfa_share'].get('no_age'),
    'low_confidence_gfa_share': profiles[w]['low_confidence_gfa_share'],
    'profile': profiles[w]['profile']} for w in zero_age}

results = {
    'meta': {
        'script': 'v05/exp2_power.py', 'seed': 0,
        'priors': 'power_priors_v05.json (= probe v01 verbatim, literature priors, not calibrated)',
        'aggregation': 'GFA-weighted mean over labeled buildings; unlabeled excluded',
        'low_confidence_def': "prior-cell confidence in {'low','medium-low'}",
        'age_regimes': 'pre1985: age<=1985 (Age layer truncates at 1984/85 -> read as 1985-or-earlier); '
                       '1985_1997: 1986-1997; post1998: age>=1998; NaN -> no_age',
        'classes_added_as_baseline_fallback': added_fallback,
        'n_buildings_total': int(len(df)), 'n_labeled': int(len(lab)),
        'n_train_labeled': int(len(tr)),
    },
    'window_profiles': profiles,
    'bridge_train': {
        'counts': cnt.astype(int).to_dict(orient='index'),
        'counts_row_normalized': cnt_rn.round(4).to_dict(orient='index'),
        'gfa_row_normalized': gfa_rn.round(4).to_dict(orient='index'),
        'row_totals_count': cnt.sum(axis=1).astype(int).to_dict(),
        'row_totals_gfa_m2': gfa.sum(axis=1).round(0).to_dict(),
    },
    'flagship_comparison': {
        'probe_3patch': probe_flag,
        'v05_train18_count': v05_flag_cnt,
        'v05_train18_gfa': v05_flag_gfa,
    },
    'findings': {
        'caoyang_A13': caoyang_check,
        'gov_dominant_windows': gov_dominant,
        'dev_dominant_windows': dev_dominant,
        'other_dominant_windows': sorted(set(profiles) - set(gov_dominant) - set(dev_dominant)),
        'zero_age_filled_windows': zero_age_summary,
    },
}
json.dump(results, open(V05 / 'results_e2.json', 'w'), indent=2, ensure_ascii=False)

# ================================================================ FIGURES
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})

# --- fig_e2_profiles: 4-column dot panel, additive non-exclusive weights ---
def short_name(name, maxlen=22):
    """Shorten window names at a word boundary (no mid-word truncation)."""
    if len(name) <= maxlen:
        return name
    cut = name[:maxlen]
    if ' ' in cut:
        cut = cut[:cut.rfind(' ')]
    return cut.rstrip(' /-')

order = sorted(profiles, key=lambda w: profiles[w]['profile']['government'])
ylab = [f"{w} {short_name(profiles[w]['name'])}" + (' *' if profiles[w]['role'] == 'heldout' else '')
        for w in order]
colors = {'government': '#b2182b', 'developer_capital': '#2166ac',
          'resident_community': '#1a9850', 'informal': '#8c510a'}

fig, axes = plt.subplots(1, 4, figsize=(13, 7.5), sharey=True)
ypos = np.arange(len(order))
for ax, d in zip(axes, DIMS):
    vals = np.array([profiles[w]['profile'][d] for w in order])
    lowc = np.array([profiles[w]['low_confidence_gfa_share'] > 0.5 for w in order])
    for y, v, lc in zip(ypos, vals, lowc):
        ax.hlines(y, 0, v, color=colors[d], lw=1.2, alpha=0.45)
        if lc:
            ax.plot(v, y, 'o', mfc='white', mec=colors[d], mew=1.4, ms=6)
        else:
            ax.plot(v, y, 'o', color=colors[d], ms=6)
    ax.set_title(d.replace('_', ' '))
    ax.set_xlim(0, 0.85)
    ax.grid(axis='x', alpha=0.3, lw=0.5)
    ax.set_xlabel('GFA-weighted prior weight')
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(ylab, fontsize=7.5)
# annotate unlabeled share > 20% on first panel
for y, w in zip(ypos, order):
    u = profiles[w]['unlabeled_gfa_share']
    if u > 0.20:
        axes[0].annotate(f'unlab {u:.0%}', xy=(0.02, y + 0.32), fontsize=7,
                         color='dimgray', style='italic')
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker='o', ls='', color='gray', label='filled: low-conf GFA share <= 0.5'),
           Line2D([], [], marker='o', ls='', mfc='white', mec='gray', label='open: low-conf GFA share > 0.5'),
           Line2D([], [], ls='', marker='$*$', color='black', label='* heldout window')]
fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=8)
fig.suptitle('v05 E2 — window power profiles (additive non-exclusive literature priors; sorted by government weight)',
             y=0.985)
fig.tight_layout(rect=[0, 0.045, 1, 0.97])
fig.savefig(FIG / 'fig_e2_profiles.png', dpi=140)
plt.close(fig)

# --- fig_e2_bridge: heatmap + flagship comparison panel ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2),
                               gridspec_kw={'width_ratios': [1.5, 1]})
M = cnt_rn.values
im = ax1.imshow(M, cmap='viridis', vmin=0, vmax=1, aspect='auto')
ax1.set_xticks(range(len(FAM5)))
ax1.set_xticklabels(FAM5, rotation=30, ha='right')
ax1.set_yticks(range(len(SEM8)))
ax1.set_yticklabels([f'{s} (n={int(cnt.sum(axis=1)[s])})' for s in SEM8], fontsize=8)
for i in range(len(SEM8)):
    for j in range(len(FAM5)):
        v = M[i, j]
        ax1.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7.5,
                 color='white' if v < 0.55 else 'black')
ax1.set_title('Semantic -> family bridge, row-normalized counts\n(18 TRAIN windows, labeled buildings)')
fig.colorbar(im, ax=ax1, shrink=0.85, label='P(family | semantic)')

labels = ['res_lowrise\n-> house', 'res_estate\n-> slab', 'res_estate\n-> midrise']
keys = list(probe_flag.keys())
x = np.arange(len(keys))
wd = 0.27
ax2.bar(x - wd, [probe_flag[k] for k in keys], wd, label='probe (3 patches)', color='#999999')
ax2.bar(x, [v05_flag_cnt[k] for k in keys], wd, label='v05 train18 (count)', color='#2166ac')
ax2.bar(x + wd, [v05_flag_gfa[k] for k in keys], wd, label='v05 train18 (GFA)', color='#92c5de')
for xi, k in zip(x, keys):
    for off, src in [(-wd, probe_flag), (0, v05_flag_cnt), (wd, v05_flag_gfa)]:
        ax2.text(xi + off, src[k] + 0.015, f'{src[k]:.2f}', ha='center', fontsize=7.5)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel('P(family | semantic)')
ax2.set_title('Flagship bridge cells: probe vs v05')
ax2.legend(fontsize=8, frameon=False)
ax2.grid(axis='y', alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(FIG / 'fig_e2_bridge.png', dpi=140)
plt.close(fig)

# ---------------------------------------------------------------- console
print('classes added as fallback:', added_fallback or 'none (all 8 in v01)')
print('Caoyang A13:', caoyang_check)
print('gov-dominant:', gov_dominant)
print('dev-dominant:', dev_dominant)
print('zero-age windows:', zero_age)
for w in zero_age:
    print(' ', w, zero_age_summary[w]['name'], 'no_age share',
          zero_age_summary[w]['res_estate_no_age_share'],
          'lowconf', zero_age_summary[w]['low_confidence_gfa_share'])
print('flagship probe :', probe_flag)
print('flagship v05cnt:', v05_flag_cnt)
print('flagship v05gfa:', v05_flag_gfa)
print('low-conf share range:',
      min(p['low_confidence_gfa_share'] for p in profiles.values()),
      max(p['low_confidence_gfa_share'] for p in profiles.values()))
print('done.')
