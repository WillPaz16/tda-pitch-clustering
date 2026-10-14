"""
Pitch-design neighborhoods: an illustrative use of the Mapper graph.

For one pitcher's pitch (a training archetype), its neighborhood is every
archetype in the pitch's own Mapper node(s) plus the graph-adjacent nodes,
kept only if it comes from another pitcher with a similar delivery (release
side/height and extension within DELIVERY_TOL ft, in the handedness-mirrored
frame the Mapper model uses), similar velocity and raw spin rate (within
VELO_TOL mph, SPIN_TOL rpm) and movement within MOVE_TOL inches on each axis.
Those are realistic nearby shapes: pitches other big-leaguers actually throw
from roughly the same arm slot with roughly the same velocity and spin,
differing modestly in movement and spin axis -- the part pitch design (grip,
seam orientation) can change; velocity and raw spin are largely arm talent. The tolerances are our choice, not an
industry standard. Without them, a node neighborhood mixes e.g. a soft-
tosser's four-seamer with hard changeups at the same velocity (the continuum
at work, since nodes at this cover resolution are coarse), and the 'better'
neighbors are just a different pitch.

Each neighbor is scored with the leak-free CatBoost models in models/
(trained on 2022-2024 out-of-fold, so 2025 is out-of-sample): the mean
per-pitch predicted xwOBA, swinging-strike and chase probabilities. Because
neighbors carry different pitch-type labels, the composite is z-scored
POOLED across all archetypes (equal weights, as in weights_used), not per
pitch type like production Stuff+. Actual 2025 SwStr%, Whiff% and Chase% are
attached as a sanity check on the model ranking.

ILLUSTRATIVE, NOT CAUSAL: a better-scoring neighbor shows a shape that exists
near this one, not that this pitcher can throw it or that it would play the
same in their arsenal (tunneling, sequencing, location are all outside the
model). Run: python3 src/tda/pitch_design_neighborhood.py [statcast.parquet]
"""

import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / 'src' / 'stuffplus'))
from tda_classifier import load_tda_model, fetch_savant_csv
from pitcher_repertoire_overlap import build_networkx_graph
from stuff_plus_calculator import StuffPlusCalculator

DELIVERY_TOL = 0.5     # ft, on release_pos_x, release_pos_z, release_extension
VELO_TOL = 1.5         # mph
SPIN_TOL = 200         # rpm
MOVE_TOL = 4.0 / 12    # ft (= 4 in), on pfx_x and pfx_z
MIN_TARGET = 300       # pitches for the pitch being improved
MIN_NEIGHBOR = 150     # pitches for each neighbor
MIN_NEIGHBORS = 8      # qualifying neighbors needed to call it a neighborhood

# clearest case found by this script + the tolerance sweep (DISCOVERY_FINDINGS.md):
# Nathan Eovaldi's four-seamer, last in its neighborhood at every setting
EXAMPLE = (543135, 'FF')

MISS = ['swinging_strike', 'swinging_strike_blocked', 'missed_bunt']
SWING = MISS + ['foul', 'foul_bunt', 'foul_tip', 'bunt_foul_tip', 'hit_into_play']


def score_archetypes(df):
    """Per (pitcher, pitch_type): n, model predictions, actual 2025 rates."""
    calc = StuffPlusCalculator()
    X = calc._build_features(df)
    out = df[['pitcher', 'pitch_type', 'player_name', 'p_throws', 'description']].copy()
    for name in ['xwoba', 'miss', 'chase']:
        feats = calc.metadata[name]['features']
        out['pred_' + name] = calc._predict_safe(calc.models[name], X.reindex(columns=feats, fill_value=0))

    # same zone/swing definitions as the chase/miss training targets
    in_zone = (df['plate_x'].abs() <= 8.5 / 12) & df['plate_z'].between(df['sz_bot'], df['sz_top'])
    out['swing'] = df['description'].isin(SWING)
    out['miss'] = df['description'].isin(MISS)
    out['ozone'] = ~in_zone.fillna(False)
    out['chase'] = out['swing'] & out['ozone']

    g = out.groupby(['pitcher', 'pitch_type'])
    a = g.agg(name=('player_name', 'first'), throws=('p_throws', 'first'), n=('miss', 'size'),
              pred_xwoba=('pred_xwoba', 'mean'), pred_miss=('pred_miss', 'mean'),
              pred_chase=('pred_chase', 'mean'), swstr=('miss', 'mean'),
              swings=('swing', 'sum'), misses=('miss', 'sum'),
              ozone=('ozone', 'sum'), chases=('chase', 'sum'))
    a['whiff'] = a['misses'] / a['swings'].where(a['swings'] > 0)
    a['chase_pct'] = a['chases'] / a['ozone'].where(a['ozone'] > 0)

    # pooled composite: higher = better for the pitcher (low xwOBA, more miss/chase)
    q = a[a['n'] >= MIN_NEIGHBOR]
    z = lambda c, s: s * (a[c] - q[c].mean()) / q[c].std(ddof=0)
    a['comp_z'] = (z('pred_xwoba', -1) + z('pred_miss', 1) + z('pred_chase', 1)) / 3
    a['score'] = 100 + 15 * (a['comp_z'] - a.loc[q.index, 'comp_z'].mean()) / a.loc[q.index, 'comp_z'].std(ddof=0)
    return a


def physical(arch):
    """Archetype features in baseball units (mirrored frame: + HB = arm side)."""
    return pd.DataFrame({
        'mph': arch['release_speed'],
        'HB_in': -arch['pfx_x'] * 12,        # mirrored RHP frame: arm side is -x
        'IVB_in': arch['pfx_z'] * 12,
        'spin_rpm': arch['release_spin_rate'],
        'axis_deg': np.degrees(np.arctan2(arch['spin_sin'], arch['spin_cos'])) % 360,
        'ext_ft': arch['release_extension'],
        'rel_x_ft': arch['release_pos_x'],
        'rel_z_ft': arch['release_pos_z'],
    }, index=arch.index)


def neighborhoods(mc, scores):
    graph, arch = mc['graph'], mc['original_data']
    G = build_networkx_graph(graph)
    node_of = {}
    for nid, members in graph['nodes'].items():
        for m in members:
            node_of.setdefault(m, set()).add(nid)
    keys = list(arch.index)
    sc = scores.reindex(keys)
    rel = arch[['release_pos_x', 'release_pos_z', 'release_extension']].values
    velo = arch['release_speed'].values
    spin = arch['release_spin_rate'].values.astype(float)
    move = arch[['pfx_x', 'pfx_z']].values.astype(float)

    rows = []
    for i, key in enumerate(keys):
        if not sc['n'].iloc[i] >= MIN_TARGET or i not in node_of:
            continue
        nodes = set(node_of[i])
        for nid in node_of[i]:
            nodes |= set(G.neighbors(nid))
        cand = {m for nid in nodes for m in graph['nodes'][nid]} - {i}
        cand = [j for j in cand
                if keys[j][0] != key[0]
                and np.all(np.abs(rel[j] - rel[i]) <= DELIVERY_TOL)
                and abs(velo[j] - velo[i]) <= VELO_TOL
                and abs(spin[j] - spin[i]) <= SPIN_TOL
                and np.all(np.abs(move[j] - move[i]) <= MOVE_TOL)
                and sc['n'].iloc[j] >= MIN_NEIGHBOR]
        if len(cand) < MIN_NEIGHBORS:
            continue
        nb = sc.iloc[cand]
        rho = spearmanr(nb['score'], nb['swstr']).correlation
        pct = (nb['score'] < sc['score'].iloc[i]).mean()
        top = nb[nb['score'] >= nb['score'].quantile(2 / 3)]
        rows.append({'pitcher': key[0], 'pitch_type': key[1], 'name': sc['name'].iloc[i],
                     'n': int(sc['n'].iloc[i]), 'n_nodes': len(nodes), 'n_neighbors': len(cand),
                     'score': sc['score'].iloc[i], 'pct_in_hood': pct,
                     'top_score': top['score'].median(), 'swstr': sc['swstr'].iloc[i],
                     'top_swstr': top['swstr'].median(), 'rho_score_swstr': rho,
                     'neighbor_idx': cand})
    return pd.DataFrame(rows)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = pd.read_parquet(path) if path else fetch_savant_csv('2025-03-28', '2025-11-04')
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    scores = score_archetypes(df)
    hoods = neighborhoods(mc, scores)

    # clearest cases: a pitch near the bottom of its own neighborhood, where the
    # model ranking agrees with actual swinging strikes inside that neighborhood
    hoods['gap'] = hoods['top_score'] - hoods['score']
    hoods['swstr_gap'] = hoods['top_swstr'] - hoods['swstr']
    clear = hoods[(hoods['pct_in_hood'] <= 0.25) & (hoods['rho_score_swstr'] >= 0.4)
                  & (hoods['swstr_gap'] > 0)].sort_values('gap', ascending=False)
    cols = ['name', 'pitch_type', 'n', 'n_neighbors', 'score', 'top_score', 'swstr', 'top_swstr', 'rho_score_swstr']
    print(f"{len(hoods)} pitches with a neighborhood; {len(clear)} pass the clarity filter\n")
    print(clear[cols].head(15).round(3).to_string())

    # the example's neighborhood, in baseball units, for the deck figure
    keys = list(mc['original_data'].index)
    r = hoods[(hoods['pitcher'] == EXAMPLE[0]) & (hoods['pitch_type'] == EXAMPLE[1])].iloc[0]
    idx = [keys.index(EXAMPLE)] + list(r['neighbor_idx'])
    ex = physical(mc['original_data'].astype(float)).iloc[idx]
    ex = scores.reindex(ex.index)[['name', 'n', 'score', 'swstr', 'whiff', 'chase_pct', 'misses']].join(ex)
    d = ex[['mph', 'spin_rpm', 'HB_in', 'IVB_in']]
    d = d.drop(EXAMPLE) - d.loc[EXAMPLE]
    assert (d['mph'].abs() <= VELO_TOL).all() and (d['spin_rpm'].abs() <= SPIN_TOL).all()
    assert (d[['HB_in', 'IVB_in']].abs() <= MOVE_TOL * 12 + 1e-9).all().all()
    ex.to_csv(_ROOT / 'data' / 'pitch_design_example.csv')

    out = _ROOT / 'data' / 'pitch_design_neighborhoods.csv'
    hoods.drop(columns='neighbor_idx').to_csv(out, index=False)
    scores.to_csv(_ROOT / 'data' / 'archetype_model_scores_2025.csv')
    print(f"\nSaved {out}")
    return mc, scores, hoods


if __name__ == "__main__":
    main()
