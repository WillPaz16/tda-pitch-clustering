"""
Development planner: the cheapest realistic path from a pitch's current shape
to a better-playing shape, given the pitcher's fixed profile, with
season-by-season benchmarks. ILLUSTRATIVE: built from tested pieces, but the
plan itself is not validated (that needs prospective data).

Pieces, each from an earlier result:
  Feasible population (profile held fixed). Other pitchers' 2025 shapes with
    the same arm slot (Statcast arm angle within ARM_TOL degrees) and
    extension (within EXT_TOL ft), velocity within VELO_TOL mph and raw spin
    within SPIN_TOL rpm. Release coordinates are not also constrained: they
    measure the same slot plus body size and rubber position (requiring both
    left only 11 shapes for the example, i.e. no path to plan). Grip and delivery mechanics are NOT modelled
    (no pronation/supination or delivery-marker data in Statcast: future work).
  Value, in SwStr% units. V(x) = gamma * pm(x) + beta * typ(x), with pm the
    physics-only model's predicted swinging-strike score and typ the
    typicality z-score; gamma and beta are the coefficients of the
    pre-specified typicality model (src/stuffplus/typicality.py), refit here
    on data/typicality.csv, primary seasons. Label fixed effects are left out:
    labels are conventions, shapes are what is compared.
  Steps. A K-nearest-neighbour graph on the feasible shapes in the
    standardized flight space (velocity relative to the pitcher's fastest
    shape, movement, spin rate, spin axis); every waypoint is a pitch someone
    throws. Edge cost = Euclidean distance.
  Benchmarks. One season's budget = the 90th percentile of observed
    one-season shape changes (data/pitch_development.csv, 'move'). For
    k = 1, 2, 3 seasons: the best-V shape reachable within k budgets, and the
    shortest path to it.

Run: python3 src/stuffplus/development_planner.py statcast_2025.parquet [pitcher_id pitch_type]
"""
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.neighbors import NearestNeighbors

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / 'src' / 'tda'))
from tda_classifier import load_tda_model, season_shapes
from label_sensitivity import FLIGHT

EXAMPLE = (543135, 'FF')     # Nathan Eovaldi's four-seamer, as in the pitch-design illustration
ARM_TOL, EXT_TOL, VELO_TOL, SPIN_TOL = 5.0, 0.5, 1.5, 200
K_GRAPH, SEASONS, BUDGET_Q = 8, 3, 0.9


def value_coefficients():
    t = pd.read_csv(_ROOT / 'data' / 'typicality.csv')
    t = t[t['season'].isin([2025, 2026])].dropna(subset=['swstr', 'pm', 'typ'])
    m = smf.wls("swstr ~ typ + pm + C(pitch_type) + C(season)", data=t, weights=t['n']).fit(
        cov_type='cluster', cov_kwds={'groups': t['pitcher']})
    return m.params['pm'], m.params['typ']


def main(path, key=EXAMPLE):
    gamma, beta = value_coefficients()
    budget = pd.read_csv(_ROOT / 'data' / 'pitch_development.csv')['move'].quantile(BUDGET_Q)

    df = pd.read_parquet(path)
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    raw, _ = season_shapes(df, list(mc['stuff_columns']))
    arm = df.groupby('pitcher')['arm_angle'].mean()
    names = df.drop_duplicates('pitcher').set_index('pitcher')['player_name']

    t = pd.read_csv(_ROOT / 'data' / 'typicality.csv')
    t = t[t['season'] == 2025].set_index(['pitcher', 'pitch_type'])
    t = t.join(raw[['release_pos_x', 'release_pos_z', 'release_extension', 'release_spin_rate', 'pfx_x', 'pfx_z',
                    'spin_cos', 'spin_sin']].add_prefix('raw_'), how='inner')
    t['arm'] = t.index.get_level_values(0).map(arm)
    t['V'] = gamma * t['pm'] + beta * t['typ']

    me = t.loc[key]
    ok = ((t.index.get_level_values(0) != key[0])
          & ((t['raw_release_extension'] - me['raw_release_extension']).abs() <= EXT_TOL)
          & ((t['arm'] - me['arm']).abs() <= ARM_TOL)
          & ((t['abs_speed'] - me['abs_speed']).abs() <= VELO_TOL)
          & ((t['raw_release_spin_rate'] - me['raw_release_spin_rate']).abs() <= SPIN_TOL))
    pop = pd.concat([t.loc[[key]], t[ok]])
    X = pop[FLIGHT].values
    dist, idx = NearestNeighbors(n_neighbors=min(K_GRAPH + 1, len(pop))).fit(X).kneighbors(X)
    G = nx.Graph()
    for i in range(len(pop)):
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            G.add_edge(i, j, weight=d)
    cost, paths = nx.single_source_dijkstra(G, 0, weight='weight')
    comp = len(nx.node_connected_component(G, 0))

    hb = lambda r: -r['raw_pfx_x'] * 12
    axis = lambda r: np.degrees(np.arctan2(r['raw_spin_sin'], r['raw_spin_cos'])) % 360
    desc = lambda i: (f"{names.get(pop.index[i][0], pop.index[i][0])} {pop.index[i][1]}: "
                      f"{pop.iloc[i]['abs_speed']:.1f} mph, HB {hb(pop.iloc[i]):.1f} in, IVB {12 * pop.iloc[i]['raw_pfx_z']:.1f} in, "
                      f"spin {pop.iloc[i]['raw_release_spin_rate']:.0f} rpm, axis {axis(pop.iloc[i]):.0f} deg, "
                      f"V {100 * pop.iloc[i]['V']:+.2f}, typ {pop.iloc[i]['typ']:+.2f}, actual SwStr {100 * pop.iloc[i]['swstr']:.1f}%")
    print(f"value: V = {gamma:.3f} * pm + {beta * 100:.2f} pp * typ   |   one-season budget {budget:.3f}")
    print(f"feasible population: {ok.sum()} shapes; {comp} reachable from the start through the graph")
    print("start:", desc(0))
    rows = []
    for k in range(1, SEASONS + 1):
        reach = [i for i, c in cost.items() if 0 < c <= k * budget + 1e-12]
        if not reach:
            print(f"season {k}: nothing reachable within {k} budgets"); continue
        best = max(reach, key=lambda i: pop.iloc[i]['V'])
        p = paths[best]
        assert abs(sum(G[a][b]['weight'] for a, b in zip(p, p[1:])) - cost[best]) < 1e-9
        gain = 100 * (pop.iloc[best]['V'] - pop.iloc[0]['V'])
        print(f"\nbenchmark after {k} season(s): value {gain:+.2f} pp SwStr vs start; path cost {cost[best]:.3f} "
              f"(straight line {np.linalg.norm(X[best] - X[0]):.3f}); {len(p) - 1} steps")
        for s, i in enumerate(p[1:], 1):
            print(f"   step {s}: {desc(i)}")
            rows.append({'season_budget': k, 'step': s, 'pitcher': pop.index[i][0], 'pitch_type': pop.index[i][1],
                         'name': names.get(pop.index[i][0], ''), 'cum_cost': cost[i], 'V': pop.iloc[i]['V'],
                         'typ': pop.iloc[i]['typ'], 'swstr': pop.iloc[i]['swstr'], 'mph': pop.iloc[i]['abs_speed'],
                         'hb_in': hb(pop.iloc[i]), 'ivb_in': 12 * pop.iloc[i]['raw_pfx_z'],
                         'spin_rpm': pop.iloc[i]['raw_release_spin_rate'], 'axis_deg': axis(pop.iloc[i])})
    out = pd.DataFrame(rows)
    out.to_csv(_ROOT / 'data' / f'development_plan_{key[0]}_{key[1]}.csv', index=False)
    pop.assign(cost=[cost.get(i, np.nan) for i in range(len(pop))]).to_csv(
        _ROOT / 'data' / f'development_population_{key[0]}_{key[1]}.csv')
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], (int(a[1]), a[2]) if len(a) >= 3 else EXAMPLE)
