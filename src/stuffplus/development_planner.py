"""
Development planner: the cheapest realistic path from a pitch's current shape
to a better-playing shape, given the pitcher's fixed profile, with
season-by-season benchmarks. ILLUSTRATIVE: built from tested pieces, but the
plan itself is not validated (that needs prospective data).

Pieces, each from an earlier result:
  Feasible population (profile held fixed). Other pitchers' same-season
    shapes with the same arm slot (Statcast arm angle within ARM_TOL degrees)
    and extension (within EXT_TOL ft), velocity within VELO_TOL mph and raw
    spin within SPIN_TOL rpm. Release coordinates are not also constrained:
    they measure the same slot plus body size and rubber position (requiring
    both left only 11 shapes for the Eovaldi example, i.e. no path to plan).
    Grip and delivery mechanics are NOT modelled (no pronation/supination or
    delivery-marker data in Statcast: future work).
  Value, in SwStr% units. V(x) = gamma * pm(x) + beta * typ(x), with pm the
    physics-only model's predicted swinging-strike score and typ the
    typicality z-score; gamma and beta are the coefficients of the
    pre-specified typicality model (src/stuffplus/typicality.py), refit here
    on data/typicality.csv, primary seasons. Label fixed effects are left out:
    labels are conventions, shapes are what is compared.
  Steps. A K-nearest-neighbour graph on the feasible shapes in the
    standardized flight space (velocity relative to the pitcher's fastest
    shape, movement, spin rate, spin axis); every waypoint is a pitch someone
    throws. Edge cost = WEIGHTED Euclidean distance: velocity and raw spin
    rate (mostly arm talent) cost ARM_WEIGHT times as much per unit as
    movement and spin axis (grip and seam work), so plans move through shape
    and use velocity or spin only when worth it.
  Destination and benchmarks. One season's budget = the 90th percentile of
    observed one-season shape changes, measured with the same weighted
    distance (consecutive-season shapes in data/typicality.csv).
    The destination is the best-V shape reachable within SEASONS budgets;
    the benchmarks are milestones ALONG that one shortest path: after season
    k, the furthest waypoint whose cumulative cost is within k budgets.
    (Choosing a separate best shape per season gave destinations in
    different directions, which is not a development plan.) Whether value
    rises monotonically along the path is reported.

Young-arm search (--young SEASON): every pitch thrown >= YOUNG_MIN_N times by
a pitcher aged <= YOUNG_MAX_AGE (Statcast age_pit) is planned; a case is kept
as "clear" only if the feasible population has >= 30 shapes, the one-season
plan has >= 2 steps, and the one-season gain stays positive with arm angle
loosened to 7 degrees and with the budget cut to the 75th percentile.

Run: python3 src/stuffplus/development_planner.py statcast_<season>.parquet [pitcher_id pitch_type [season]]
     python3 src/stuffplus/development_planner.py statcast_<season>.parquet --young <season>
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

EXAMPLE = (543135, 'FF', 2025)     # Nathan Eovaldi's four-seamer, as in the pitch-design illustration
ARM_TOL, EXT_TOL, VELO_TOL, SPIN_TOL = 5.0, 0.5, 1.5, 200
K_GRAPH, SEASONS, BUDGET_Q = 8, 3, 0.9
ARM_WEIGHT = 3.0
WEIGHTS = np.array([ARM_WEIGHT if c in ('release_speed', 'release_spin_rate') else 1.0 for c in FLIGHT])
YOUNG_MAX_AGE, YOUNG_MIN_N = 25, 300


def value_coefficients():
    t = pd.read_csv(_ROOT / 'data' / 'typicality.csv')
    t = t[t['season'].isin([2025, 2026])].dropna(subset=['swstr', 'pm', 'typ'])
    m = smf.wls("swstr ~ typ + pm + C(pitch_type) + C(season)", data=t, weights=t['n']).fit(
        cov_type='cluster', cov_kwds={'groups': t['pitcher']})
    return m.params['pm'], m.params['typ']


def budget_at(q):
    """q-quantile of observed one-season shape changes, in the weighted metric."""
    t = pd.read_csv(_ROOT / 'data' / 'typicality.csv').set_index(['pitcher', 'pitch_type', 'season'])
    moves = []
    for y in sorted(t.index.get_level_values('season').unique())[:-1]:
        a = t.xs(y, level='season')[FLIGHT]
        b = t.xs(y + 1, level='season')[FLIGHT]
        common = a.index.intersection(b.index)
        moves.append(np.linalg.norm((b.loc[common].values - a.loc[common].values) * WEIGHTS, axis=1))
    return np.quantile(np.concatenate(moves), q)


def load_season(path, season):
    """Shapes of one season with raw features, arm angle, value V; plus names and ages."""
    gamma, beta = value_coefficients()
    df = pd.read_parquet(path)
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    raw, _ = season_shapes(df, list(mc['stuff_columns']))
    t = pd.read_csv(_ROOT / 'data' / 'typicality.csv')
    t = t[t['season'] == season].set_index(['pitcher', 'pitch_type'])
    t = t.join(raw[['release_pos_x', 'release_pos_z', 'release_extension', 'release_spin_rate', 'pfx_x', 'pfx_z',
                    'spin_cos', 'spin_sin']].add_prefix('raw_'), how='inner')
    t['arm'] = t.index.get_level_values(0).map(df.groupby('pitcher')['arm_angle'].mean())
    t['V'] = gamma * t['pm'] + beta * t['typ']
    names = df.drop_duplicates('pitcher').set_index('pitcher')['player_name']
    ages = df.groupby('pitcher')['age_pit'].max() if 'age_pit' in df.columns else pd.Series(dtype=float)
    return t, names, ages, (gamma, beta)


def plan(t, key, budget, arm_tol=ARM_TOL):
    """Feasible population, graph, and per-season benchmarks for one pitch."""
    me = t.loc[key]
    ok = ((t.index.get_level_values(0) != key[0])
          & ((t['raw_release_extension'] - me['raw_release_extension']).abs() <= EXT_TOL)
          & ((t['arm'] - me['arm']).abs() <= arm_tol)
          & ((t['abs_speed'] - me['abs_speed']).abs() <= VELO_TOL)
          & ((t['raw_release_spin_rate'] - me['raw_release_spin_rate']).abs() <= SPIN_TOL))
    pop = pd.concat([t.loc[[key]], t[ok]])
    X = pop[FLIGHT].values * WEIGHTS
    G = nx.Graph()
    G.add_node(0)
    if len(pop) > 1:
        dist, idx = NearestNeighbors(n_neighbors=min(K_GRAPH + 1, len(pop))).fit(X).kneighbors(X)
        for i in range(len(pop)):
            for d, j in zip(dist[i, 1:], idx[i, 1:]):
                G.add_edge(i, j, weight=d)
    cost, paths = nx.single_source_dijkstra(G, 0, weight='weight')
    reach = [i for i, c in cost.items() if 0 < c <= SEASONS * budget + 1e-12]
    if not reach:
        return pop, cost, None, int(ok.sum())
    dest = max(reach, key=lambda i: pop.iloc[i]['V'])
    path = paths[dest]
    assert abs(sum(G[a][b]['weight'] for a, b in zip(path, path[1:])) - cost[dest]) < 1e-9
    V = pop['V'].values
    miles = [max([i for i in path if cost[i] <= k * budget + 1e-12], key=lambda i: cost[i]) for k in range(1, SEASONS + 1)]
    return pop, cost, {'dest': dest, 'path': path, 'gain': V[dest] - V[0], 'milestones': miles,
                       'milestone_gain': [V[m] - V[0] for m in miles],
                       'monotone': bool(np.all(np.diff(V[path]) >= -1e-12))}, int(ok.sum())


def describe(pop, i, names):
    r = pop.iloc[i]
    return (f"{names.get(pop.index[i][0], pop.index[i][0])} {pop.index[i][1]}: {r['abs_speed']:.1f} mph, "
            f"HB {-12 * r['raw_pfx_x']:.1f} in, IVB {12 * r['raw_pfx_z']:.1f} in, spin {r['raw_release_spin_rate']:.0f} rpm, "
            f"axis {np.degrees(np.arctan2(r['raw_spin_sin'], r['raw_spin_cos'])) % 360:.0f} deg, V {100 * r['V']:+.2f}, "
            f"typ {r['typ']:+.2f}, actual SwStr {100 * r['swstr']:.1f}%")


def main(path, key=EXAMPLE[:2], season=EXAMPLE[2]):
    t, names, ages, (gamma, beta) = load_season(path, season)
    budget = budget_at(BUDGET_Q)
    pop, cost, b, n_feas = plan(t, key, budget)
    print(f"value: V = {gamma:.3f} * pm + {beta * 100:.2f} pp * typ   |   one-season budget {budget:.3f}")
    print(f"feasible population: {n_feas} shapes; {len(cost)} reachable from the start through the graph")
    print("start:", describe(pop, 0, names))
    rows = []
    if b is None:
        print("nothing reachable")
    else:
        p = b['path']
        print(f"\ndestination: value {100 * b['gain']:+.2f} pp SwStr vs start; path cost {cost[b['dest']]:.3f} "
              f"(straight line {np.linalg.norm((pop[FLIGHT].values[b['dest']] - pop[FLIGHT].values[0]) * WEIGHTS):.3f}); "
              f"{len(p) - 1} steps; value monotone along path: {b['monotone']}")
        for k, (m, g) in enumerate(zip(b['milestones'], b['milestone_gain']), 1):
            print(f"   benchmark after season {k}: step {p.index(m)} ({100 * g:+.2f} pp)")
        for s, i in enumerate(p[1:], 1):
            print(f"   step {s}: {describe(pop, i, names)}")
            r = pop.iloc[i]
            k = next(k for k, m in enumerate(b['milestones'], 1) if cost[i] <= cost[m] + 1e-12)
            rows.append({'season_budget': k, 'step': s, 'pitcher': pop.index[i][0], 'pitch_type': pop.index[i][1],
                         'name': names.get(pop.index[i][0], ''), 'cum_cost': cost[i], 'V': r['V'], 'typ': r['typ'],
                         'swstr': r['swstr'], 'mph': r['abs_speed'], 'hb_in': -12 * r['raw_pfx_x'],
                         'ivb_in': 12 * r['raw_pfx_z'], 'spin_rpm': r['raw_release_spin_rate'],
                         'axis_deg': np.degrees(np.arctan2(r['raw_spin_sin'], r['raw_spin_cos'])) % 360})
    pd.DataFrame(rows).to_csv(_ROOT / 'data' / f'development_plan_{key[0]}_{key[1]}.csv', index=False)
    pop.assign(cost=[cost.get(i, np.nan) for i in range(len(pop))]).to_csv(
        _ROOT / 'data' / f'development_population_{key[0]}_{key[1]}.csv')


def young_search(path, season):
    t, names, ages, _ = load_season(path, season)
    b90, b75 = budget_at(BUDGET_Q), budget_at(0.75)
    rows = []
    for key in t.index:
        if not (ages.get(key[0], 99) <= YOUNG_MAX_AGE and t.loc[key, 'n'] >= YOUNG_MIN_N):
            continue
        pop, cost, b, n_feas = plan(t, key, b90)
        if b is None:
            continue
        _, _, loose, _ = plan(t, key, b90, arm_tol=7.0)
        _, _, tight, _ = plan(t, key, b75)
        d_mph = pop.iloc[b['dest']]['abs_speed'] - pop.iloc[0]['abs_speed']
        d_typ = pop.iloc[b['dest']]['typ'] - pop.iloc[0]['typ']
        rows.append({'pitcher': key[0], 'pitch_type': key[1], 'name': names.get(key[0], ''), 'age': ages.get(key[0]),
                     'n': int(t.loc[key, 'n']), 'population': n_feas, 'steps': len(b['path']) - 1,
                     'gain': 100 * b['gain'], 'gain_season1': 100 * b['milestone_gain'][0], 'monotone': b['monotone'],
                     'd_mph': d_mph, 'd_typ': d_typ,
                     'gain_arm7': 100 * loose['gain'] if loose else np.nan,
                     'gain_b75': 100 * tight['gain'] if tight else np.nan,
                     'V': 100 * t.loc[key, 'V'], 'swstr': 100 * t.loc[key, 'swstr']})
    r = pd.DataFrame(rows)
    r['clear'] = ((r['population'] >= 30) & (r['steps'] >= 2) & r['monotone']
                  & (r['gain_arm7'] > 0) & (r['gain_b75'] > 0))
    r.to_csv(_ROOT / 'data' / f'development_young_{season}.csv', index=False)
    print(f"{len(r)} young-arm pitches planned; {r['clear'].sum()} clear cases (>= 30 feasible, >= 2 steps, "
          f"value monotone along the path, gain robust to arm angle 7 deg and to the 75th-percentile budget)")
    print(r[r['clear']].sort_values('gain', ascending=False).head(12).round(2).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[1] == '--young':
        young_search(a[0], int(a[2]))
    elif len(a) >= 3:
        main(a[0], (int(a[1]), a[2]), int(a[3]) if len(a) >= 4 else EXAMPLE[2])
    else:
        main(a[0])
