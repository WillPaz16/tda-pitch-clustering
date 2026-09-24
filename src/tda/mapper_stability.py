"""
Topological stability of the Mapper graph, plus null models.

Refits Mapper on the saved (pitcher, pitch_type) archetypes under:
  grid       -- n_cubes x perc_overlap x eps around the chosen parameters
  bootstrap  -- pitchers resampled with replacement, chosen parameters
  null_gauss -- Gaussian with the archetypes' mean/covariance (no structure)
  null_shuffle -- each feature shuffled within pitch type (keeps per-type
                  marginals, destroys within-type joint structure)

Node IDs change between fits, so every tracked feature is defined by content:
  n_components / giant_frac -- connectivity
  slow_isolated -- of <70 mph archetypes covered by any node, fraction whose
                   nodes all lie outside the giant component
  bridge_frac   -- of the 5 highest-betweenness giant-component nodes, fraction
                   whose majority pitch type is SL or FC
  ordering_rho  -- Spearman rho of node mean velocity vs. position along the
                   shortest path from the most-CU node to the most-FF node

Null archetypes get pitch-type labels from their nearest real archetype.
"""
import warnings
warnings.filterwarnings("ignore")

import itertools
import sys
from collections import Counter
from pathlib import Path

import kmapper as km
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model

BASE = dict(n_cubes=10, perc_overlap=0.3, eps=1.0)
GRID = dict(n_cubes=[8, 10, 12], perc_overlap=[0.2, 0.3, 0.4], eps=[0.8, 1.0, 1.2])
N_BOOT, N_NULL, SEED = 30, 10, 42
SLOW_MPH, BRIDGE_TYPES, TOP_K, MIN_ANCHOR_SIZE = 70, {'SL', 'FC'}, 5, 10


def fit_graph(X_raw, n_cubes, perc_overlap, eps):
    X = StandardScaler().fit_transform(X_raw)
    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(X, projection=PCA(n_components=2))
    graph = mapper.map(lens, X, clusterer=DBSCAN(eps=eps, min_samples=4),
                       cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))
    G = nx.Graph()
    G.add_nodes_from(graph['nodes'])
    G.add_edges_from((s, t) for s, ts in graph['links'].items() for t in ts)
    return G, graph['nodes']


def features(G, nodes, speed, ptype):
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = comps[0] if comps else set()
    out = dict(n_nodes=G.number_of_nodes(), n_components=len(comps),
               giant_frac=len(giant) / max(G.number_of_nodes(), 1))

    node_of = {}
    for n, members in nodes.items():
        for m in members:
            node_of.setdefault(m, set()).add(n)
    slow = [i for i in np.flatnonzero(speed < SLOW_MPH) if i in node_of]
    out['slow_isolated'] = (np.mean([not (node_of[i] & giant) for i in slow]) if slow else np.nan)

    Gg = G.subgraph(giant)
    majority = {n: Counter(ptype[nodes[n]]).most_common(1)[0][0] for n in giant}
    btw = nx.betweenness_centrality(Gg)
    top = sorted(btw, key=btw.get, reverse=True)[:TOP_K]
    out['bridge_frac'] = np.mean([majority[n] in BRIDGE_TYPES for n in top]) if top else np.nan

    def anchor(t):
        big = [n for n in giant if len(nodes[n]) >= MIN_ANCHOR_SIZE]
        return max(big, key=lambda n: np.mean(ptype[nodes[n]] == t), default=None)
    a, b = anchor('CU'), anchor('FF')
    rho = np.nan
    if a and b and a != b:
        path = nx.shortest_path(Gg, a, b)
        if len(path) >= 3:
            rho = stats.spearmanr(range(len(path)), [speed[nodes[n]].mean() for n in path])[0]
    out['ordering_rho'] = rho
    return out


def main():
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    orig = mc['original_data']
    X_raw = orig[mc['stuff_columns']].values.astype(np.float64)
    speed = orig['release_speed'].values
    ptype = orig.index.get_level_values('pitch_type').values.astype(str)
    pitchers = orig.index.get_level_values('pitcher').values
    rng = np.random.default_rng(SEED)
    rows = []

    def record(kind, Xr, sp, pt, **params):
        G, nodes = fit_graph(Xr, **params)
        rows.append(dict(kind=kind, **params, **features(G, nodes, sp, pt)))

    # saved model, as fitted (sanity reference)
    G0 = nx.Graph()
    G0.add_nodes_from(mc['graph']['nodes'])
    G0.add_edges_from((s, t) for s, ts in mc['graph']['links'].items() for t in ts)
    rows.append(dict(kind='saved_model', **BASE, **features(G0, mc['graph']['nodes'], speed, ptype)))

    for vals in itertools.product(*GRID.values()):
        record('grid', X_raw, speed, ptype, **dict(zip(GRID, vals)))

    uniq = np.unique(pitchers)
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(pitchers == p) for p in pick])
        record('bootstrap', X_raw[idx], speed[idx], ptype[idx], **BASE)

    Xs_real = StandardScaler().fit(X_raw)
    for _ in range(N_NULL):
        Xg = rng.multivariate_normal(X_raw.mean(0), np.cov(X_raw, rowvar=False), size=len(X_raw))
        nn = np.argmin(cdist(Xs_real.transform(Xg), Xs_real.transform(X_raw)), axis=1)
        record('null_gauss', Xg, Xg[:, 0], ptype[nn], **BASE)

        Xsh = X_raw.copy()
        for t in np.unique(ptype):
            ix = np.flatnonzero(ptype == t)
            for j in range(Xsh.shape[1]):
                Xsh[ix, j] = rng.permutation(Xsh[ix, j])
        record('null_shuffle', Xsh, Xsh[:, 0], ptype, **BASE)

    res = pd.DataFrame(rows)
    out = _ROOT / 'data' / 'mapper_stability.csv'
    res.to_csv(out, index=False)

    pd.set_option('display.width', 160)
    print(res[res.kind == 'saved_model'].to_string(index=False))
    base_grid = res[(res.kind == 'grid') & (res.n_cubes == BASE['n_cubes'])
                    & (res.perc_overlap == BASE['perc_overlap']) & (res.eps == BASE['eps'])]
    print('\nGrid point at chosen parameters (should match saved model):')
    print(base_grid.to_string(index=False))

    summ = res[res.kind != 'saved_model'].groupby('kind').agg(
        fits=('n_nodes', 'size'), nodes=('n_nodes', 'median'), components=('n_components', 'median'),
        slow_isolated_med=('slow_isolated', 'median'),
        slow_isolated_ge_half=('slow_isolated', lambda s: np.mean(s.dropna() >= 0.5) if s.notna().any() else np.nan),
        bridge_med=('bridge_frac', 'median'),
        bridge_ge_60pct=('bridge_frac', lambda s: np.mean(s.dropna() >= 0.6)),
        rho_med=('ordering_rho', 'median'),
        rho_ge_07=('ordering_rho', lambda s: np.mean(s.dropna() >= 0.7)),
    )
    print('\nPersistence summary (fraction of fits in which each feature holds):')
    print(summ.round(2).to_string())
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
