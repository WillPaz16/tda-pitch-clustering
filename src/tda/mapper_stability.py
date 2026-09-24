"""
Topological stability of the Mapper graph, plus null models.

Refits Mapper on the saved (pitcher, pitch_type) archetypes under:
  grid       -- n_cubes x perc_overlap x eps around the chosen parameters
  subsample  -- 80% of pitchers drawn without replacement, chosen parameters
  null_gauss -- Gaussian with the archetypes' mean/covariance (no structure)
  null_shuffle -- each feature shuffled within pitch type (keeps per-type
                  marginals, destroys within-type joint structure)

Every dataset above is run under each lens: pca (the saved model's),
baseball (velocity x induced vertical break), isomap (2-D, nonlinear), and
density (1-D mean distance to 10 nearest neighbors).

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
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model

BASE = dict(n_cubes=10, perc_overlap=0.3, eps=1.0)
GRID = dict(n_cubes=[8, 10, 12], perc_overlap=[0.2, 0.3, 0.4], eps=[0.8, 1.0, 1.2])
N_BOOT, N_NULL, SEED, SUBSAMPLE_FRAC = 30, 10, 42, 0.8
SLOW_MPH, BRIDGE_TYPES, TOP_K, MIN_ANCHOR_SIZE = 70, {'SL', 'FC'}, 5, 10
LENSES, K_NN = ['pca', 'baseball', 'isomap', 'density'], 10
SPEED_COL, IVB_COL = 0, 2   # release_speed, pfx_z in stuff_columns (checked in main)


def gaussian_null(X_raw, rng):
    """Same mean/covariance as the archetypes, no other structure."""
    return rng.multivariate_normal(X_raw.mean(0), np.cov(X_raw, rowvar=False), size=len(X_raw))


def shuffle_within_type_null(X_raw, ptype, rng, keep_together=()):
    """Each feature permuted independently within pitch type: keeps per-type
    marginals, destroys the joint structure inside each type.
    keep_together: column groups permuted as a unit (e.g. the spin_cos /
    spin_sin pair, which encodes one angle and must stay on the circle)."""
    Xsh = X_raw.copy()
    groups = [list(g) for g in keep_together]
    grouped = {j for g in groups for j in g}
    groups += [[j] for j in range(X_raw.shape[1]) if j not in grouped]
    for t in np.unique(ptype):
        ix = np.flatnonzero(ptype == t)
        for g in groups:
            Xsh[ix[:, None], g] = X_raw[rng.permutation(ix)[:, None], g]
    return Xsh


def make_lens(X, name):
    """Lens on standardized features X, min-max scaled to [0, 1] like
    KeplerMapper's default. 'pca' is the saved model's lens."""
    if name == 'pca':
        return km.KeplerMapper(verbose=0).fit_transform(X, projection=PCA(n_components=2))
    if name == 'baseball':      # velocity x induced vertical break
        L = X[:, [SPEED_COL, IVB_COL]]
    elif name == 'isomap':
        L = Isomap(n_neighbors=K_NN, n_components=2).fit_transform(X)
    elif name == 'density':     # mean distance to k nearest neighbors (1-D)
        d, _ = NearestNeighbors(n_neighbors=K_NN + 1).fit(X).kneighbors(X)
        L = d[:, 1:].mean(axis=1, keepdims=True)
    return MinMaxScaler().fit_transform(L)


def fit_graph(X, lens, n_cubes, perc_overlap, eps):
    graph = km.KeplerMapper(verbose=0).map(
        lens, X, clusterer=DBSCAN(eps=eps, min_samples=4),
        cover=km.Cover(n_cubes=n_cubes, perc_overlap=perc_overlap))
    G = nx.Graph()
    G.add_nodes_from(graph['nodes'])
    G.add_edges_from((s, t) for s, ts in graph['links'].items() for t in ts)
    return G, graph['nodes']


def top_betweenness(G, nodes, ptype, k=TOP_K):
    """The k highest-betweenness giant-component nodes, with each node's
    majority pitch type. Bridge nodes = those whose majority is SL/FC."""
    giant = max(nx.connected_components(G), key=len, default=set())
    btw = nx.betweenness_centrality(G.subgraph(giant))
    top = sorted(btw, key=btw.get, reverse=True)[:k]
    return [(n, Counter(ptype[nodes[n]]).most_common(1)[0][0]) for n in top]


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
    top = top_betweenness(G, nodes, ptype)
    out['bridge_frac'] = np.mean([t in BRIDGE_TYPES for _, t in top]) if top else np.nan

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
    assert [mc['stuff_columns'][c] for c in (SPEED_COL, IVB_COL)] == ['release_speed', 'pfx_z']
    rng = np.random.default_rng(SEED)

    # draw every dataset once, then apply each lens to the same draws, so
    # lenses are compared on identical subsamples and null realizations
    datasets = [('grid', X_raw, speed, ptype,
                 [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())])]

    # subsample pitchers WITHOUT replacement: resampling with replacement
    # duplicates points, which inflates DBSCAN density (~2x the nodes)
    uniq = np.unique(pitchers)
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=int(SUBSAMPLE_FRAC * len(uniq)), replace=False)
        idx = np.flatnonzero(np.isin(pitchers, pick))
        datasets.append(('subsample', X_raw[idx], speed[idx], ptype[idx], [BASE]))

    Xs_real = StandardScaler().fit(X_raw)
    for _ in range(N_NULL):
        Xg = gaussian_null(X_raw, rng)
        nn = np.argmin(cdist(Xs_real.transform(Xg), Xs_real.transform(X_raw)), axis=1)
        datasets.append(('null_gauss', Xg, Xg[:, SPEED_COL], ptype[nn], [BASE]))

        Xsh = shuffle_within_type_null(X_raw, ptype, rng)
        datasets.append(('null_shuffle', Xsh, Xsh[:, SPEED_COL], ptype, [BASE]))

    # saved model, as fitted (sanity reference)
    G0 = nx.Graph()
    G0.add_nodes_from(mc['graph']['nodes'])
    G0.add_edges_from((s, t) for s, ts in mc['graph']['links'].items() for t in ts)
    rows = [dict(lens='pca', kind='saved_model', **BASE, **features(G0, mc['graph']['nodes'], speed, ptype))]

    for lens_name in LENSES:
        for kind, Xr, sp, pt, params_list in datasets:
            X = StandardScaler().fit_transform(Xr)
            lens = make_lens(X, lens_name)
            for p in params_list:
                G, nodes = fit_graph(X, lens, **p)
                rows.append(dict(lens=lens_name, kind=kind, **p, **features(G, nodes, sp, pt)))
        print(f"lens {lens_name} done", flush=True)

    res = pd.DataFrame(rows)
    out = _ROOT / 'data' / 'mapper_stability.csv'
    res.to_csv(out, index=False)

    pd.set_option('display.width', 160)
    print(res[res.kind == 'saved_model'].to_string(index=False))
    base_grid = res[(res.lens == 'pca') & (res.kind == 'grid') & (res.n_cubes == BASE['n_cubes'])
                    & (res.perc_overlap == BASE['perc_overlap']) & (res.eps == BASE['eps'])]
    print('\nGrid point at chosen parameters (should match saved model):')
    print(base_grid.to_string(index=False))

    summ = res[res.kind != 'saved_model'].groupby(['lens', 'kind'], sort=False).agg(
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
