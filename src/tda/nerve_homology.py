"""
First Betti number of the Mapper graph vs. the full Mapper nerve.

KeplerMapper returns only the 1-skeleton (a graph) of the nerve of the
pulled-back cover. With a 2-D lens, three or more cover elements routinely
share points, so the nerve has 2-simplices that fill many of the graph's
cycles. This computes, for each fit:
  graph_b1 = E - V + C                    (1-skeleton)
  nerve_b1 = E - V + C - rank(d2)         (2-skeleton; d2 = triangle boundary map)
Higher simplices do not change b1. Rank is over the rationals.

Runs on the parameter grid and on null models from mapper_stability.py.
"""
import warnings
warnings.filterwarnings("ignore")

import itertools
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model
import mapper_stability as ms

N_NULL, SEED = 5, 42


def betti1(nodes):
    """(graph_b1, nerve_b1) for Mapper nodes {node_id: member indices}."""
    nodes = {k: set(v) for k, v in nodes.items()}
    V = sorted(nodes)
    E = [p for p in itertools.combinations(V, 2) if nodes[p[0]] & nodes[p[1]]]
    T = [t for t in itertools.combinations(V, 3) if nodes[t[0]] & nodes[t[1]] & nodes[t[2]]]
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    graph_b1 = len(E) - len(V) + nx.number_connected_components(G)
    ei = {e: i for i, e in enumerate(E)}
    d2 = np.zeros((len(E), len(T)))
    for j, (a, b, c) in enumerate(T):     # d[a,b,c] = [b,c] - [a,c] + [a,b]
        d2[ei[(b, c)], j] += 1
        d2[ei[(a, c)], j] -= 1
        d2[ei[(a, b)], j] += 1
    return graph_b1, graph_b1 - (np.linalg.matrix_rank(d2) if T else 0)


def sanity():
    # 4 sets around a square with no triple overlaps -> one hole (b1 = 1);
    # add a point shared by all four -> the square is filled (b1 = 0)
    square = {'a': {0, 1}, 'b': {1, 2}, 'c': {2, 3}, 'd': {3, 0}}
    assert betti1(square) == (1, 1)
    assert betti1({k: v | {9} for k, v in square.items()}) == (3, 0)


def main():
    sanity()
    mc = load_tda_model(_ROOT / 'models' / 'tda_mapper_model.pkl')
    orig = mc['original_data']
    X_raw = orig[mc['stuff_columns']].values.astype(np.float64)
    ptype = orig.index.get_level_values('pitch_type').values.astype(str)
    rng = np.random.default_rng(SEED)

    rows = [dict(kind='saved_model', **ms.BASE, **dict(zip(('graph_b1', 'nerve_b1'), betti1(mc['graph']['nodes']))))]
    X = StandardScaler().fit_transform(X_raw)
    lens = ms.make_lens(X, 'pca')
    for v in itertools.product(*ms.GRID.values()):
        p = dict(zip(ms.GRID, v))
        _, nodes = ms.fit_graph(X, lens, **p)
        rows.append(dict(kind='grid', **p, **dict(zip(('graph_b1', 'nerve_b1'), betti1(nodes)))))
    for _ in range(N_NULL):
        for kind, Xn in (('null_gauss', ms.gaussian_null(X_raw, rng)),
                         ('null_shuffle', ms.shuffle_within_type_null(X_raw, ptype, rng))):
            Xs = StandardScaler().fit_transform(Xn)
            _, nodes = ms.fit_graph(Xs, ms.make_lens(Xs, 'pca'), **ms.BASE)
            rows.append(dict(kind=kind, **ms.BASE, **dict(zip(('graph_b1', 'nerve_b1'), betti1(nodes)))))

    res = pd.DataFrame(rows)
    out = _ROOT / 'data' / 'nerve_homology.csv'
    res.to_csv(out, index=False)
    print(res.groupby('kind', sort=False)[['graph_b1', 'nerve_b1']].agg(['min', 'median', 'max']).to_string())
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
