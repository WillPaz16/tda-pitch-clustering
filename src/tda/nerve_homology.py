"""
First Betti number of the Mapper graph vs. the full Mapper nerve.

KeplerMapper returns only the 1-skeleton (a graph) of the nerve of the
pulled-back cover. With a 2-D lens, three or more cover elements routinely
share points, so the nerve has 2-simplices that fill many of the graph's
cycles. This computes, for each fit:
  graph_b1 = E - V + C                    (1-skeleton)
  nerve_b1 = E - V + C - rank(d2)         (2-skeleton; d2 = triangle boundary map)
Higher simplices do not change b1. Homology is over Z2, matching
Dey, Memoli & Wang (SoCG 2017), Thm 8 / Thm 18: for the Mapper's
path-connected pullback cover, H1(X) -> H1(N(f*U)) is a surjection, so
b1(Mapper nerve) <= b1(X).

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


def rank_gf2(M):
    """Rank of a 0/1 matrix over Z2 by Gaussian elimination."""
    M = M.astype(bool).copy()
    rank = 0
    for col in range(M.shape[1]):
        pivot = np.flatnonzero(M[rank:, col])
        if not len(pivot):
            continue
        p = rank + pivot[0]
        M[[rank, p]] = M[[p, rank]]
        rows = np.flatnonzero(M[:, col])
        M[rows[rows != rank]] ^= M[rank]
        rank += 1
        if rank == M.shape[0]:
            break
    return rank


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
    d2 = np.zeros((len(E), len(T)), dtype=bool)
    for j, (a, b, c) in enumerate(T):     # over Z2: d[a,b,c] = [b,c] + [a,c] + [a,b]
        d2[[ei[(b, c)], ei[(a, c)], ei[(a, b)]], j] = True
    return graph_b1, graph_b1 - rank_gf2(d2)


def sanity():
    # 4 sets around a square with no triple overlaps -> one hole (b1 = 1);
    # add a point shared by all four -> the square is filled (b1 = 0)
    square = {'a': {0, 1}, 'b': {1, 2}, 'c': {2, 3}, 'd': {3, 0}}
    assert betti1(square) == (1, 1)
    assert betti1({k: v | {9} for k, v in square.items()}) == (3, 0)
    assert rank_gf2(np.array([[1, 1], [1, 1]])) == 1     # rank 2 over Q, 1 over Z2


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
