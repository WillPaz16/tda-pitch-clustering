"""
The paper's running example, end to end: the 240-point noisy circle of
thesis_figures.toy_mapper against a same-covariance Gaussian null, run
through the toy Mapper (height lens, 5 intervals, 30% overlap, DBSCAN
eps 0.25 / M 3) and persistent homology in the real pipeline's units
(median H0 death of the circle).

Output: data/circle_example.csv (one row per point cloud).
Run: python3 src/tda/circle_example.py

Paper (docs/paper/paper_final.tex): the running example in The Mapper Algorithm
(sec:mapper) and Review of the Running Example (sec:certify), Table tab:review.
Output: data/circle_example.csv.
"""
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from persistence_check import h0_deaths, h1_persistence, LOOP_MULT
from mapper_stability import gaussian_null
from nerve_homology import betti1

N_NULL, SEED = 5, 42


def circle():
    """Same sample as thesis_figures.toy_mapper."""
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 240); r = 1 + rng.normal(0, 0.04, 240)
    return np.c_[r * np.cos(t), r * np.sin(t)]


def toy_mapper(P, n=5, ov=0.3, eps=0.25, m=3):
    """Mapper nodes as in thesis_figures.toy_mapper: {node_id: member indices}."""
    f = P[:, 1]
    w = (f.max() - f.min()) / (1 + (n - 1) * (1 - ov)); step = w * (1 - ov)
    nodes = []
    for i in range(n):
        idx = np.flatnonzero((f >= f.min() + i * step) & (f <= f.min() + i * step + w))
        lab = DBSCAN(eps=eps, min_samples=m).fit_predict(P[idx])
        nodes += [idx[lab == c] for c in sorted(set(lab) - {-1})]
    return dict(enumerate(nodes))


def row(kind, P, unit):
    nodes = toy_mapper(P)
    gb1, nb1 = betti1(nodes)
    sets = [set(v) for v in nodes.values()]
    G = nx.Graph(); G.add_nodes_from(range(len(sets)))
    G.add_edges_from((a, b) for a, b in combinations(range(len(sets)), 2) if sets[a] & sets[b])
    tri = sum(1 for a, b, c in combinations(range(len(sets)), 3) if sets[a] & sets[b] & sets[c])
    p1 = np.sort(h1_persistence(P) / unit)[::-1]
    return dict(kind=kind, n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(),
                n_components=nx.number_connected_components(G), triangles=tri, graph_b1=gb1, nerve_b1=nb1,
                h1_max=p1[0], h1_next=p1[1], h1_long_loops=int(np.sum(p1 > LOOP_MULT)))


def main():
    P = circle()
    unit = np.median(h0_deaths(P))
    rng = np.random.default_rng(SEED)
    rows = [row('circle', P, unit)] + [row('null_gauss', gaussian_null(P, rng), unit) for _ in range(N_NULL)]
    out = pd.DataFrame(rows)
    out.to_csv(_ROOT / 'data' / 'circle_example.csv', index=False)
    print(out.round(3).to_string())
    c = out.iloc[0]
    assert (c.n_nodes, c.n_edges, c.graph_b1, c.triangles) == (8, 8, 1, 0), "circle example no longer matches the paper"


if __name__ == "__main__":
    main()
