"""
Crowded-Continuum Ambiguity Analysis

Follow-up to the finding in docs/DISCOVERY_FINDINGS.md that high-degree,
well-populated hub clusters have LOW ground-truth self-consistency (many
of their own training points, run back through the production
nearest-centroid classifier, get reassigned elsewhere).

That finding on its own doesn't say whether the reassignment is benign
(a genuinely close call between two very similar, graph-adjacent
archetypes -- expected in a continuum) or something more concerning (a
training point landing far away, in an unrelated part of the graph, which
would suggest real instability rather than boundary ambiguity).

This script checks, for every "misrouted" ground-truth point in the
graph:
  - is the cluster it gets reassigned to a graph neighbor of its own
    cluster (an edge exists in the fitted Mapper graph)?
  - how large is the margin between the distance to its own cluster's
    centroid and the distance to the reassigned cluster's centroid,
    relative to the typical centroid-to-centroid distance across the
    whole graph (a small margin = genuine close call; a large margin
    would mean the point doesn't really belong near its own centroid at
    all)?

Analysis/reporting script. See docs/DISCOVERY_FINDINGS.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'


def load_model(model_path=_DEFAULT_MODEL_PATH):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def build_networkx_graph(mapper_graph):
    G = nx.Graph()
    for node in mapper_graph['nodes']:
        G.add_node(node)
    for src, targets in mapper_graph['links'].items():
        for t in targets:
            G.add_edge(src, t)
    return G


def main():
    mc = load_model()
    G = build_networkx_graph(mc['graph'])
    scaler = mc['scaler']
    stuff_columns = mc['stuff_columns']
    input_columns = [c for c in stuff_columns if c != 'spin_axis_clock']
    orig = mc['original_data']
    graph = mc['graph']
    cs = mc['cluster_summary'].copy()
    cs['pfx_x'] = cs['HB'] / -12
    cs['pfx_z'] = cs['IVB'] / 12
    X_clusters = cs[input_columns].values.astype(np.float64)
    X_clusters_scaled = scaler.transform(X_clusters)
    cluster_ids = cs['cluster'].values
    cluster_pos = {cid: i for i, cid in enumerate(cluster_ids)}

    # typical (median) centroid-to-centroid distance across the whole graph,
    # as a reference scale for what counts as a "small" vs "large" margin
    n_clusters = len(cluster_ids)
    all_pairwise = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            all_pairwise.append(np.linalg.norm(X_clusters_scaled[i] - X_clusters_scaled[j]))
    median_pairwise_dist = np.median(all_pairwise)
    print(f"Median centroid-to-centroid distance across all {n_clusters} clusters: {median_pairwise_dist:.3f}")

    degree = dict(G.degree())

    rows = []
    for cid in G.nodes():
        if cid not in cluster_pos:
            continue
        member_indices = graph['nodes'].get(cid, [])
        own_idx = cluster_pos[cid]
        for idx in member_indices:
            row = orig.iloc[idx][input_columns].values.astype(np.float64).reshape(1, -1)
            row_scaled = scaler.transform(row)
            dists = np.linalg.norm(X_clusters_scaled - row_scaled, axis=1)
            nearest_pos = np.argmin(dists)
            nearest_cluster = cluster_ids[nearest_pos]

            if nearest_cluster == cid:
                continue  # correctly round-tripped, not a misroute

            dist_to_own = dists[own_idx]
            dist_to_reassigned = dists[nearest_pos]
            is_neighbor = G.has_edge(cid, nearest_cluster)
            margin = (dist_to_own - dist_to_reassigned) / median_pairwise_dist

            rows.append({
                'origin_cluster': cid,
                'origin_degree': degree.get(cid, 0),
                'origin_train_n': cs.iloc[own_idx]['size'],
                'reassigned_to': nearest_cluster,
                'is_graph_neighbor': is_neighbor,
                'dist_to_own_centroid': dist_to_own,
                'dist_to_reassigned_centroid': dist_to_reassigned,
                'margin_normalized': margin,
            })

    df = pd.DataFrame(rows)
    print(f"\nTotal ground-truth misroutes across the whole graph: {len(df)}")
    print(f"Misroutes that land on a graph-adjacent (edge-connected) cluster: "
          f"{df['is_graph_neighbor'].sum()} ({100 * df['is_graph_neighbor'].mean():.1f}%)")
    print(f"Misroutes that land on a NON-adjacent cluster: "
          f"{(~df['is_graph_neighbor']).sum()} ({100 * (~df['is_graph_neighbor']).mean():.1f}%)")

    print(f"\nMargin (normalized by median centroid-to-centroid distance) stats:")
    print(df['margin_normalized'].describe())

    print(f"\nMargin stats split by neighbor vs. non-neighbor:")
    print(df.groupby('is_graph_neighbor')['margin_normalized'].describe())

    print("\nWorst 15 misroutes by margin (largest gap = least defensible as a 'close call'):")
    cols = ['origin_cluster', 'origin_train_n', 'reassigned_to', 'is_graph_neighbor', 'margin_normalized']
    print(df.sort_values('margin_normalized', ascending=False)[cols].head(15).to_string(index=False))

    out_path = _ROOT / 'data' / 'crowded_continuum_misroutes.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved full misroute table to {out_path}")


if __name__ == "__main__":
    main()
