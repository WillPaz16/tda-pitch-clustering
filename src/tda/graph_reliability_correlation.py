"""
Graph Isolation vs. Classification Reliability

Tests the claim from docs/DISCOVERY_FINDINGS.md across every cluster (not
just the 11 outliers found by inspection): does topological isolation in
the fitted Mapper graph predict where downstream classification is
unreliable?

For every node in the graph, computes:
  - degree: number of edges in the fitted Mapper graph (topology)
  - component_size: size of the connected component it belongs to (topology)
  - train_n: number of pitcher-pitch-type training points in the cluster
  - roundtrip_accuracy: fraction of the cluster's own training points that
    correctly route back to their own cluster when pushed through the
    actual production nearest-centroid classifier (ground truth, no live
    data involved)
  - live_speed_error: for real per-pitch data in
    data/pitch_stuffplus_clusters.csv, the mean absolute difference between
    a cluster's trained release_speed and the real release_speed of pitches
    actually assigned there in production (a live-data reliability signal)

Then reports Spearman correlations between the topology measures (degree,
component_size) and the reliability measures (roundtrip_accuracy,
live_speed_error), plus train_n as a candidate mediating variable.

Analysis/reporting script, not part of the production pipeline. See
docs/DISCOVERY_FINDINGS.md for the write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'
_DEFAULT_PITCH_DATA_PATH = _ROOT / 'data' / 'pitch_stuffplus_clusters.csv'

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tda_classifier import load_tda_model


def load_model(model_path=_DEFAULT_MODEL_PATH):
    return load_tda_model(model_path)


def build_networkx_graph(mapper_graph):
    G = nx.Graph()
    for node in mapper_graph['nodes']:
        G.add_node(node)
    for src, targets in mapper_graph['links'].items():
        for t in targets:
            G.add_edge(src, t)
    return G


def compute_roundtrip_accuracy(mc, G):
    """For every cluster, what fraction of its own training points route
    back to their own cluster via the production nearest-centroid logic?"""
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

    results = {}
    for cid in G.nodes():
        member_indices = graph['nodes'].get(cid, [])
        if not member_indices:
            results[cid] = np.nan
            continue
        n_correct = 0
        for idx in member_indices:
            row = orig.iloc[idx][input_columns].values.astype(np.float64).reshape(1, -1)
            row_scaled = scaler.transform(row)
            dists = np.linalg.norm(X_clusters_scaled - row_scaled, axis=1)
            nearest_cluster = cluster_ids[np.argmin(dists)]
            n_correct += (nearest_cluster == cid)
        results[cid] = n_correct / len(member_indices)
    return results


def compute_live_speed_error(cluster_summary, pitch_data_path=_DEFAULT_PITCH_DATA_PATH):
    """For real per-pitch data, how far off is the actual assigned pitch's
    release_speed from the trained cluster's release_speed?"""
    if not pitch_data_path.exists():
        return {}
    df = pd.read_csv(pitch_data_path)
    cs = cluster_summary.set_index('cluster')
    results = {}
    for cid, grp in df.groupby('cluster_id'):
        if cid not in cs.index:
            continue
        trained_speed = cs.loc[cid, 'release_speed']
        results[cid] = (grp['release_speed'] - trained_speed).abs().mean()
    return results


def main():
    mc = load_model()
    G = build_networkx_graph(mc['graph'])
    cs = mc['cluster_summary'].set_index('cluster')

    degree = dict(G.degree())
    comps = {n: len(c) for c in nx.connected_components(G) for n in c}
    roundtrip = compute_roundtrip_accuracy(mc, G)
    live_error = compute_live_speed_error(mc['cluster_summary'])

    rows = []
    for cid in G.nodes():
        if cid not in cs.index:
            continue
        rows.append({
            'cluster': cid,
            'degree': degree.get(cid, 0),
            'component_size': comps.get(cid, 1),
            'train_n': cs.loc[cid, 'size'],
            'roundtrip_accuracy': roundtrip.get(cid, np.nan),
            'live_speed_error': live_error.get(cid, np.nan),
        })
    df = pd.DataFrame(rows)

    print(f"Clusters analyzed: {len(df)}")
    print(f"Clusters with roundtrip accuracy computed: {df['roundtrip_accuracy'].notna().sum()}")
    print(f"Clusters with live data assigned: {df['live_speed_error'].notna().sum()}")
    print()

    pairs = [
        ('degree', 'roundtrip_accuracy'),
        ('component_size', 'roundtrip_accuracy'),
        ('train_n', 'roundtrip_accuracy'),
        ('degree', 'live_speed_error'),
        ('component_size', 'live_speed_error'),
        ('train_n', 'live_speed_error'),
        ('degree', 'train_n'),
    ]
    print(f"{'x':<18}{'y':<20}{'spearman_r':<14}{'p_value':<12}{'n'}")
    for x, y in pairs:
        sub = df[[x, y]].dropna()
        if len(sub) < 3:
            continue
        r, p = stats.spearmanr(sub[x], sub[y])
        print(f"{x:<18}{y:<20}{r:<14.3f}{p:<12.4f}{len(sub)}")

    print()
    print("Full table (sorted by degree, ascending -- most isolated first):")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 160)
    print(df.sort_values('degree').to_string(index=False))

    out_path = _ROOT / 'data' / 'graph_reliability_correlation.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved full table to {out_path}")


if __name__ == "__main__":
    main()
