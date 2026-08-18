"""
TDA Mapper Graph Topology Discovery

Analyzes the actual fitted KeplerMapper graph (models/tda_mapper_model.pkl)
for connectivity structure (connected components, independent loops via the
first Betti number, branch points), checks robustness of that structure
across different nerve min_intersection thresholds, and cross-references the
resulting cluster groupings against real per-pitch Stuff+ data
(data/pitch_stuffplus_clusters.csv) to see whether topological structure
found in the *fitted* graph is reliably reproducible when new pitches are
classified against it.

This is an analysis/reporting script, not part of the production pipeline.
See docs/DISCOVERY_FINDINGS.md for the write-up of what this found.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

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
    """Build a networkx graph from the raw KeplerMapper graph dict."""
    G = nx.Graph()
    for node in mapper_graph['nodes']:
        G.add_node(node)
    for src, targets in mapper_graph['links'].items():
        for t in targets:
            G.add_edge(src, t)
    return G


def describe_components(G, cluster_summary):
    """Print connected-component structure: giant component vs. isolated outliers."""
    cs = cluster_summary.set_index('cluster')
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    giant = comps[0]
    print(f"Connected components: {len(comps)}")
    print(f"Giant component size: {len(giant)} of {G.number_of_nodes()} nodes")

    giant_sub = G.subgraph(giant)
    V, E = giant_sub.number_of_nodes(), giant_sub.number_of_edges()
    print(f"Giant component: V={V}, E={E}, first Betti number (independent loops) b1={E - V + 1}")

    print("\nNodes outside the giant component:")
    for c in comps[1:]:
        descs = []
        for n in c:
            if n in cs.index:
                r = cs.loc[n]
                descs.append(f"{n} ({r['most_common_pitch_type']}, {r['release_speed']:.1f}mph, n={int(r['size'])})")
            else:
                descs.append(n)
        print("  -", ", ".join(descs))

    return giant, set().union(*comps[1:]) if len(comps) > 1 else set()


def min_intersection_robustness_check(model_components, thresholds=(1, 2, 3, 4)):
    """
    Rebuild the Mapper graph at different nerve min_intersection thresholds
    to check whether the connectivity structure (giant component + isolated
    outliers) is a robust finding or an artifact of the default threshold=1.
    """
    import kmapper as km
    from kmapper.nerve import GraphNerve
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    scaler = model_components['scaler']
    orig = model_components['original_data']
    stuff_columns = model_components['stuff_columns']

    X = orig[stuff_columns].values
    X_scaled = scaler.transform(X)
    mapper = km.KeplerMapper()
    lens = mapper.fit_transform(X_scaled, projection=PCA(n_components=2))

    print("\nRobustness check across nerve min_intersection:")
    for min_int in thresholds:
        graph = mapper.map(
            lens, X_scaled,
            clusterer=DBSCAN(eps=1, min_samples=4),
            cover=km.Cover(n_cubes=10, perc_overlap=0.3),
            nerve=GraphNerve(min_intersection=min_int),
        )
        G = build_networkx_graph(graph)
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        giant_sub = G.subgraph(comps[0])
        V, E = giant_sub.number_of_nodes(), giant_sub.number_of_edges()
        print(f"  min_intersection={min_int}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
              f"components={len(comps)} giant_size={V} giant_b1={E - V + 1}")


def check_stuffplus_by_component(giant_nodes, outlier_nodes, pitch_data_path=_DEFAULT_PITCH_DATA_PATH):
    """
    Cross-reference the giant-component vs. outlier-component grouping against
    real per-pitch Stuff+ data to see whether the topological distinction
    corresponds to a measurable Stuff+ difference, and whether pitches are
    being reliably routed to the archetype clusters they should match.
    """
    df = pd.read_csv(pitch_data_path)
    df['component'] = df['cluster_id'].apply(
        lambda c: 'giant' if c in giant_nodes else ('outlier' if c in outlier_nodes else 'unknown')
    )

    print(f"\nReal pitches in giant-component clusters: {(df['component'] == 'giant').sum()}")
    print(f"Real pitches in outlier clusters: {(df['component'] == 'outlier').sum()}")
    print(f"Real pitches in cluster_ids not found in the graph at all: {(df['component'] == 'unknown').sum()}")

    g = df[df['component'] == 'giant']['stuff_plus']
    o = df[df['component'] == 'outlier']['stuff_plus']
    if len(o) > 0:
        u_stat, p_value = stats.mannwhitneyu(g, o, alternative='two-sided')
        print(f"\nStuff+ giant (mean={g.mean():.2f}) vs outlier (mean={o.mean():.2f}): "
              f"Mann-Whitney p={p_value:.3f}")

    print("\nSanity check: are real slow pitches (<70mph) landing in the slow-trained outlier clusters?")
    slow = df[df['release_speed'] < 70]
    if len(slow):
        print(slow.groupby('component').size().to_string())
        print("(if these mostly land in 'giant' rather than 'outlier', the outlier clusters trained on "
              "slow archetypes are not reliably reachable by real new pitch data)")
    else:
        print("No pitches under 70mph found in this dataset.")


def main():
    mc = load_model()
    G = build_networkx_graph(mc['graph'])
    giant_nodes, outlier_nodes = describe_components(G, mc['cluster_summary'])
    min_intersection_robustness_check(mc)
    if _DEFAULT_PITCH_DATA_PATH.exists():
        check_stuffplus_by_component(giant_nodes, outlier_nodes)


if __name__ == "__main__":
    main()
