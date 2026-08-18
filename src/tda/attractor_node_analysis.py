"""
Attractor Node Verification

Follow-up to an unverified observation in docs/DISCOVERY_FINDINGS.md: a
few specific nodes (cube59_cluster1, cube62_cluster2, cube62_cluster3)
showed up repeatedly as the destination of the largest-margin misroutes
from several different large neighboring hub clusters, which was flagged
as noticed-but-not-tested.

This checks it systematically: computes betweenness centrality (a
standard graph measure of how often a node lies on shortest paths between
other node pairs -- the network-theoretic definition of a "bridge" or
"crossroads" node) and large-neighbor count for every node, and compares
the three candidate nodes against other nodes of similar degree (since
raw attractor-ness could just be a trivial consequence of having high
degree in the first place).

Also reports each candidate node's baseball identity (dominant pitch
type, velocity, movement) and its largest neighbors', since a network
statistic on its own doesn't mean anything without grounding it in what
pitch shapes are actually involved.

Analysis/reporting script. See docs/DISCOVERY_FINDINGS.md for write-up.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import pandas as pd
import networkx as nx

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / 'models' / 'tda_mapper_model.pkl'
LARGE_NEIGHBOR_THRESHOLD = 100  # training points; matches the scale of the hub clusters in question

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


def describe_cluster(cs, cid):
    r = cs.loc[cid]
    return (f"{cid}: {r['most_common_pitch_type']} @ {r['release_speed']:.1f}mph, "
            f"HB={r['HB']:.1f}, IVB={r['IVB']:.1f}, n={int(r['size'])}, "
            f"unique_types_within_node={int(r['unique_types'])}")


def main():
    mc = load_model()
    G = build_networkx_graph(mc['graph'])
    cs = mc['cluster_summary'].set_index('cluster')

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    large_neighbor_count = {
        n: sum(1 for nb in G.neighbors(n) if nb in cs.index and cs.loc[nb, 'size'] >= LARGE_NEIGHBOR_THRESHOLD)
        for n in G.nodes()
    }

    rows = []
    for n in G.nodes():
        if n not in cs.index:
            continue
        rows.append({
            'cluster': n,
            'degree': degree[n],
            'betweenness': betweenness[n],
            'large_neighbor_count': large_neighbor_count[n],
            'train_n': cs.loc[n, 'size'],
        })
    df = pd.DataFrame(rows)

    candidates = ['cube59_cluster1', 'cube62_cluster2', 'cube62_cluster3']

    print("Candidate attractor nodes vs. all other nodes of the same degree:\n")
    for cand in candidates:
        if cand not in df['cluster'].values:
            print(f"{cand}: not found in graph")
            continue
        cand_row = df[df['cluster'] == cand].iloc[0]
        same_degree = df[df['degree'] == cand_row['degree']]
        print(f"--- {cand} (degree={int(cand_row['degree'])}) ---")
        print(f"  betweenness: {cand_row['betweenness']:.4f}  "
              f"(same-degree nodes: mean={same_degree['betweenness'].mean():.4f}, "
              f"max={same_degree['betweenness'].max():.4f}, n={len(same_degree)})")
        print(f"  large_neighbor_count (>= {LARGE_NEIGHBOR_THRESHOLD} training pts): "
              f"{int(cand_row['large_neighbor_count'])}  "
              f"(same-degree nodes: mean={same_degree['large_neighbor_count'].mean():.2f}, "
              f"max={same_degree['large_neighbor_count'].max()})")
        print(f"  betweenness percentile among same-degree nodes: "
              f"{100 * (same_degree['betweenness'] <= cand_row['betweenness']).mean():.0f}th")
        print()

    print("Overall correlation check: does betweenness explain more than degree alone?")
    from scipy import stats
    r_deg_bet, p_deg_bet = stats.spearmanr(df['degree'], df['betweenness'])
    print(f"  degree vs betweenness: Spearman r={r_deg_bet:.3f}, p={p_deg_bet:.4f}")

    print("\nTop 10 nodes by betweenness centrality (whole-graph ranking):")
    print(df.sort_values('betweenness', ascending=False)[['cluster', 'degree', 'betweenness', 'large_neighbor_count', 'train_n']].head(10).to_string(index=False))

    print("\n--- Baseball identity of the candidate nodes ---")
    for cand in candidates:
        if cand in cs.index:
            print(" ", describe_cluster(cs, cand))

    print("\n--- Baseball identity of their large (>=100 pt) neighbors ---")
    for cand in candidates:
        if cand not in G:
            continue
        for nb in G.neighbors(cand):
            if nb in cs.index and cs.loc[nb, 'size'] >= LARGE_NEIGHBOR_THRESHOLD:
                print(f"  {cand} -> {describe_cluster(cs, nb)}")

    out_path = _ROOT / 'data' / 'attractor_node_stats.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved full node-stats table to {out_path}")


if __name__ == "__main__":
    main()
