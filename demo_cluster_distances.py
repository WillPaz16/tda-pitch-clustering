"""
Demo of distance query capabilities
"""

import pickle
import numpy as np
import networkx as nx
from pathlib import Path

print("="*70)
print("TDA MAPPER CLUSTER DISTANCE ANALYSIS - DEMO")
print("="*70)

# Load TDA model
print("\nLoading TDA model...")
with open('tda_mapper_model.pkl', 'rb') as f:
    model_components = pickle.load(f)

cluster_summary = model_components['cluster_summary'].copy()
stuff_columns = model_components['stuff_columns']
scaler = model_components['scaler']

input_columns = [col for col in stuff_columns if col != 'spin_axis_clock']
cluster_summary['pfx_x'] = cluster_summary['HB'] / -12
cluster_summary['pfx_z'] = cluster_summary['IVB'] / 12

X_clusters = cluster_summary[input_columns].values.astype(np.float64)
X_clusters_scaled = scaler.transform(X_clusters)

# Build graph
print("Building graph...")
n_clusters = len(cluster_summary)
distances = np.zeros((n_clusters, n_clusters))

for i in range(n_clusters):
    for j in range(i, n_clusters):
        dist = np.linalg.norm(X_clusters_scaled[i] - X_clusters_scaled[j])
        distances[i, j] = dist
        distances[j, i] = dist

G = nx.Graph()
for idx, row in cluster_summary.iterrows():
    cluster_id = row['cluster']
    G.add_node(cluster_id, size=int(row['size']), pitch_type=row['most_common_pitch_type'])

for i in range(n_clusters):
    for j in range(i + 1, n_clusters):
        corr = 1 / (1 + distances[i, j])
        if corr > 0.05:
            cluster_i = cluster_summary.iloc[i]['cluster']
            cluster_j = cluster_summary.iloc[j]['cluster']
            G.add_edge(cluster_i, cluster_j, weight=float(distances[i, j]))

print(f"✓ Graph built: {len(G.nodes())} clusters, {len(G.edges())} connections\n")

# Example 1: Query distances between major clusters
print("="*70)
print("EXAMPLE 1: Distances between largest clusters")
print("="*70)

large_clusters = cluster_summary.nlargest(5, 'size')[['cluster', 'size', 'most_common_pitch_type']]
print(f"\nTop 5 clusters by pitches:")
for idx, (_, row) in enumerate(large_clusters.iterrows(), 1):
    print(f"  {idx}. {row['cluster']:20} {int(row['size']):5} pitches ({row['most_common_pitch_type']})")

print("\nPairwise distances:")
for i in range(len(large_clusters)):
    for j in range(i + 1, len(large_clusters)):
        c1 = large_clusters.iloc[i]['cluster']
        c2 = large_clusters.iloc[j]['cluster']
        dist, path = nx.single_source_dijkstra(G, c1, target=c2, weight='weight')
        print(f"  {c1:20} → {c2:20} distance: {dist:.4f}")

# Example 2: Nearest neighbors
print("\n" + "="*70)
print("EXAMPLE 2: Nearest neighbors to a cluster")
print("="*70)

query_cluster = 'cube60_cluster0'  # Largest fastball cluster
print(f"\nFinding 5 nearest clusters to {query_cluster}:")

lengths = nx.single_source_dijkstra_path_length(G, query_cluster, weight='weight')
nearest = sorted([(c, d) for c, d in lengths.items() if c != query_cluster], key=lambda x: x[1])[:5]

for i, (cluster_id, distance) in enumerate(nearest, 1):
    info = cluster_summary[cluster_summary['cluster'] == cluster_id].iloc[0]
    print(f"  {i}. {cluster_id:20} distance={distance:.4f}  {int(info['size']):5} pitches ({info['most_common_pitch_type']})")

# Example 3: Different pitch types
print("\n" + "="*70)
print("EXAMPLE 3: Distance between clusters of different pitch types")
print("="*70)

pitch_types = ['FF', 'SL', 'CH', 'CU']
clusters_by_type = {}

for pt in pitch_types:
    pt_clusters = cluster_summary[cluster_summary['most_common_pitch_type'] == pt].nlargest(1, 'size')
    if len(pt_clusters) > 0:
        clusters_by_type[pt] = pt_clusters.iloc[0]['cluster']

print(f"\nUsing largest cluster of each type:")
for pt, cluster in clusters_by_type.items():
    info = cluster_summary[cluster_summary['cluster'] == cluster].iloc[0]
    print(f"  {pt}: {cluster} ({int(info['size'])} pitches, {info['release_speed']:.1f} mph)")

print(f"\nCross-pitch-type distances:")
clusters_list = list(clusters_by_type.items())
for i in range(len(clusters_list)):
    for j in range(i + 1, len(clusters_list)):
        pt1, c1 = clusters_list[i]
        pt2, c2 = clusters_list[j]
        if c1 in G and c2 in G:
            try:
                dist, _ = nx.single_source_dijkstra(G, c1, target=c2, weight='weight')
                print(f"  {pt1} ({c1}) → {pt2} ({c2}): {dist:.4f}")
            except:
                print(f"  {pt1} ({c1}) → {pt2} ({c2}): No path")

# Example 4: Cluster statistics
print("\n" + "="*70)
print("EXAMPLE 4: Graph connectivity statistics")
print("="*70)

print(f"\nConnectivity analysis:")
print(f"  Total nodes: {len(G.nodes())}")
print(f"  Total edges: {len(G.edges())}")
print(f"  Average degree: {2*len(G.edges())/len(G.nodes()):.2f}")
print(f"  Connected components: {nx.number_connected_components(G)}")

# Diameter and radius
if nx.is_connected(G):
    diameter = nx.diameter(G, weight='weight')
    radius = nx.radius(G, weight='weight')
    print(f"  Graph diameter: {diameter:.4f}")
    print(f"  Graph radius: {radius:.4f}")

print("\n" + "="*70)
print("✓ Demo complete!")
print("\nNext steps:")
print("  1. View tda_mapper_graph.html in a browser for interactive visualization")
print("  2. Run: python3 query_cluster_distances.py")
print("  3. Query cluster_distances.csv for full distance matrix")
print("="*70)
