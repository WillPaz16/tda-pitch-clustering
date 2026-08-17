import json
import networkx as nx
import numpy as np
from collections import defaultdict

# Load the graph data
with open('graph_data.json', 'r') as f:
    graph_data = json.load(f)

# Create NetworkX graph
G = nx.Graph()
for link in graph_data['links']:
    G.add_edge(link['source'], link['target'], weight=1)  # All weights are 1

# Get node names mapping
node_names = {}
for i, node in enumerate(graph_data['nodes']):
    node_names[i] = node['name']

print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")
print(f"Connected components: {nx.number_connected_components(G)}")

# Calculate shortest path distances
distances = dict(nx.all_pairs_shortest_path_length(G))

# Save distances to file
with open('graph_distances.json', 'w') as f:
    # Convert to more readable format with node names
    named_distances = {}
    for source, targets in distances.items():
        source_name = node_names[source]
        named_distances[source_name] = {}
        for target, dist in targets.items():
            target_name = node_names[target]
            named_distances[source_name][target_name] = dist
    json.dump(named_distances, f, indent=2)

print("Saved graph distances to graph_distances.json")

# Analyze distance distribution
all_distances = []
for source, targets in distances.items():
    for target, dist in targets.items():
        if source != target:
            all_distances.append(dist)

print(f"Distance statistics:")
print(f"Max distance: {max(all_distances)}")
print(f"Mean distance: {np.mean(all_distances):.2f}")
print(f"Median distance: {np.median(all_distances)}")
print(f"95th percentile: {np.percentile(all_distances, 95)}")

# Show some examples
print("\nExample distances from cube1_cluster0:")
source_name = "cube1_cluster0"
if source_name in named_distances:
    dists = named_distances[source_name]
    sorted_dists = sorted(dists.items(), key=lambda x: x[1])
    for name, dist in sorted_dists[:10]:
        print(f"  {name}: {dist}")
    print("  ...")
    for name, dist in sorted_dists[-5:]:
        print(f"  {name}: {dist}")