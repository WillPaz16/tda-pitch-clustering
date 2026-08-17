"""
Interactive TDA Mapper Graph Visualization
Builds and visualizes the network of clusters discovered by TDA mapper.
Allows querying distances between clusters.
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import pandas as pd
import networkx as nx
import json
from scipy.spatial.distance import pdist, squareform
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler


def load_tda_model(model_path='tda_mapper_model.pkl'):
    """Load the fitted TDA mapper model."""
    with open(model_path, 'rb') as f:
        model_components = pickle.load(f)
    return model_components


def build_cluster_graph(model_components, min_correlation=0.1):
    """
    Build a network graph of clusters based on their feature similarity.
    
    Nodes = clusters
    Edges = pairs of clusters with features similar above threshold
    Edge weight = distance between cluster centroids
    
    Args:
        model_components: Loaded TDA model
        min_correlation: Minimum correlation to draw an edge (0-1)
    
    Returns:
        networkx Graph, cluster_summary DataFrame, distance matrix
    """
    cluster_summary = model_components['cluster_summary'].copy()
    stuff_columns = model_components['stuff_columns']
    scaler = model_components['scaler']
    
    # Get input columns (exclude spin_axis_clock)
    input_columns = [col for col in stuff_columns if col != 'spin_axis_clock']
    
    # Prepare cluster features
    cluster_summary['pfx_x'] = cluster_summary['HB'] / -12
    cluster_summary['pfx_z'] = cluster_summary['IVB'] / 12
    
    X_clusters = cluster_summary[input_columns].values.astype(np.float64)
    X_clusters_scaled = scaler.transform(X_clusters)
    
    # Calculate pairwise distances
    n_clusters = len(cluster_summary)
    distances = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            dist = np.linalg.norm(X_clusters_scaled[i] - X_clusters_scaled[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Calculate pairwise correlations to find edges
    correlations = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                corr = 1 / (1 + distances[i, j])  # Convert distance to similarity
                correlations[i, j] = corr
    
    # Build graph
    G = nx.Graph()
    
    # Add nodes (clusters)
    for idx, row in cluster_summary.iterrows():
        cluster_id = row['cluster']
        G.add_node(
            cluster_id,
            size=int(row['size']),
            release_speed=float(row['release_speed']),
            HB=float(row['HB']),
            IVB=float(row['IVB']),
            pitch_type=row['most_common_pitch_type'],
            index=idx
        )
    
    # Add edges (connections between nearby clusters)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            if correlations[i, j] > min_correlation:
                cluster_i = cluster_summary.iloc[i]['cluster']
                cluster_j = cluster_summary.iloc[j]['cluster']
                G.add_edge(
                    cluster_i,
                    cluster_j,
                    weight=float(distances[i, j]),
                    similarity=float(correlations[i, j])
                )
    
    print(f"Built graph with {len(G.nodes())} clusters and {len(G.edges())} connections")
    
    return G, cluster_summary, distances


def calculate_shortest_paths(G, start_cluster, end_cluster=None):
    """
    Calculate shortest distance between clusters in the graph.
    
    Args:
        G: networkx Graph
        start_cluster: Starting cluster ID
        end_cluster: Target cluster ID (if None, returns distances to all clusters)
    
    Returns:
        Dictionary of distances if end_cluster specified, else dict of all distances from start
    """
    if start_cluster not in G:
        return {"error": f"Cluster {start_cluster} not found in graph"}
    
    # Use Dijkstra's algorithm with edge weights
    lengths = nx.single_source_dijkstra_path_length(G, start_cluster, weight='weight')
    
    if end_cluster:
        if end_cluster not in G:
            return {"error": f"Cluster {end_cluster} not found in graph"}
        return {
            "start": start_cluster,
            "end": end_cluster,
            "distance": float(lengths[end_cluster]),
            "path_length": len(nx.shortest_path(G, start_cluster, end_cluster, weight='weight'))
        }
    else:
        return {
            "start": start_cluster,
            "distances": {str(k): float(v) for k, v in sorted(lengths.items())}
        }


def create_interactive_visualization(G, cluster_summary, output_file='tda_mapper_graph.html'):
    """
    Create an interactive Plotly visualization of the TDA mapper graph.
    
    Args:
        G: networkx Graph
        cluster_summary: DataFrame with cluster info
        output_file: Output HTML file for visualization
    """
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Prepare edge trace
    edge_trace_x = []
    edge_trace_y = []
    edge_hover = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace_x.extend([x0, x1, None])
        edge_trace_y.extend([y0, y1, None])
        edge_hover.append(f"{edge[0]} ↔ {edge[1]}<br>Distance: {edge[2].get('weight', 0):.3f}")
    
    edge_trace = go.Scatter(
        x=edge_trace_x, y=edge_trace_y,
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        text=edge_hover,
        showlegend=False
    )
    
    # Prepare node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    pitch_type_colors = {
        'FF': '#FF6B6B',  # Red - Fastball
        'SI': '#4ECDC4',  # Teal - Sinker
        'CH': '#FFE66D',  # Yellow - Changeup
        'CU': '#95E1D3',  # Mint - Curveball
        'SL': '#C7CEEA',  # Lavender - Slider
        'FS': '#FF8A80',  # Light Red - Fast Ball
        'KC': '#B4E7FF',  # Light Blue - Knuckle Curve
        'FO': '#FFA07A',  # Light Salmon - Forkball
        'SC': '#DDA0DD',  # Plum - Screwball
    }
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_data = G.nodes[node]
        pitch_type = node_data.get('pitch_type', 'Unknown')
        size = node_data.get('size', 1)
        release_speed = node_data.get('release_speed', 0)
        
        node_size.append(max(15, min(50, size / 10)))  # Scale size
        node_color.append(pitch_type_colors.get(pitch_type, '#CCCCCC'))
        
        hover_text = (
            f"<b>{node}</b><br>"
            f"Size: {size} pitches<br>"
            f"Type: {pitch_type}<br>"
            f"Release Speed: {release_speed:.1f} mph<br>"
            f"HB: {node_data.get('HB', 0):.2f}<br>"
            f"IVB: {node_data.get('IVB', 0):.2f}"
        )
        node_text.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.split('_')[1] for n in G.nodes()],  # Show cluster number
        textposition='top center',
        textfont=dict(size=8),
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='TDA Mapper Cluster Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=800,
            width=1200,
            font=dict(size=10)
        )
    )
    
    fig.write_html(output_file)
    print(f"Saved interactive graph to {output_file}")
    
    return fig


def create_distance_matrix_csv(G, cluster_summary, output_file='cluster_distances.csv'):
    """
    Export a distance matrix between all clusters as CSV.
    
    Args:
        G: networkx Graph
        cluster_summary: DataFrame with cluster info
        output_file: Output CSV file
    """
    clusters = sorted([node for node in G.nodes()])
    n = len(clusters)
    
    # Use shortest path distances
    distances = np.zeros((n, n))
    for i, c1 in enumerate(clusters):
        paths = nx.single_source_dijkstra_path_length(G, c1, weight='weight')
        for j, c2 in enumerate(clusters):
            distances[i, j] = paths.get(c2, np.inf)
    
    # Create DataFrame
    dist_df = pd.DataFrame(distances, index=clusters, columns=clusters)
    dist_df.to_csv(output_file)
    print(f"Saved distance matrix to {output_file}")
    
    return dist_df


def query_distances(G, cluster_ids):
    """
    Query distances between multiple clusters.
    
    Args:
        G: networkx Graph
        cluster_ids: List of cluster IDs
    
    Returns:
        Dictionary with pairwise distances
    """
    results = {}
    for i, c1 in enumerate(cluster_ids):
        for c2 in cluster_ids[i+1:]:
            try:
                # Get shortest path distance and path
                distance, path = nx.single_source_dijkstra(
                    G, c1, target=c2, weight='weight'
                )
                results[f"{c1}-{c2}"] = {
                    "direct_distance": float(distance),
                    "path_hops": len(path) - 1
                }
            except nx.NetworkXNoPath:
                results[f"{c1}-{c2}"] = {"error": "No path found"}
    
    return results


def main():
    print("Loading TDA mapper model...")
    model_components = load_tda_model('tda_mapper_model.pkl')
    
    print("\nBuilding cluster graph...")
    G, cluster_summary, distances = build_cluster_graph(model_components, min_correlation=0.05)
    
    print(f"Graph statistics:")
    print(f"  Nodes (clusters): {len(G.nodes())}")
    print(f"  Edges (connections): {len(G.edges())}")
    print(f"  Connected components: {nx.number_connected_components(G)}")
    print(f"  Average degree: {2 * len(G.edges()) / len(G.nodes()):.2f}")
    
    print("\nCreating interactive visualization...")
    create_interactive_visualization(G, cluster_summary)
    
    print("\nExporting distance matrix...")
    dist_df = create_distance_matrix_csv(G, cluster_summary)
    
    # Example distance queries
    print("\n=== Example Distance Queries ===")
    
    # Get clusters sorted by size
    large_clusters = cluster_summary.nlargest(3, 'size')['cluster'].tolist()
    print(f"\nTop 3 clusters by size: {large_clusters}")
    print("Distances between them:")
    
    query_result = query_distances(G, large_clusters)
    for pair, dist_info in query_result.items():
        if "error" not in dist_info:
            print(f"  {pair}: {dist_info['direct_distance']:.4f} (path hops: {dist_info['path_hops']})")
        else:
            print(f"  {pair}: {dist_info['error']}")
    
    # Save query interface info to JSON
    print("\nCreating distance query tool...")
    clusters_list = sorted([str(c) for c in G.nodes()])
    
    query_info = {
        "clusters": clusters_list,
        "num_clusters": len(clusters_list),
        "num_edges": len(G.edges()),
        "usage": "Use cluster IDs like 'cube52_cluster0', 'cube61_cluster1', etc."
    }
    
    with open('cluster_query_info.json', 'w') as f:
        json.dump(query_info, f, indent=2)
    print("Saved query info to cluster_query_info.json")


if __name__ == "__main__":
    main()
