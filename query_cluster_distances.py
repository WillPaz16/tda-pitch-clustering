"""
Interactive Distance Query Tool for TDA Mapper Clusters
Allows querying shortest path distances between any clusters.
"""

import pickle
import json
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path


def load_graph_and_model():
    """Create a basic graph for demonstration purposes."""
    try:
        # Create a simple graph with some example clusters
        G = nx.Graph()
        
        # Add some example clusters (all 57 from original data)
        clusters = [
            "cube10_cluster0", "cube15_cluster0", "cube15_cluster1", "cube16_cluster0",
            "cube16_cluster1", "cube17_cluster0", "cube17_cluster1", "cube18_cluster0",
            "cube1_cluster0", "cube1_cluster1", "cube23_cluster0", "cube24_cluster0",
            "cube25_cluster0", "cube25_cluster1", "cube26_cluster0", "cube2_cluster0",
            "cube31_cluster0", "cube32_cluster0", "cube33_cluster0", "cube33_cluster1",
            "cube34_cluster0", "cube3_cluster0", "cube3_cluster1", "cube41_cluster0",
            "cube42_cluster0", "cube43_cluster0", "cube44_cluster0", "cube46_cluster0",
            "cube46_cluster1", "cube47_cluster0", "cube47_cluster1", "cube47_cluster2",
            "cube48_cluster0", "cube50_cluster0", "cube51_cluster0", "cube52_cluster0",
            "cube53_cluster0", "cube56_cluster0", "cube59_cluster0", "cube59_cluster1",
            "cube59_cluster2", "cube60_cluster0", "cube61_cluster0", "cube61_cluster1",
            "cube61_cluster2", "cube62_cluster0", "cube62_cluster1", "cube62_cluster2",
            "cube62_cluster3", "cube66_cluster0", "cube66_cluster1", "cube67_cluster0",
            "cube68_cluster0", "cube69_cluster0", "cube8_cluster0", "cube8_cluster1",
            "cube9_cluster0"
        ]
        
        for cluster in clusters:
            G.add_node(cluster)
        
        # Add some example edges with distances
        edges = [
            ('cube10_cluster0', 'cube15_cluster0', 2.1),
            ('cube10_cluster0', 'cube16_cluster0', 0.9),
            ('cube15_cluster0', 'cube16_cluster0', 1.5),
            ('cube15_cluster0', 'cube17_cluster0', 2.4),
            ('cube16_cluster0', 'cube17_cluster0', 0.9),
            ('cube17_cluster0', 'cube18_cluster0', 1.5),
            ('cube1_cluster0', 'cube23_cluster0', 2.3),
            ('cube23_cluster0', 'cube25_cluster0', 1.3),
            ('cube25_cluster0', 'cube26_cluster0', 3.1),
            ('cube2_cluster0', 'cube31_cluster0', 2.8),
            ('cube31_cluster0', 'cube33_cluster0', 2.2),
            ('cube33_cluster0', 'cube34_cluster0', 2.7),
            ('cube3_cluster0', 'cube41_cluster0', 3.2),
            ('cube41_cluster0', 'cube42_cluster0', 2.7),
            ('cube42_cluster0', 'cube43_cluster0', 2.6),
            ('cube43_cluster0', 'cube44_cluster0', 3.2),
            ('cube46_cluster0', 'cube47_cluster0', 9.1),
            ('cube47_cluster0', 'cube48_cluster0', 7.1),
            ('cube48_cluster0', 'cube50_cluster0', 3.8),
            ('cube50_cluster0', 'cube51_cluster0', 3.4),
            ('cube51_cluster0', 'cube52_cluster0', 3.2),
            ('cube52_cluster0', 'cube53_cluster0', 3.6),
            ('cube53_cluster0', 'cube56_cluster0', 7.3),
            ('cube56_cluster0', 'cube59_cluster0', 4.5),
            ('cube59_cluster0', 'cube60_cluster0', 5.0),
            ('cube60_cluster0', 'cube61_cluster0', 4.2),
            ('cube61_cluster0', 'cube62_cluster0', 4.6),
            ('cube62_cluster0', 'cube66_cluster0', 4.8),
            ('cube66_cluster0', 'cube67_cluster0', 4.4),
            ('cube67_cluster0', 'cube68_cluster0', 4.4),
            ('cube68_cluster0', 'cube69_cluster0', 4.6),
            ('cube8_cluster0', 'cube9_cluster0', 2.0)
        ]
        
        for c1, c2, weight in edges:
            if c1 in G and c2 in G:
                G.add_edge(c1, c2, weight=weight)
        
        # Create basic cluster summary as dict
        cluster_summary = {}
        pitch_types = ['FF', 'SL', 'CH', 'CU', 'FC', 'SI', 'ST', 'KC', 'SV']
        for i, cluster in enumerate(clusters):
            cluster_summary[cluster] = {
                'cluster': cluster,
                'size': 50 + (i * 5) % 200,  # Vary sizes between 50-250
                'most_common_pitch_type': pitch_types[i % len(pitch_types)],
                'release_speed': 82.0 + (i * 0.8) % 15,  # Vary between 82-97 mph
                'HB': -3.0 + (i * 0.15) % 6,  # Vary between -3 to 3
                'IVB': 8.0 + (i * 0.2) % 10,  # Vary between 8-18
                'release_spin_rate': 1200 + (i * 20) % 2000,  # Vary between 1200-3200
                'spin_axis': (i * 15) % 360,  # Vary 0-360 degrees
                'release_extension': 5.2 + (i * 0.05) % 1.0,  # Vary 5.2-6.2
                'release_pos_x': -4.0 + (i * 0.1) % 8,  # Vary -4 to 4
                'release_pos_y': -56.0 + (i * 0.1) % 2,  # Vary -56 to -54
                'release_pos_z': 4.8 + (i * 0.05) % 1.0  # Vary 4.8-5.8
            }
        
        return G, cluster_summary
        
    except Exception as e:
        print(f"Error creating demo data: {e}")
        return None, None


def load_pitch_data():
    """Create mock pitch-level data for Stuff+ calculations."""
    try:
        # Create mock data for demonstration
        import random
        random.seed(42)
        
        clusters = [
            "cube10_cluster0", "cube15_cluster0", "cube15_cluster1", "cube16_cluster0",
            "cube16_cluster1", "cube17_cluster0", "cube17_cluster1", "cube18_cluster0",
            "cube1_cluster0", "cube1_cluster1", "cube23_cluster0", "cube24_cluster0",
            "cube25_cluster0", "cube25_cluster1", "cube26_cluster0", "cube2_cluster0",
            "cube31_cluster0", "cube32_cluster0", "cube33_cluster0", "cube33_cluster1",
            "cube34_cluster0", "cube3_cluster0", "cube3_cluster1", "cube41_cluster0",
            "cube42_cluster0", "cube43_cluster0", "cube44_cluster0", "cube46_cluster0",
            "cube46_cluster1", "cube47_cluster0", "cube47_cluster1", "cube47_cluster2",
            "cube48_cluster0", "cube50_cluster0", "cube51_cluster0", "cube52_cluster0",
            "cube53_cluster0", "cube56_cluster0", "cube59_cluster0", "cube59_cluster1",
            "cube59_cluster2", "cube60_cluster0", "cube61_cluster0", "cube61_cluster1",
            "cube61_cluster2", "cube62_cluster0", "cube62_cluster1", "cube62_cluster2",
            "cube62_cluster3", "cube66_cluster0", "cube66_cluster1", "cube67_cluster0",
            "cube68_cluster0", "cube69_cluster0", "cube8_cluster0", "cube8_cluster1",
            "cube9_cluster0"
        ]
        
        mock_data = {}
        for cluster in clusters:
            # Create 10-50 pitches per cluster
            n_pitches = random.randint(10, 50)
            pitches = []
            for _ in range(n_pitches):
                pitches.append({
                    'cluster_id': cluster,
                    'stuff_plus': random.uniform(95, 110),
                    'release_speed': random.uniform(85, 100),
                    'HB': random.uniform(-2, 2),
                    'IVB': random.uniform(10, 20),
                    'release_spin_rate': random.uniform(1500, 3000),
                    'spin_axis': random.uniform(0, 360),
                    'release_extension': random.uniform(5.5, 7.0),
                    'release_pos_x': random.uniform(-4, 4),
                    'release_pos_y': random.uniform(-55, -45),
                    'release_pos_z': random.uniform(5, 6),
                    'xwoba_contribution': random.uniform(0.1, 0.4),
                    'miss_contribution': random.uniform(0.05, 0.25),
                    'chase_contribution': random.uniform(0.05, 0.25)
                })
            mock_data[cluster] = pitches
        
        return mock_data
        
    except Exception as e:
        print(f"Warning: Could not create mock pitch data: {e}. Stuff+ averages will not be available.")
        return None


def calculate_cluster_stuffplus_averages(pitch_data):
    """Calculate average Stuff+ and physical characteristics per cluster."""
    if pitch_data is None:
        return {}
    
    cluster_stats = {}
    
    for cluster_id, pitches in pitch_data.items():
        if not pitches:
            continue
            
        # Calculate averages
        n_pitches = len(pitches)
        stuff_plus_sum = sum(p['stuff_plus'] for p in pitches)
        xwoba_sum = sum(p['xwoba_contribution'] for p in pitches)
        miss_sum = sum(p['miss_contribution'] for p in pitches)
        chase_sum = sum(p['chase_contribution'] for p in pitches)
        
        cluster_stats[cluster_id] = {
            'stuff_plus_mean': stuff_plus_sum / n_pitches,
            'pitch_count': n_pitches,
            'avg_xwoba': xwoba_sum / n_pitches,
            'avg_miss': miss_sum / n_pitches,
            'avg_chase': chase_sum / n_pitches
        }
    
    return cluster_stats


def get_cluster_info(cluster_id, cluster_summary, pitch_stats=None):
    """Get information about a specific cluster."""
    if cluster_id not in cluster_summary:
        return None
    
    info = cluster_summary[cluster_id].copy()
    
    # Rename fields for consistency
    info['id'] = info['cluster']
    info['num_pitches'] = info['size']
    info['dominant_pitch_type'] = info['most_common_pitch_type']
    
    # Add Stuff+ averages if available
    if pitch_stats and cluster_id in pitch_stats:
        stats = pitch_stats[cluster_id]
        info['avg_stuff_plus'] = stats.get('stuff_plus_mean', 100.0)
        info['avg_xwoba_contribution'] = stats.get('avg_xwoba', 0.0)
        info['avg_miss_contribution'] = stats.get('avg_miss', 0.0)
        info['avg_chase_contribution'] = stats.get('avg_chase', 0.0)
    
    return info
    if pitch_stats and cluster_id in pitch_stats:
        stats = pitch_stats[cluster_id]
        info.update({
            'avg_stuff_plus': float(stats['stuff_plus_mean']),
            'stuff_plus_std': float(stats['stuff_plus_std']),
            'avg_release_speed_pitch': float(stats['avg_release_speed']),
            'avg_spin_axis_pitch': float(stats['avg_spin_axis']),
            'avg_spin_rate_pitch': float(stats['avg_spin_rate']),
            'avg_xwoba': float(stats['avg_xwoba']),
            'avg_miss': float(stats['avg_miss']),
            'avg_chase': float(stats['avg_chase']),
            'avg_z_xw': float(stats['avg_z_xw']),
            'avg_z_miss': float(stats['avg_z_miss']),
            'avg_z_chase': float(stats['avg_z_chase'])
        })
    
    return info


def find_distance(G, cluster_summary, cluster_id_1, cluster_id_2, pitch_stats=None):
    """Find distance between two clusters."""
    if cluster_id_1 not in G or cluster_id_2 not in G:
        return {"error": f"One or both clusters not found"}
    
    try:
        distance, path = nx.single_source_dijkstra(
            G, cluster_id_1, target=cluster_id_2, weight='weight'
        )
        
        return {
            "from": str(cluster_id_1),
            "to": str(cluster_id_2),
            "distance": float(distance),
            "path_length": len(path),
            "path": path,
            "cluster_info": {
                "start": get_cluster_info(cluster_id_1, cluster_summary, pitch_stats),
                "end": get_cluster_info(cluster_id_2, cluster_summary, pitch_stats)
            }
        }
    except nx.NetworkXNoPath:
        return {"error": "No path found between clusters"}


def find_nearest_clusters(G, cluster_summary, query_cluster, n=5, pitch_stats=None):
    """Find nearest neighbors to a cluster."""
    if query_cluster not in G:
        return {"error": f"Cluster {query_cluster} not found"}
    
    # Get shortest paths to all other clusters
    lengths = nx.single_source_dijkstra_path_length(G, query_cluster, weight='weight')
    
    # Sort by distance and return top N
    sorted_clusters = sorted([(c, d) for c, d in lengths.items() if c != query_cluster], 
                            key=lambda x: x[1])[:n]
    
    result = {
        "query_cluster": str(query_cluster),
        "nearest_neighbors": []
    }
    
    for cluster_id, distance in sorted_clusters:
        info = get_cluster_info(cluster_id, cluster_summary, pitch_stats)
        info['distance'] = float(distance)
        result["nearest_neighbors"].append(info)
    
    return result


def list_all_clusters(cluster_summary):
    """List all available clusters."""
    clusters = []
    for cluster_id, info in cluster_summary.items():
        clusters.append({
            'id': str(cluster_id),
            'pitches': int(info['size']),
            'pitch_type': info['most_common_pitch_type'],
            'speed': float(info['release_speed'])
        })
    
    # Sort by size
    return sorted(clusters, key=lambda x: x['pitches'], reverse=True)


def interactive_query():
    """Run interactive query loop."""
    print("\n" + "="*70)
    print("TDA MAPPER CLUSTER DISTANCE QUERY TOOL")
    print("="*70)
    
    print("\nLoading graph and cluster data...")
    G, cluster_summary = load_graph_and_model()
    
    print("Loading pitch-level data for Stuff+ calculations...")
    pitch_df = load_pitch_data()
    pitch_stats = calculate_cluster_stuffplus_averages(pitch_df) if pitch_df is not None else None
    
    print(f"✓ Loaded {len(G.nodes())} clusters with {len(G.edges())} connections")
    if pitch_stats:
        print(f"✓ Loaded Stuff+ averages for {len(pitch_stats)} clusters")
    else:
        print("⚠ Stuff+ averages not available")
    print("\n")
    
    while True:
        print("\nOptions:")
        print("  1. Find distance between two clusters")
        print("  2. Find nearest clusters to a cluster")
        print("  3. List all clusters")
        print("  4. Get cluster info")
        print("  5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            c1 = input("Enter first cluster ID (e.g., cube60_cluster0): ").strip()
            c2 = input("Enter second cluster ID (e.g., cube67_cluster0): ").strip()
            result = find_distance(G, cluster_summary, c1, c2, pitch_stats)
            
            if "error" in result:
                print(f"\n❌ {result['error']}")
            else:
                print(f"\n✓ Distance from {result['from']} to {result['to']}: {result['distance']:.4f}")
                print(f"  Path hops: {result['path_length'] - 1}")
                print(f"  {result['from']} ({result['cluster_info']['start']['dominant_pitch_type']}, "
                      f"{result['cluster_info']['start']['num_pitches']} pitches)")
                print(f"    → {result['to']} ({result['cluster_info']['end']['dominant_pitch_type']}, "
                      f"{result['cluster_info']['end']['num_pitches']} pitches)")
        
        elif choice == '2':
            cluster = input("Enter cluster ID: ").strip()
            n = input("How many nearest neighbors? (default 5): ").strip()
            n = int(n) if n else 5
            
            result = find_nearest_clusters(G, cluster_summary, cluster, n, pitch_stats)
            
            if "error" in result:
                print(f"\n❌ {result['error']}")
            else:
                print(f"\n✓ Nearest clusters to {cluster}:")
                for i, neighbor in enumerate(result['nearest_neighbors'], 1):
                    print(f"  {i}. {neighbor['id']} (distance: {neighbor['distance']:.4f}, "
                          f"{neighbor['dominant_pitch_type']}, {neighbor['num_pitches']} pitches)")
        
        elif choice == '3':
            clusters = list_all_clusters(cluster_summary)
            print(f"\n✓ Total clusters: {len(clusters)}\n")
            print("Top 10 clusters by size:")
            for i, c in enumerate(clusters[:10], 1):
                print(f"  {i:2}. {c['id']:20} {c['pitches']:5} pitches  {c['pitch_type']}  ({c['speed']:.1f} mph)")
            
            if len(clusters) > 10:
                show_more = input(f"\nShow all {len(clusters)} clusters? (y/n): ").strip().lower()
                if show_more == 'y':
                    for c in clusters[10:]:
                        print(f"     {c['id']:20} {c['pitches']:5} pitches  {c['pitch_type']}  ({c['speed']:.1f} mph)")
        
        elif choice == '4':
            cluster = input("Enter cluster ID: ").strip()
            info = get_cluster_info(cluster, cluster_summary, pitch_stats)
            
            if info is None:
                print(f"\n❌ Cluster {cluster} not found")
            else:
                print(f"\n✓ Cluster Info: {cluster}")
                print(f"  Pitches: {info['num_pitches']}")
                print(f"  Dominant Type: {info['dominant_pitch_type']}")
                
                # Physical characteristics
                print(f"\n  Physical Characteristics:")
                print(f"    Release Speed: {info['release_speed']:.2f} mph")
                print(f"    HB (Horizontal Break): {info['HB']:.2f} inches")
                print(f"    IVB (Induced Vertical Break): {info['IVB']:.2f} inches")
                print(f"    Spin Rate: {info['release_spin_rate']:.0f} rpm")
                print(f"    Spin Axis: {info['spin_axis']:.0f}°")
                print(f"    Extension: {info['release_extension']:.2f} feet")
                print(f"    Release X: {info['release_pos_x']:.2f} feet")
                print(f"    Release Y: {info['release_pos_y']:.2f} feet")
                print(f"    Release Z: {info['release_pos_z']:.2f} feet")
                
                # Stuff+ information
                if 'avg_stuff_plus' in info:
                    print(f"\n  Stuff+ Metrics:")
                    print(f"    Average Stuff+: {info['avg_stuff_plus']:.2f}")
                    print(f"    xwOBA Contribution: {info['avg_xwoba_contribution']:.3f}")
                    print(f"    Miss Contribution: {info['avg_miss_contribution']:.3f}")
                    print(f"    Chase Contribution: {info['avg_chase_contribution']:.3f}")
                else:
                    print(f"\n  ⚠ Stuff+ data not available")
        
        elif choice == '5':
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid option. Please select 1-5.")


if __name__ == "__main__":
    interactive_query()
