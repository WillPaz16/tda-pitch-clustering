#!/usr/bin/env python3
"""
Graph-Based Pitcher Consistency Analysis

Implements the user's idea: Find the centroid of a pitcher's delivery for each pitch type,
determine the "center" cluster, then penalize pitches based on graph distance from center.
"""

import csv
import numpy as np
import json
from collections import defaultdict
import math

def load_csv_safe(csv_file):
    """Load CSV data safely."""
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except UnicodeDecodeError:
        with open(csv_file, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    return data

def calculate_centroid(pitches):
    """
    Calculate the centroid (average) of physical characteristics for a set of pitches.

    Returns the average vector of: release_speed, spin_axis, release_spin_rate,
    pred_xw, pred_miss, pred_chase, z_xw, z_miss, z_chase, stuff_plus
    """
    if not pitches:
        return None

    # Physical characteristics to include in centroid calculation
    features = [
        'release_speed', 'spin_axis', 'release_spin_rate',
        'pred_xw', 'pred_miss', 'pred_chase',
        'z_xw', 'z_miss', 'z_chase', 'stuff_plus'
    ]

    centroid = {}
    for feature in features:
        values = []
        for pitch in pitches:
            try:
                val = float(pitch[feature])
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
            except (ValueError, KeyError):
                continue

        if values:
            centroid[feature] = np.mean(values)
        else:
            centroid[feature] = 0.0

    return centroid

def find_center_cluster_by_frequency(pitches):
    """
    Find the center cluster by simply using the most frequent cluster.
    This is simpler and more intuitive than centroid-based approach.
    """
    cluster_counts = {}
    for pitch in pitches:
        cluster = pitch['cluster_id']
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    # Return the most frequent cluster
    return max(cluster_counts.items(), key=lambda x: x[1])[0]

def calculate_graph_distance(cluster1, cluster2, graph_distances=None):
    """
    Calculate the shortest path distance between two clusters in the graph.

    For now, since we don't have the exact graph distances, we'll use a simple
    approximation based on cluster naming similarity.
    """
    if cluster1 == cluster2:
        return 0

    # Simple distance approximation based on cluster names
    # Clusters in the same "cube" are closer
    try:
        c1_parts = cluster1.replace('cube', '').replace('_cluster', '').split('_')
        c2_parts = cluster2.replace('cube', '').replace('_cluster', '').split('_')

        if len(c1_parts) >= 2 and len(c2_parts) >= 2:
            cube1 = c1_parts[0]
            cube2 = c2_parts[0]

            if cube1 == cube2:
                # Same cube, distance based on cluster number difference
                cluster1_num = int(c1_parts[1]) if c1_parts[1].isdigit() else 0
                cluster2_num = int(c2_parts[1]) if c2_parts[1].isdigit() else 0
                return abs(cluster1_num - cluster2_num) + 1  # +1 because same cluster = 0
            else:
                # Different cubes, minimum distance of 2
                return 2
        else:
            return 2  # Default distance for unrecognized format
    except:
        return 2  # Default distance

def calculate_graph_consistency_score(pitches, center_cluster, penalty_function='gaussian', sigma=2.0):
    """
    Calculate consistency score based on graph distance from center.

    penalty_function options:
    - 'inverse': score = 1 / (distance + 1)
    - 'exponential': score = exp(-distance)
    - 'linear': score = max(0, 1 - distance/10)
    - 'gaussian': score = exp(-distance^2 / (2*sigma^2))
    """
    if not pitches:
        return 0.0

    total_score = 0.0

    for pitch in pitches:
        cluster = pitch['cluster_id']
        distance = calculate_graph_distance(center_cluster, cluster)

        if penalty_function == 'inverse':
            # User's suggestion: 1/distance, but avoid division by zero
            score = 1.0 / (distance + 1)  # +1 to avoid division by zero
        elif penalty_function == 'exponential':
            score = math.exp(-distance)
        elif penalty_function == 'linear':
            score = max(0, 1.0 - distance / 10.0)
        elif penalty_function == 'gaussian':
            score = math.exp(-(distance ** 2) / (2.0 * sigma ** 2))
        else:
            score = 1.0 / (distance + 1)

        total_score += score

    # Normalize to 0-100 scale
    max_possible_score = len(pitches) * 1.0  # Best case: all pitches at distance 0
    consistency_score = (total_score / max_possible_score) * 100

    return consistency_score

def analyze_pitcher_graph_consistency(csv_file='pitch_stuffplus_clusters.csv'):
    """
    Main analysis function implementing graph-based consistency scoring.
    """
    print("Loading pitch data...")
    data = load_csv_safe(csv_file)
    print(f"Loaded {len(data)} pitches")

    # Group pitches by pitcher and pitch type
    pitcher_pitchtype_data = defaultdict(lambda: defaultdict(list))

    for pitch in data:
        pitcher = pitch['pitcher_id']
        pitch_type = pitch['pitch_type']
        pitcher_pitchtype_data[pitcher][pitch_type].append(pitch)

    print(f"Found {len(pitcher_pitchtype_data)} pitchers")

    results = []

    print("Analyzing pitcher consistency...")

    for pitcher_id, pitchtype_data in pitcher_pitchtype_data.items():
        pitcher_results = {'pitcher_id': pitcher_id, 'pitch_types': {}}

        for pitch_type, pitches in pitchtype_data.items():
            if len(pitches) < 5:  # Skip pitch types with too few pitches
                continue

            # Find the center cluster (most frequent cluster for this pitch type)
            center_cluster = find_center_cluster_by_frequency(pitches)

            # Calculate graph-based consistency score using a Gaussian distance penalty
            consistency_score = calculate_graph_consistency_score(pitches, center_cluster, 'gaussian', sigma=2.0)

            pitcher_results['pitch_types'][pitch_type] = {
                'consistency_score': consistency_score,
                'center_cluster': center_cluster,
                'num_pitches': len(pitches)
            }

        # Calculate weighted average across pitch types
        if pitcher_results['pitch_types']:
            total_pitches = sum(pt_data['num_pitches'] for pt_data in pitcher_results['pitch_types'].values())
            weighted_score = sum(
                pt_data['consistency_score'] * pt_data['num_pitches']
                for pt_data in pitcher_results['pitch_types'].values()
            ) / total_pitches

            pitcher_results['overall_consistency'] = weighted_score
            pitcher_results['total_pitches'] = total_pitches

            results.append(pitcher_results)

    # Sort by overall consistency
    results.sort(key=lambda x: x['overall_consistency'], reverse=True)

    return results

def print_results(results, top_n=10):
    """Print the top N most consistent pitchers."""
    print(f"\n{'='*60}")
    print("GRAPH-BASED PITCHER CONSISTENCY ANALYSIS")
    print(f"{'='*60}")

    print(f"\nTop {top_n} Most Consistent Pitchers (Graph Distance Method):")
    print("-" * 60)

    for i, result in enumerate(results[:top_n]):
        pitcher_id = result['pitcher_id']
        consistency = result['overall_consistency']
        total_pitches = result['total_pitches']
        num_pitch_types = len(result['pitch_types'])

        print(f"{i+1:2d}. {pitcher_id} - Consistency: {consistency:5.1f}/100  ({num_pitch_types} pitch types, {total_pitches} total pitches)")

        # Show breakdown by pitch type
        for pt, pt_data in result['pitch_types'].items():
            pt_consistency = pt_data['consistency_score']
            pt_pitches = pt_data['num_pitches']
            center_cluster = pt_data['center_cluster']
            print(f"      {pt}: {pt_consistency:5.1f}/100 ({pt_pitches} pitches, center: {center_cluster})")

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    if results:
        consistencies = [r['overall_consistency'] for r in results]
        print(f"Average consistency: {np.mean(consistencies):.1f}/100")
        print(f"Median consistency:  {np.median(consistencies):.1f}/100")
        print(f"Std deviation:       {np.std(consistencies):.1f}")
        print(f"Total pitchers analyzed: {len(results)}")

if __name__ == "__main__":
    results = analyze_pitcher_graph_consistency()
    print_results(results)