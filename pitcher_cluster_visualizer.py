#!/usr/bin/env python3
"""
Pitcher Cluster Distribution Visualizer

Creates visualizations showing how each pitch type is distributed across TDA clusters
for individual pitchers. Helps understand delivery consistency and pitch clustering patterns.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')

def load_csv_safe(csv_file):
    """Load CSV data safely, handling potential encoding issues."""
    data = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except UnicodeDecodeError:
        # Try different encoding
        with open(csv_file, 'r', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

    return data

def get_pitcher_data(data, pitcher_id):
    """Extract all pitches for a specific pitcher."""
    return [row for row in data if str(row['pitcher_id']) == str(pitcher_id)]

def create_pitch_type_cluster_matrix(pitcher_data):
    """
    Create a matrix showing pitch type distribution across clusters.

    Returns:
        dict: pitch_type -> {cluster_id: count}
    """
    matrix = defaultdict(lambda: defaultdict(int))

    for pitch in pitcher_data:
        pitch_type = pitch['pitch_type']
        cluster_id = pitch['cluster_id']
        matrix[pitch_type][cluster_id] += 1

    return dict(matrix)

def plot_pitch_type_distributions(pitcher_data, pitcher_id, save_path=None):
    """
    Create comprehensive visualization of pitch type cluster distributions for a pitcher.
    """
    # Get pitch type cluster matrix
    cluster_matrix = create_pitch_type_cluster_matrix(pitcher_data)

    # Get unique pitch types and clusters
    pitch_types = sorted(cluster_matrix.keys())
    all_clusters = set()
    for pt_data in cluster_matrix.values():
        all_clusters.update(pt_data.keys())
    clusters = sorted(all_clusters)

    # Create figure with subplots
    n_pitch_types = len(pitch_types)
    fig, axes = plt.subplots(n_pitch_types, 1, figsize=(15, 4*n_pitch_types))
    if n_pitch_types == 1:
        axes = [axes]

    fig.suptitle(f'Pitch Type Cluster Distributions - Pitcher {pitcher_id}', fontsize=16, fontweight='bold')

    # Color map for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, len(clusters)))

    for idx, pitch_type in enumerate(pitch_types):
        ax = axes[idx]

        # Get cluster counts for this pitch type
        cluster_counts = cluster_matrix[pitch_type]
        counts = [cluster_counts.get(cluster, 0) for cluster in clusters]
        total_pitches = sum(counts)

        # Create horizontal bar chart
        bars = ax.barh(clusters, counts, color=colors, alpha=0.7)

        # Add percentage labels
        for i, (cluster, count) in enumerate(zip(clusters, counts)):
            if count > 0:
                percentage = (count / total_pitches) * 100
                ax.text(count + max(counts)*0.01, i, '.1f',
                       va='center', fontsize=9)

        # Formatting
        ax.set_title(f'{pitch_type} Distribution (n={total_pitches})', fontweight='bold')
        ax.set_xlabel('Number of Pitches')
        ax.set_ylabel('Cluster ID')
        ax.grid(True, alpha=0.3)

        # Set x-axis to start from 0
        ax.set_xlim(0, max(counts) * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def plot_pitch_type_heatmap(pitcher_data, pitcher_id, save_path=None):
    """
    Create a heatmap showing pitch type vs cluster distribution using matplotlib.
    """
    cluster_matrix = create_pitch_type_cluster_matrix(pitcher_data)

    # Get all pitch types and clusters
    pitch_types = sorted(cluster_matrix.keys())
    all_clusters = set()
    for pt_data in cluster_matrix.values():
        all_clusters.update(pt_data.keys())
    clusters = sorted(all_clusters)

    # Create data matrix
    data_matrix = []
    for pt in pitch_types:
        row = [cluster_matrix[pt].get(cluster, 0) for cluster in clusters]
        data_matrix.append(row)

    # Convert to percentages
    data_matrix_pct = []
    for row in data_matrix:
        total = sum(row)
        if total > 0:
            pct_row = [(x / total) * 100 for x in row]
        else:
            pct_row = [0] * len(row)
        data_matrix_pct.append(pct_row)

    # Create heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(16, len(pitch_types) * 0.8 + 2))

    # Create the heatmap
    im = ax.imshow(data_matrix_pct, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage of Pitches (%)', rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_yticks(np.arange(len(pitch_types)))
    ax.set_xticklabels(clusters)
    ax.set_yticklabels(pitch_types)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(pitch_types)):
        for j in range(len(clusters)):
            if data_matrix[i][j] > 0:  # Only annotate non-zero values
                text = ax.text(j, i, '.1f',
                             ha="center", va="center", color="black", fontsize=8)

    ax.set_title(f'Pitch Type vs Cluster Distribution Heatmap - Pitcher {pitcher_id}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Pitch Type')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()

def plot_consistency_comparison(pitcher_data, pitcher_id, save_path=None):
    """
    Create a bar chart comparing consistency scores across pitch types.
    """
    cluster_matrix = create_pitch_type_cluster_matrix(pitcher_data)

    consistency_data = []

    for pitch_type, cluster_counts in cluster_matrix.items():
        total_pitches = sum(cluster_counts.values())

        if total_pitches > 0:
            # Calculate concentration (percentage in top cluster)
            sorted_counts = sorted(cluster_counts.values(), reverse=True)
            top_cluster_pct = (sorted_counts[0] / total_pitches) * 100

            consistency_data.append({
                'pitch_type': pitch_type,
                'consistency_score': top_cluster_pct,
                'total_pitches': total_pitches,
                'num_clusters': len(cluster_counts)
            })

    if not consistency_data:
        print("No data to plot")
        return

    # Sort by consistency score
    consistency_data.sort(key=lambda x: x['consistency_score'], reverse=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    pitch_types = [d['pitch_type'] for d in consistency_data]
    scores = [d['consistency_score'] for d in consistency_data]
    pitch_counts = [d['total_pitches'] for d in consistency_data]

    bars = ax.bar(pitch_types, scores, alpha=0.7, color='skyblue')

    # Add pitch count labels on bars
    for bar, count in zip(bars, pitch_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    ax.set_title(f'Pitch Type Consistency Scores - Pitcher {pitcher_id}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Pitch Type')
    ax.set_ylabel('Consistency Score (% in top cluster)')
    ax.set_ylim(0, max(scores) * 1.1)
    ax.grid(True, alpha=0.3)

    # Rotate x labels if many pitch types
    if len(pitch_types) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved consistency plot to {save_path}")
    else:
        plt.show()

def analyze_pitcher(pitcher_id, csv_file='pitch_stuffplus_clusters.csv'):
    """
    Main function to analyze and visualize a pitcher's cluster distributions.
    """
    print(f"Loading data for pitcher {pitcher_id}...")

    # Load data
    data = load_csv_safe(csv_file)
    print(f"Loaded {len(data)} total pitches")

    pitcher_data = get_pitcher_data(data, pitcher_id)
    print(f"Found {len(pitcher_data)} pitches for pitcher {pitcher_id}")

    if not pitcher_data:
        print(f"No data found for pitcher {pitcher_id}")
        # Debug: show first few pitcher IDs
        sample_pitchers = set(str(row['pitcher_id']) for row in data[:10])
        print(f"Sample pitcher IDs in data: {sample_pitchers}")
        return

    print(f"Found {len(pitcher_data)} pitches for pitcher {pitcher_id}")

    # Get pitch type summary
    pitch_types = set(p['pitch_type'] for p in pitcher_data)
    print(f"Pitch types: {sorted(pitch_types)}")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Individual pitch type distributions
    plot_pitch_type_distributions(pitcher_data, pitcher_id,
                                save_path=f'pitcher_{pitcher_id}_distributions.png')

    # 2. Heatmap
    plot_pitch_type_heatmap(pitcher_data, pitcher_id,
                           save_path=f'pitcher_{pitcher_id}_heatmap.png')

    # 3. Consistency comparison
    plot_consistency_comparison(pitcher_data, pitcher_id,
                              save_path=f'pitcher_{pitcher_id}_consistency.png')

    print("✓ Visualizations complete!")

if __name__ == "__main__":
    # Example usage - analyze a pitcher with high pitch volume
    # You can change this pitcher ID to analyze different pitchers

    # Let's analyze pitcher 680885 (792 pitches - highest volume)
    analyze_pitcher('680885')