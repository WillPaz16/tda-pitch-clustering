#!/usr/bin/env python3
"""
Calculate advanced metrics from real Statcast data and integrate with pitch clusters.
"""

import csv
from pathlib import Path
from collections import defaultdict

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

def load_statcast_data():
    """Load the real Statcast data we retrieved."""
    print("Loading real Statcast data...")
    try:
        data = []
        with open(_DEFAULT_DATA_DIR / 'real_statcast_data_april_2023.csv', 'r') as f:  # Use the full month data
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"Loaded {len(data)} pitches")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_pitch_clusters():
    """Load the classified pitches with cluster assignments."""
    print("Loading pitch cluster data...")
    import glob
    cluster_files = glob.glob(str(_DEFAULT_DATA_DIR / 'classified_pitches_*.csv'))
    if not cluster_files:
        print("No classified pitches file found")
        return None

    latest_file = max(cluster_files)
    print(f"Using cluster file: {latest_file}")

    try:
        data = []
        with open(latest_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"Loaded {len(data)} classified pitches")
        return data
    except Exception as e:
        print(f"Error loading cluster data: {e}")
        return None

def calculate_advanced_metrics(statcast_data):
    """Calculate advanced metrics from raw Statcast data."""
    print("Calculating advanced metrics...")

    # Group by pitcher for pitcher-level metrics
    pitcher_metrics = defaultdict(lambda: {
        'total_pitches': 0,
        'swings': 0,
        'whiffs': 0,
        'chases': 0,
        'balls_in_play': 0,
        'ground_balls': 0,
        'fly_balls': 0,
        'hard_hits': 0,
        'xwoba_numerator': 0,
        'xwoba_denominator': 0
    })

    for pitch in statcast_data:
        pitcher_id = pitch.get('pitcher', '')
        if not pitcher_id:
            continue

        metrics = pitcher_metrics[pitcher_id]
        metrics['total_pitches'] += 1

        description = str(pitch.get('description', '')).lower()
        result = str(pitch.get('result', '')).lower()

        # Whiff% calculation (swinging strikes)
        if 'strike' in description and 'swing' in description:
            metrics['swings'] += 1
            metrics['whiffs'] += 1
        elif 'swing' in description or 'foul' in description:
            metrics['swings'] += 1

        # Chase% calculation (swings at pitches outside zone)
        try:
            zone = int(pitch.get('zone', 0))
            if zone not in [1, 2, 3, 4, 5, 6, 7, 8, 9]:  # Outside zone
                if 'swing' in description or 'foul' in description:
                    metrics['chases'] += 1
        except:
            pass

        # GB%/FB% calculation
        if 'ground' in result or 'ground' in description:
            metrics['balls_in_play'] += 1
            metrics['ground_balls'] += 1
        elif 'fly' in result or 'fly' in description or 'pop' in result:
            metrics['balls_in_play'] += 1
            metrics['fly_balls'] += 1

        # xwOBA calculation (simplified)
        if 'hit' in result or 'single' in result or 'double' in result or 'triple' in result or 'home' in result:
            if 'single' in result:
                metrics['xwoba_numerator'] += 0.9
            elif 'double' in result:
                metrics['xwoba_numerator'] += 1.25
            elif 'triple' in result:
                metrics['xwoba_numerator'] += 1.6
            elif 'home' in result:
                metrics['xwoba_numerator'] += 2.0
            metrics['xwoba_denominator'] += 1

    # Calculate final metrics
    results = []
    for pitcher_id, metrics in pitcher_metrics.items():
        if metrics['total_pitches'] >= 10:  # Minimum pitches threshold
            result = {
                'pitcher_id': pitcher_id,
                'total_pitches': metrics['total_pitches'],
                'whiff_percent': (metrics['whiffs'] / metrics['swings'] * 100) if metrics['swings'] > 0 else 0,
                'chase_percent': (metrics['chases'] / metrics['total_pitches'] * 100) if metrics['total_pitches'] > 0 else 0,
                'groundball_percent': (metrics['ground_balls'] / metrics['balls_in_play'] * 100) if metrics['balls_in_play'] > 0 else 0,
                'flyball_percent': (metrics['fly_balls'] / metrics['balls_in_play'] * 100) if metrics['balls_in_play'] > 0 else 0,
                'hard_hit_percent': 0,  # Would need launch speed data
                'xwoba': (metrics['xwoba_numerator'] / metrics['xwoba_denominator']) if metrics['xwoba_denominator'] > 0 else 0
            }
            results.append(result)

    print(f"Calculated metrics for {len(results)} pitchers")
    return results

def integrate_with_clusters(metrics_data, clusters_data):
    """Integrate advanced metrics with pitch cluster data."""
    print("Integrating metrics with pitch clusters...")

    # Create lookup for metrics by pitcher_id
    metrics_lookup = {str(m['pitcher_id']): m for m in metrics_data}

    # Group clusters by pitcher
    pitcher_clusters = defaultdict(list)
    for row in clusters_data:
        pitcher_id = str(row.get('pitcher_id') or row.get('pitcher', ''))
        if pitcher_id:
            pitcher_clusters[pitcher_id].append({
                'pitch_type': row.get('pitch_type', ''),
                'cluster': row.get('cluster_id', ''),  # Use cluster_id column
                'velocity': float(row.get('release_speed', 0) or 0),
                'stuff_plus': 0.0  # Not available in this dataset
            })

    # Merge with metrics
    integrated_data = []
    for pitcher_id, clusters in pitcher_clusters.items():
        metrics = metrics_lookup.get(pitcher_id)
        if metrics:
            for cluster_info in clusters:
                integrated_row = {
                    'pitcher_id': pitcher_id,
                    'pitch_type': cluster_info['pitch_type'],
                    'cluster': cluster_info['cluster'],
                    'velocity': cluster_info['velocity'],
                    'stuff_plus': cluster_info['stuff_plus'],
                    'whiff_percent': metrics['whiff_percent'],
                    'chase_percent': metrics['chase_percent'],
                    'groundball_percent': metrics['groundball_percent'],
                    'flyball_percent': metrics['flyball_percent'],
                    'hard_hit_percent': metrics['hard_hit_percent'],
                    'xwoba': metrics['xwoba'],
                    'total_pitches': metrics['total_pitches']
                }
                integrated_data.append(integrated_row)

    print(f"Integrated data has {len(integrated_data)} rows")
    return integrated_data

def perform_real_data_analysis(integrated_data):
    """Perform ANOVA analysis on real data."""
    print("Performing statistical analysis on real data...")

    # Group by cluster and calculate means
    cluster_stats = defaultdict(lambda: defaultdict(list))

    for row in integrated_data:
        cluster = row['cluster']
        cluster_stats[cluster]['whiff_percent'].append(row['whiff_percent'])
        cluster_stats[cluster]['chase_percent'].append(row['chase_percent'])
        cluster_stats[cluster]['groundball_percent'].append(row['groundball_percent'])
        cluster_stats[cluster]['flyball_percent'].append(row['flyball_percent'])
        cluster_stats[cluster]['xwoba'].append(row['xwoba'])
        cluster_stats[cluster]['stuff_plus'].append(row['stuff_plus'])
        cluster_stats[cluster]['velocity'].append(row['velocity'])

    # Calculate means
    results = []
    for cluster, metrics in cluster_stats.items():
        result = {
            'cluster': cluster,
            'whiff_percent_mean': sum(metrics['whiff_percent']) / len(metrics['whiff_percent']) if metrics['whiff_percent'] else 0,
            'chase_percent_mean': sum(metrics['chase_percent']) / len(metrics['chase_percent']) if metrics['chase_percent'] else 0,
            'groundball_percent_mean': sum(metrics['groundball_percent']) / len(metrics['groundball_percent']) if metrics['groundball_percent'] else 0,
            'flyball_percent_mean': sum(metrics['flyball_percent']) / len(metrics['flyball_percent']) if metrics['flyball_percent'] else 0,
            'xwoba_mean': sum(metrics['xwoba']) / len(metrics['xwoba']) if metrics['xwoba'] else 0,
            'stuff_plus_mean': sum(metrics['stuff_plus']) / len(metrics['stuff_plus']) if metrics['stuff_plus'] else 0,
            'velocity_mean': sum(metrics['velocity']) / len(metrics['velocity']) if metrics['velocity'] else 0,
            'sample_size': len(metrics['whiff_percent'])
        }
        results.append(result)

    print("\nCluster Statistics (Real Data):")
    print("Cluster | Whiff% | Chase% | GB% | FB% | xwOBA | Stuff+ | Velocity | N")
    print("-" * 70)
    for result in sorted(results, key=lambda x: x['cluster']):
        print(f"{result['cluster']:7} | {result['whiff_percent_mean']:6.1f} | {result['chase_percent_mean']:6.1f} | {result['groundball_percent_mean']:4.1f} | {result['flyball_percent_mean']:4.1f} | {result['xwoba_mean']:5.3f} | {result['stuff_plus_mean']:6.1f} | {result['velocity_mean']:8.1f} | {result['sample_size']}")

    # Save results
    with open(_DEFAULT_DATA_DIR / 'real_data_cluster_analysis.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['cluster', 'whiff_percent_mean', 'chase_percent_mean', 'groundball_percent_mean', 'flyball_percent_mean', 'xwoba_mean', 'stuff_plus_mean', 'velocity_mean', 'sample_size'])
        writer.writeheader()
        writer.writerows(results)

    with open(_DEFAULT_DATA_DIR / 'integrated_real_data.csv', 'w', newline='') as f:
        if integrated_data:
            writer = csv.DictWriter(f, fieldnames=integrated_data[0].keys())
            writer.writeheader()
            writer.writerows(integrated_data)

    return results

def main():
    print("=== Real Statcast Data Integration and Analysis ===")

    # Load data
    statcast_df = load_statcast_data()
    clusters_df = load_pitch_clusters()

    if statcast_df is None or clusters_df is None:
        print("Missing required data files")
        return

    # Calculate advanced metrics
    metrics_df = calculate_advanced_metrics(statcast_df)

    # Integrate with clusters
    integrated_df = integrate_with_clusters(metrics_df, clusters_df)

    # Perform analysis
    results = perform_real_data_analysis(integrated_df)

    print("\n=== Analysis Complete ===")
    print("Results saved to:")
    print("- real_data_cluster_analysis.csv")
    print("- integrated_real_data.csv")

if __name__ == "__main__":
    main()