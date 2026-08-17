"""
Pitcher Consistency Metric Calculator
Analyzes how consistently each pitcher's pitches land in the same clusters by pitch type.
Measures delivery repeatability and identifies high-variance pitchers.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import csv

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def load_csv_safe(filename):
    """Load CSV file using basic Python csv module to avoid pandas issues."""
    print(f"Loading {filename}...")
    data = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and row.get('pitcher_id') and row.get('pitch_type') and row.get('cluster_id'):
                    data.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    if not data:
        return None
    
    print(f"Loaded {len(data)} valid pitch records")
    return data


def calculate_cluster_distribution(pitcher_pitches, pitch_type):
    """Calculate cluster distribution for a specific pitch type."""
    pitch_data = [p for p in pitcher_pitches if p.get('pitch_type') == pitch_type]
    
    if not pitch_data:
        return None
    
    # Count clusters
    cluster_counts = {}
    for pitch in pitch_data:
        cluster = pitch['cluster_id']
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    
    # Sort by count and convert to percentages
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    total = sum(c[1] for c in sorted_clusters)
    
    cluster_pcts = {cluster: round(count / total * 100, 2) for cluster, count in sorted_clusters}
    
    return {
        'pitch_type': pitch_type,
        'total_pitches': total,
        'num_clusters': len(cluster_counts),
        'cluster_distribution': cluster_pcts,
        'top_cluster': sorted_clusters[0][0],
        'top_cluster_pct': cluster_pcts[sorted_clusters[0][0]],
        'top_3_clusters_pct': sum(list(cluster_pcts.values())[:3]),
        'top_5_clusters_pct': sum(list(cluster_pcts.values())[:5])
    }


def calculate_consistency_metrics(cluster_distribution):
    """
    Calculate consistency metrics for a pitch type.
    
    Returns:
        dict: Contains consistency scores
        - concentration: % in top cluster (higher = more consistent)
        - entropy: Shannon entropy of distribution (lower = more consistent)
        - consistency_score: 0-100 scale (higher = more consistent)
    """
    if cluster_distribution is None:
        return None
    
    # cluster_pcts is now a dict, not a pandas Series
    cluster_pcts = cluster_distribution['cluster_distribution']
    pcts_list = list(cluster_pcts.values())
    
    # Normalize to probabilities for entropy calculation
    probs = [p / 100.0 for p in pcts_list]
    
    # Shannon entropy calculation (without scipy)
    # H = -sum(p * log(p)) for p > 0
    # Lower entropy = more consistent
    ent = -sum([p * np.log(p) for p in probs if p > 0])
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 0
    
    # Normalize entropy to 0-100 scale (0 = consistent, 100 = random)
    entropy_normalized = (ent / max_entropy * 100) if max_entropy > 0 else 0
    
    # CONSISTENCY SCORE: Use concentration (% in top cluster) directly
    # This is more intuitive: 80% in top cluster = 80/100 consistency
    # This is what actually matters for pitcher repeatability
    top_cluster_pct = cluster_distribution['top_cluster_pct']
    consistency_score = top_cluster_pct  # 0-100, direct concentration metric
    
    return {
        'concentration': cluster_distribution['top_cluster_pct'],  # % in top cluster
        'entropy': ent,
        'entropy_pct': entropy_normalized,
        'consistency_score': consistency_score,  # 0-100, % in top cluster (higher = more consistent)
        'num_clusters_used': cluster_distribution['num_clusters']
    }


def generate_pitcher_consistency_report(pitch_stuffplus_clusters_csv):
    """
    Generate comprehensive consistency report for all pitchers.
    
    Returns:
        list: List of dicts with consistency metrics for each pitcher × pitch_type combo
    """
    # Use safe CSV loader - returns list of dicts
    data = load_csv_safe(pitch_stuffplus_clusters_csv)
    
    if not data:
        raise ValueError("Could not load data from CSV file")
    
    if not data[0].get('pitcher_id'):
        raise ValueError("pitcher_id column not found in dataset")
    
    # Group pitches by pitcher_id
    pitcher_pitches_map = {}
    unique_pitch_types = set()
    
    for pitch in data:
        pitcher = pitch['pitcher_id']
        pitch_type = pitch.get('pitch_type')
        
        if pitcher not in pitcher_pitches_map:
            pitcher_pitches_map[pitcher] = []
        pitcher_pitches_map[pitcher].append(pitch)
        unique_pitch_types.add(pitch_type)
    
    print(f"Found {len(pitcher_pitches_map)} pitchers with {len(unique_pitch_types)} pitch types")
    
    all_results = []
    
    for pitcher_id, pitcher_pitches in pitcher_pitches_map.items():
        pitcher_pitch_types = set(p.get('pitch_type') for p in pitcher_pitches)
        
        # Calculate metrics for each pitch type
        for pitch_type in pitcher_pitch_types:
            cluster_dist = calculate_cluster_distribution(pitcher_pitches, pitch_type)
            if not cluster_dist:
                continue
            
            consistency_metrics = calculate_consistency_metrics(cluster_dist)
            
            # Create entry for this pitcher × pitch_type combination
            entry = {
                'pitcher_id': pitcher_id,
                'pitch_type': pitch_type,
                'pitch_count': cluster_dist['total_pitches'],
                'num_clusters': cluster_dist['num_clusters'],
                'top_cluster': cluster_dist['top_cluster'],
                'top_cluster_pct': cluster_dist['top_cluster_pct'],
                'top_3_clusters_pct': cluster_dist['top_3_clusters_pct'],
                'top_5_clusters_pct': cluster_dist['top_5_clusters_pct'],
                'concentration': consistency_metrics['concentration'],
                'entropy': consistency_metrics['entropy'],
                'consistency_score': consistency_metrics['consistency_score'],
                'num_clusters_used': consistency_metrics['num_clusters_used']
            }
            
            # Add top 3 clusters to report
            for rank, (cluster, pct) in enumerate(list(cluster_dist['cluster_distribution'].items())[:3], 1):
                entry[f'cluster_rank{rank}'] = cluster
                entry[f'cluster_rank{rank}_pct'] = pct
            
            all_results.append(entry)
    
    return all_results


def generate_pitcher_summary_report(consistency_list):
    """
    Generate pitcher-level summary (averaging across pitch types).
    
    Returns:
        list of dicts with overall consistency metrics per pitcher
    """
    pitcher_data = {}
    
    for entry in consistency_list:
        pitcher = entry['pitcher_id']
        if pitcher not in pitcher_data:
            pitcher_data[pitcher] = []
        pitcher_data[pitcher].append(entry)
    
    pitcher_summary = []
    
    for pitcher_id, entries in pitcher_data.items():
        consistency_scores = [e['consistency_score'] for e in entries]
        pitch_counts = [e['pitch_count'] for e in entries]
        entropies = [e['entropy'] for e in entries]
        num_clusters = [e['num_clusters'] for e in entries]
        concentrations = [e['concentration'] for e in entries]
        
        total_pitches = sum(pitch_counts)
        
        # Weighted average consistency by pitch count
        weighted_consistency = sum(score * count for score, count in zip(consistency_scores, pitch_counts)) / total_pitches
        
        summary = {
            'pitcher_id': pitcher_id,
            'total_pitches': total_pitches,
            'num_pitch_types': len(entries),
            'avg_consistency_score': round(weighted_consistency, 2),
            'min_consistency_score': round(min(consistency_scores), 2),
            'max_consistency_score': round(max(consistency_scores), 2),
            'std_consistency_score': round(np.std(consistency_scores), 2),
            'avg_entropy': round(sum(entropies) / len(entropies), 3),
            'avg_num_clusters': round(sum(num_clusters) / len(num_clusters), 1),
            'avg_concentration': round(sum(concentrations) / len(concentrations), 2)
        }
        
        pitcher_summary.append(summary)
    
    # Sort by consistency score
    pitcher_summary.sort(key=lambda x: x['avg_consistency_score'], reverse=True)
    return pitcher_summary


def generate_pitch_type_summary_report(consistency_list):
    """
    Generate pitch-type-level summary (averaging across pitchers).
    
    Returns:
        list of dicts with consistency metrics per pitch type
    """
    pitchtype_data = {}
    
    for entry in consistency_list:
        pt = entry['pitch_type']
        if pt not in pitchtype_data:
            pitchtype_data[pt] = []
        pitchtype_data[pt].append(entry)
    
    pitchtype_summary = []
    
    for pitch_type, entries in pitchtype_data.items():
        consistency_scores = [e['consistency_score'] for e in entries]
        pitch_counts = [e['pitch_count'] for e in entries]
        entropies = [e['entropy'] for e in entries]
        num_clusters = [e['num_clusters'] for e in entries]
        concentrations = [e['concentration'] for e in entries]
        
        total_pitches = sum(pitch_counts)
        
        # Weighted average consistency by pitch count
        weighted_consistency = sum(score * count for score, count in zip(consistency_scores, pitch_counts)) / total_pitches
        
        summary = {
            'pitch_type': pitch_type,
            'num_pitchers': len(entries),
            'total_pitches': total_pitches,
            'avg_consistency_score': round(weighted_consistency, 2),
            'std_consistency_score': round(np.std(consistency_scores), 2),
            'min_consistency_score': round(min(consistency_scores), 2),
            'max_consistency_score': round(max(consistency_scores), 2),
            'avg_entropy': round(sum(entropies) / len(entropies), 3),
            'avg_num_clusters': round(sum(num_clusters) / len(num_clusters), 1),
            'avg_concentration': round(sum(concentrations) / len(concentrations), 2)
        }
        
        pitchtype_summary.append(summary)
    
    # Sort by consistency score
    pitchtype_summary.sort(key=lambda x: x['avg_consistency_score'], reverse=True)
    return pitchtype_summary


def save_list_to_csv(data, filename, fieldnames=None):
    """Save list of dicts to CSV file."""
    if not data:
        print(f"No data to save to {filename}")
        return
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"✓ Saved to {filename} ({len(data)} rows)")
    except Exception as e:
        print(f"Error saving {filename}: {e}")


def main():
    """Main execution - generate all consistency reports."""
    
    # Generate base consistency metrics
    print("\n" + "="*80)
    print("PITCHER CONSISTENCY ANALYSIS")
    print("="*80)
    
    consistency_list = generate_pitcher_consistency_report(str(_DEFAULT_DATA_DIR / 'pitch_stuffplus_clusters.csv'))
    
    # Sort by pitcher and pitch type
    consistency_list.sort(key=lambda x: (x['pitcher_id'], x['pitch_type']))
    
    # Generate summaries
    pitcher_summary = generate_pitcher_summary_report(consistency_list)
    pitchtype_summary = generate_pitch_type_summary_report(consistency_list)
    
    # Save to CSV
    save_list_to_csv(consistency_list, str(_DEFAULT_DATA_DIR / 'pitcher_consistency_detailed.csv'))
    save_list_to_csv(pitcher_summary, str(_DEFAULT_DATA_DIR / 'pitcher_consistency_summary.csv'))
    save_list_to_csv(pitchtype_summary, str(_DEFAULT_DATA_DIR / 'pitch_type_consistency_summary.csv'))
    
    print("\n" + "="*80)
    print("PITCHER-LEVEL STATISTICS (averaged across pitch types)")
    print("="*80)
    
    print(f"\nTop 10 Most Consistent Pitchers:")
    for i, row in enumerate(pitcher_summary[:10], 1):
        print(f"  {i:2}. {row['pitcher_id']:15} Consistency: {row['avg_consistency_score']:6.1f}/100  "
              f"({int(row['num_pitch_types'])} pitch types, {int(row['total_pitches'])} total pitches)")
    
    print(f"\nTop 10 Least Consistent Pitchers (High Variance):")
    for i, row in enumerate(pitcher_summary[-10:], 1):
        print(f"  {i:2}. {row['pitcher_id']:15} Consistency: {row['avg_consistency_score']:6.1f}/100  "
              f"({int(row['num_pitch_types'])} pitch types, {int(row['total_pitches'])} total pitches)")
    
    print("\n" + "="*80)
    print("PITCH-TYPE-LEVEL STATISTICS (averaged across pitchers)")
    print("="*80)
    
    print("\nConsistency by Pitch Type (sorted by consistency):")
    for row in pitchtype_summary:
        print(f"  {row['pitch_type']:4}  Consistency: {row['avg_consistency_score']:6.1f}/100  "
              f"Avg # clusters: {row['avg_num_clusters']:4.1f}  "
              f"({int(row['num_pitchers'])} pitchers, {int(row['total_pitches'])} total pitches)")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Analysis of consistency scores
    consistency_scores = [p['avg_consistency_score'] for p in pitcher_summary]
    low_var_count = len([s for s in consistency_scores if s > 75])
    high_var_count = len([s for s in consistency_scores if s < 50])
    
    print(f"\nConsistent Pitchers (score > 75): {low_var_count} pitchers")
    print(f"High Variance Pitchers (score < 50): {high_var_count} pitchers")
    print(f"\nAverage Pitcher Consistency: {np.mean(consistency_scores):.1f}/100")
    print(f"Median Pitcher Consistency: {np.median(consistency_scores):.1f}/100")
    print(f"Std Dev of Consistency: {np.std(consistency_scores):.2f}")
    
    # Which pitch type is most/least consistent
    most_consistent_pt = pitchtype_summary[0]
    least_consistent_pt = pitchtype_summary[-1]
    
    print(f"\nMost Consistent Pitch Type: {most_consistent_pt['pitch_type']} ({most_consistent_pt['avg_consistency_score']:.1f}/100)")
    print(f"Least Consistent Pitch Type: {least_consistent_pt['pitch_type']} ({least_consistent_pt['avg_consistency_score']:.1f}/100)")
    
    print("\n✓ Analysis complete. Check CSV files for details.")


if __name__ == "__main__":
    main()
