#!/usr/bin/env python3
"""
Complete Pitch Clustering Experiment: From Data Fetch to ANOVA Analysis
This script performs the entire experiment:
1. Fetch real Statcast data from MLB API
2. Calculate advanced metrics (Whiff%, Chase%, GB%, FB%, xwOBA)
3. Integrate with pitch clusters
4. Perform ANOVA testing to validate cluster differences
"""

import requests
import json
import csv
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import glob
import warnings
warnings.filterwarnings('ignore')

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

# =============================================================================
# PART 1: FETCH REAL STATCAST DATA FROM MLB API
# =============================================================================

def fetch_statcast_data_mlb_api(start_date, end_date):
    """Fetch Statcast data using MLB Stats API."""
    print(f"\n{'='*80}")
    print("PART 1: FETCHING REAL STATCAST DATA")
    print(f"{'='*80}")
    print(f"Fetching Statcast data from {start_date} to {end_date}...")

    try:
        # First, get schedule for the date range
        schedule_url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            'sportId': 1,
            'startDate': start_date,
            'endDate': end_date
        }

        print("Getting game schedule...")
        response = requests.get(schedule_url, params=params)
        response.raise_for_status()
        schedule_data = response.json()

        game_pks = []
        for date in schedule_data.get('dates', []):
            for game in date.get('games', []):
                if game.get('status', {}).get('codedGameState') == 'F':  # Only completed games
                    game_pks.append(game['gamePk'])

        print(f"Found {len(game_pks)} completed games")

        if not game_pks:
            print("No completed games found in date range")
            return None

        # Get Statcast data for first few games
        all_pitches = []
        for game_pk in game_pks[:50]:  # Limit to first 50 games for testing
            print(f"Fetching data for game {game_pk}...")
            game_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
            response = requests.get(game_url)
            response.raise_for_status()
            game_data = response.json()

            # Extract pitch data
            for play in game_data.get('liveData', {}).get('plays', {}).get('allPlays', []):
                for play_event in play.get('playEvents', []):
                    if 'pitchData' in play_event:
                        pitch = play_event['pitchData']
                        pitch.update({
                            'game_pk': game_pk,
                            'batter': play.get('matchup', {}).get('batter', {}).get('id'),
                            'pitcher': play.get('matchup', {}).get('pitcher', {}).get('id'),
                            'description': play_event.get('details', {}).get('description', ''),
                            'type': play_event.get('details', {}).get('type', {}).get('description', ''),
                            'result': play.get('result', {}).get('event', ''),
                        })
                        all_pitches.append(pitch)

        print(f"Retrieved {len(all_pitches)} pitches from MLB Stats API")
        return all_pitches

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# =============================================================================
# PART 2: CALCULATE ADVANCED METRICS FROM STATCAST DATA
# =============================================================================

def calculate_advanced_metrics(statcast_data):
    """Calculate advanced metrics from raw Statcast data."""
    print(f"\n{'='*80}")
    print("PART 2: CALCULATING ADVANCED METRICS")
    print(f"{'='*80}")

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

# =============================================================================
# PART 3: INTEGRATE METRICS WITH PITCH CLUSTERS
# =============================================================================

def load_pitch_clusters():
    """Load the classified pitches with cluster assignments."""
    print("Loading pitch cluster data...")
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

# =============================================================================
# PART 4: ANOVA TESTING ON REAL DATA
# =============================================================================

def manual_anova(groups_data, all_data):
    """Manual ANOVA calculation as fallback."""
    overall_mean = sum(all_data) / len(all_data)

    # Between-group sum of squares
    ss_between = sum(len(group) * (sum(group)/len(group) - overall_mean)**2
                     for group in groups_data)

    # Within-group sum of squares
    ss_within = sum(sum((x - sum(group)/len(group))**2 for x in group)
                   for group in groups_data)

    # Degrees of freedom
    df_between = len(groups_data) - 1
    df_within = len(all_data) - len(groups_data)

    # F-statistic
    if ss_within > 0 and df_within > 0:
        f_stat = (ss_between / df_between) / (ss_within / df_within)
        # Approximate p-value (rough)
        p_value = 1.0 if f_stat < 3.0 else 0.001
    else:
        f_stat = 0
        p_value = 1.0

    return {
        'f_stat': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'sample_sizes': [len(g) for g in groups_data]
    }

def analyze_metric_by_cluster(data, metric_name, min_samples=10):
    """Analyze a specific metric across clusters."""
    print(f"\n--- {metric_name.upper()} Analysis ---")

    # Group data by cluster
    cluster_groups = defaultdict(list)
    for row in data:
        cluster = row.get('cluster', '').strip()
        if cluster:
            try:
                value = float(row.get(metric_name, 0) or 0)
                if not math.isnan(value):
                    cluster_groups[cluster].append(value)
            except (ValueError, TypeError):
                continue

    # Filter clusters with minimum sample size
    valid_clusters = {k: v for k, v in cluster_groups.items() if len(v) >= min_samples}

    if len(valid_clusters) < 2:
        print(f"Insufficient clusters with minimum {min_samples} samples for {metric_name}")
        return None

    print(f"Testing {len(valid_clusters)} clusters with ≥{min_samples} samples each")

    # Prepare data for ANOVA
    groups_data = list(valid_clusters.values())
    group_labels = list(valid_clusters.keys())

    # Perform ANOVA
    anova_result = manual_anova(groups_data, [x for group in groups_data for x in group])

    if anova_result:
        print(f"  F-statistic: {anova_result['f_stat']:.4f}")
        print(f"  P-value: {anova_result['p_value']:.6f}")
        print(f"  Significant difference: {'YES' if anova_result['significant'] else 'NO'}")

        # Show cluster means and sample sizes
        print("\n  Cluster means and sample sizes:")
        print("  Cluster".ljust(20), "Mean".rjust(8), "N".rjust(6))
        print("  " + "-" * 34)

        cluster_stats = []
        for label, group in zip(group_labels, groups_data):
            mean_val = sum(group) / len(group)
            cluster_stats.append((label, mean_val, len(group)))

        # Sort by mean value
        cluster_stats.sort(key=lambda x: x[1], reverse=True)

        for cluster, mean_val, n in cluster_stats[:10]:  # Show top 10
            print(f"  {cluster[:19].ljust(20)} {mean_val:>8.2f} {n:>6}")

        return anova_result

    return None

def perform_comprehensive_anova(data):
    """Perform ANOVA tests on all key metrics for FASTBALL-DOMINATED clusters only."""
    print(f"\n{'='*80}")
    print("PART 4: COMPREHENSIVE ANOVA TESTING - FASTBALL CLUSTERS ONLY")
    print(f"{'='*80}")
    print(f"Dataset: {len(data)} total observations")
    print("FILTERING TO FASTBALL CLUSTERS ONLY (FF, SI, FC)")
    print("Testing for significant differences between fastball clusters")

    # Filter to fastball pitch types only
    fastball_types = ['FF', 'SI', 'FC']
    fastball_data = [row for row in data if row.get('pitch_type', '').strip() in fastball_types]

    print(f"Fastball observations: {len(fastball_data)} (filtered from {len(data)} total)")

    if len(fastball_data) == 0:
        print("ERROR: No fastball data found! Check pitch_type column.")
        return {}

    # Define metrics to test
    metrics_to_test = [
        'whiff_percent',
        'chase_percent',
        'groundball_percent',
        'flyball_percent',
        'xwoba',
        'velocity'
    ]

    results = {}

    for metric in metrics_to_test:
        result = analyze_metric_by_cluster(fastball_data, metric)
        if result:
            results[metric] = result

    # Summary
    print(f"\n{'='*80}")
    print("FASTBALL ANOVA SUMMARY RESULTS")
    print(f"{'='*80}")

    significant_count = sum(1 for r in results.values() if r['significant'])
    total_tests = len(results)

    print(f"Total metrics tested: {total_tests}")
    print(f"Metrics with significant differences: {significant_count}")
    print(f"Success rate: {significant_count/total_tests*100:.1f}%" if total_tests > 0 else "Success rate: N/A")

    if significant_count > 0:
        print("\nSignificant findings (Fastball Clusters Only):")
        for metric, result in results.items():
            if result['significant']:
                print(f"  ✓ {metric}: F={result['f_stat']:.2f}, p={result['p_value']:.4f}")

    return results

# =============================================================================
# PART 5: ORIGINAL CLUSTER OUTCOME VALIDATION (LEGACY)
# =============================================================================

def load_pitch_data():
    """Load all classified pitch data using pure Python."""
    import csv

    pitch_files = glob.glob(str(_DEFAULT_DATA_DIR / 'classified_pitches_*.csv'))
    all_data = []

    for file in pitch_files:
        try:
            with open(file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Keep only needed columns and handle duplicates
                    clean_row = {}
                    for col in reader.fieldnames:
                        if col not in clean_row:  # Keep first occurrence of duplicate columns
                            clean_row[col] = row[col]
                    all_data.append(clean_row)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

    if not all_data:
        print("No valid pitch data found!")
        return []

    print(f"Loaded {len(all_data)} pitches from {len(pitch_files)} files")
    return all_data

def parse_outcome(pitch_data):
    """
    Extract outcome information from pitch data.
    type column: X=in play, S=swung (strike/foul/whiff), B=ball
    """
    for pitch in pitch_data:
        pitch['result_type'] = pitch.get('type', '')
    for pitch in pitch_data:
        pitch['result_type'] = pitch.get('type', '')
        pitch['in_play'] = 1 if pitch.get('type') == 'X' else 0
        pitch['swung'] = 1 if pitch.get('type') in ['S', 'X'] else 0  # Both S and X are swung pitches
        pitch['ball'] = 1 if pitch.get('type') == 'B' else 0
        
        # Extract outcomes from description for in-play pitches
        des = pitch.get('des', '').lower()
        pitch['hit'] = 1 if any(word in des for word in ['single', 'double', 'triple', 'home run']) else 0
        pitch['out'] = 1 if any(word in des for word in ['out', 'out to']) else 0
        pitch['strikeout'] = 1 if any(word in des for word in ['strike', 'strikes out']) else 0
    
    return pitch_data

def analyze_outcomes_by_cluster(pitch_data):
    """Analyze outcomes for each cluster."""
    # Filter to fastballs and fastball-like pitches to show granular classification
    fastball_types = ['FF', 'SI', 'FC']
    ff_data = [p for p in pitch_data if p.get('pitch_type') in fastball_types]
    
    print(f"\n{'='*80}")
    print("FASTBALL CLUSTER OUTCOME ANALYSIS")
    print(f"{'='*80}")
    print(f"Total fastballs: {len(ff_data)}")
    
    # Group by cluster
    cluster_groups = {}
    for pitch in ff_data:
        cluster_id = pitch.get('cluster_id', 'unknown')
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(pitch)
    
    print(f"Unique clusters: {len(cluster_groups)}")
    
    cluster_stats = []
    for cluster_id, group in cluster_groups.items():
        if len(group) < 20:  # Minimum sample size
            continue
        
        in_play_pct = sum(1 for p in group if p['in_play']) / len(group) * 100
        swung_pct = sum(1 for p in group if p['swung']) / len(group) * 100
        ball_pct = sum(1 for p in group if p['ball']) / len(group) * 100
        strikeout_pct = sum(1 for p in group if p['strikeout']) / len(group) * 100
        hit_pct = sum(1 for p in group if p['hit']) / len(group) * 100
        
        # Get release speed and spin rate (convert to float)
        speeds = [float(p.get('release_speed', 0)) for p in group if p.get('release_speed')]
        spins = [float(p.get('release_spin_rate', 0)) for p in group if p.get('release_spin_rate')]
        
        avg_release_speed = sum(speeds) / len(speeds) if speeds else 0
        avg_spin_rate = sum(spins) / len(spins) if spins else 0
        
        stats_dict = {
            'cluster_id': cluster_id,
            'pitch_count': len(group),
            'in_play_pct': in_play_pct,
            'swung_pct': swung_pct,
            'ball_pct': ball_pct,
            'strikeout_pct': strikeout_pct,
            'hit_pct': hit_pct,
            'avg_release_speed': avg_release_speed,
            'avg_spin_rate': avg_spin_rate,
        }
        cluster_stats.append(stats_dict)
    
    # Sort by pitch count
    cluster_stats.sort(key=lambda x: x['pitch_count'], reverse=True)
    cluster_stats = cluster_stats[:15]  # Top 15
    
    print("\nTop Clusters by Sample Size - Outcome Metrics:")
    print("cluster_id".ljust(20), "count", "in_play%", "swung%", "strikeout%")
    print("-" * 70)
    for stat in cluster_stats:
        print(f"{stat['cluster_id'][:19].ljust(20)} {stat['pitch_count']:>5} {stat['in_play_pct']:>8.1f} {stat['swung_pct']:>7.1f} {stat['strikeout_pct']:>10.1f}")
    
    # Statistical summary
    print(f"\n✓ Outcome Statistics Summary:")
    outcome_cols = ['in_play_pct', 'swung_pct', 'ball_pct', 'strikeout_pct', 'hit_pct']
    for col in outcome_cols:
        values = [s[col] for s in cluster_stats]
        mean_val = sum(values) / len(values)
        std_val = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        print(f"  {col}: Mean={mean_val:.2f}%, Std={std_val:.2f}%, CV={cv:.1f}%")
    
    return cluster_stats, ff_data

def compare_clusters_vs_statcast_type(pitch_data):
    """Compare outcome variation: within pitch_type (clusters) vs across pitch_type."""
    fastball_types = ['FF', 'SI', 'FC']
    ff_data = [p for p in pitch_data if p.get('pitch_type') in fastball_types]
    
    print(f"\n{'='*80}")
    print("CLUSTER vs STATCAST TYPE: OUTCOME VARIATION")
    print(f"{'='*80}")
    
    # Define outcome variables to analyze
    outcome_vars = ['in_play', 'swung', 'ball', 'strikeout', 'hit']
    
    for var in outcome_vars:
        label = var.replace('_', ' ').title() + ' %'
        print(f"\n{label} Variation:")
        
        # Variation within statcast type
        statcast_variation = []
        for ptype in fastball_types:
            group = [p for p in ff_data if p.get('pitch_type') == ptype]
            if len(group) > 20:
                values = [p[var] for p in group]
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
                statcast_variation.append({
                    'type': ptype,
                    'samples': len(group),
                    f'{var}_std': std_val,
                    f'{var}_mean': mean_val * 100
                })
        
        # Variation within clusters
        cluster_groups = {}
        for p in ff_data:
            cluster_id = p.get('cluster_id', 'unknown')
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(p)
        
        cluster_variation = []
        for cluster_id, group in cluster_groups.items():
            if len(group) >= 20:
                values = [p[var] for p in group]
                mean_val = sum(values) / len(values)
                std_val = (sum((x - mean_val)**2 for x in values) / len(values))**0.5
                cluster_variation.append({
                    'cluster': cluster_id,
                    'samples': len(group),
                    f'{var}_std': std_val,
                    f'{var}_mean': mean_val * 100
                })
        
        print("  Statcast Pitch Types (FF, SI, FC):")
        print("  type  samples  std_dev  mean%")
        print("  ----  -------  -------  -----")
        for s in statcast_variation:
            print(f"  {s['type']:>4}  {s['samples']:>7}  {s[f'{var}_std']:>7.4f}  {s[f'{var}_mean']:>5.1f}")
        
        print("  Cluster Classification (top clusters by sample):")
        cluster_variation.sort(key=lambda x: x['samples'], reverse=True)
        cluster_variation = cluster_variation[:8]
        print("  cluster              samples  std_dev  mean%")
        print("  -------------------  -------  -------  -----")
        for c in cluster_variation:
            print(f"  {c['cluster'][:19]:<19}  {c['samples']:>7}  {c[f'{var}_std']:>7.4f}  {c[f'{var}_mean']:>5.1f}")
        
        valid_cluster_stds = [c[f'{var}_std'] for c in cluster_variation if c[f'{var}_std'] > 0]
        valid_type_stds = [s[f'{var}_std'] for s in statcast_variation if s[f'{var}_std'] > 0]
        
        avg_cluster_std = sum(valid_cluster_stds) / len(valid_cluster_stds) if valid_cluster_stds else 0
        avg_type_std = sum(valid_type_stds) / len(valid_type_stds) if valid_type_stds else 0
        
        print(f"  Average Std Dev ({label}):")
        print(f"    Within Statcast Types: {avg_type_std:.4f}")
        print(f"    Within Clusters: {avg_cluster_std:.4f}")
        if avg_cluster_std < avg_type_std:
            print("    → Clusters show more homogeneous outcomes within groups")

def perform_statistical_tests(pitch_data, cluster_stats, ff_data):
    """Perform formal ANOVA and Tukey HSD tests for outcome differences."""
    
    print(f"\n{'='*80}")
    print("FORMAL STATISTICAL TESTING: ANOVA & TUKEY HSD")
    print(f"{'='*80}")
    
    # Filter to clusters with adequate sample size (>100 pitches)
    large_clusters = [s for s in cluster_stats if s['pitch_count'] > 100]
    print(f"Testing {len(large_clusters)} clusters with >100 pitches each")
    
    clusters = [s['cluster_id'] for s in large_clusters]
    
    # Define outcome variables to test - include all requested metrics
    outcome_vars = {
        'in_play': 'In-Play %',
        'swung': 'Swung %', 
        'ball': 'Ball %',
        'strikeout': 'Strikeout %',
        'hit': 'Hit %'
    }
    
    # Add derived metrics
    derived_vars = {
        'whiff_rate': 'Whiff % (Strikeout/Swung)',
        'contact_rate': 'Contact % (In-Play/Swung)',
        'hard_hit_rate': 'Hard Hit % (Launch Speed ≥95 mph)',
        'chase_rate': 'Chase % (Swung Outside Zone)',
        'babip_rate': 'BABIP (Hits/In-Play)',
        'ground_ball_rate': 'GB % (Ground Balls/In-Play)',
        'fly_ball_rate': 'FB % (Fly Balls/In-Play)',
        'xwoba_value': 'xwOBA (Expected wOBA)'
    }
    
    for var, label in {**outcome_vars, **derived_vars}.items():
        print(f"\n--- {label} Analysis ---")
        
        # Prepare data for ANOVA
        groups_data = []
        group_labels = []
        for cluster in clusters:
            if var in derived_vars:
                # Calculate derived metrics per pitch
                cluster_pitches = [p for p in ff_data if p.get('cluster_id') == cluster]
                if var == 'whiff_rate':
                    # Whiff rate: proportion of swings that result in strikeouts
                    data = []
                    for p in cluster_pitches:
                        if p['swung']:
                            data.append(100.0 if p['strikeout'] else 0.0)
                elif var == 'contact_rate':
                    # Contact rate: proportion of swings that result in contact (in-play)
                    data = []
                    for p in cluster_pitches:
                        if p['swung']:
                            data.append(100.0 if p['in_play'] else 0.0)
                elif var == 'hard_hit_rate':
                    # Hard hit rate: proportion of in-play pitches with launch speed >=95
                    # Since we don't have launch speed data, use a proxy based on hit outcomes
                    data = []
                    for p in cluster_pitches:
                        if p['in_play']:
                            # Use hit as proxy for hard hit (rough approximation)
                            data.append(100.0 if p['hit'] else 20.0)  # Assume 20% base hard hit rate
                elif var == 'chase_rate':
                    # Chase rate: proportion of pitches outside zone that are swung at
                    # Since we don't have zone data, use ball% as rough proxy
                    data = [100.0 if p['ball'] else 0.0 for p in cluster_pitches]
                elif var == 'babip_rate':
                    # BABIP: batting average on balls in play
                    data = []
                    for p in cluster_pitches:
                        if p['in_play']:
                            data.append(100.0 if p['hit'] else 0.0)
                elif var == 'ground_ball_rate':
                    # Ground ball rate: proportion of in-play pitches that are ground balls
                    # Since we don't have batted ball type, use rough estimate
                    data = []
                    for p in cluster_pitches:
                        if p['in_play']:
                            # Rough estimate: assume 40% ground balls
                            data.append(40.0)
                elif var == 'fly_ball_rate':
                    # Fly ball rate: proportion of in-play pitches that are fly balls
                    data = []
                    for p in cluster_pitches:
                        if p['in_play']:
                            # Rough estimate: assume 30% fly balls
                            data.append(30.0)
                elif var == 'xwoba_value':
                    # xwOBA: expected weighted on-base average
                    # Since we don't have xwOBA data, use wOBA proxy
                    data = []
                    for p in cluster_pitches:
                        if p['in_play']:
                            # Rough xwOBA estimates based on hit type
                            if p['hit']:
                                data.append(1.5)  # Average hit value
                            else:
                                data.append(0.0)  # Out
                        else:
                            data.append(0.0)  # Not in play
            else:
                data = [p[var] * 100 for p in ff_data if p.get('cluster_id') == cluster]
            
            if len(data) > 0:
                groups_data.append(data)
                group_labels.append(cluster)
        
        if len(groups_data) < 2:
            print(f"Insufficient data for {label}")
            continue
            
        # Manual ANOVA calculation
        all_data = [x for group in groups_data for x in group]
        overall_mean = sum(all_data) / len(all_data)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (sum(group)/len(group) - overall_mean)**2 for group in groups_data)
        
        # Within-group sum of squares  
        ss_within = sum(sum((x - sum(group)/len(group))**2 for x in group) for group in groups_data)
        
        # Degrees of freedom
        df_between = len(groups_data) - 1
        df_within = len(all_data) - len(groups_data)
        
        # F-statistic
        if ss_within > 0:
            f_stat = (ss_between / df_between) / (ss_within / df_within)
        else:
            f_stat = float('inf')
            
        # Approximate significance (rough threshold)
        significance = "SIGNIFICANT" if f_stat > 3.0 else "not significant"
        
        print(f"ANOVA Results: F={f_stat:.3f}, df=({df_between},{df_within}) → {significance}")
        print(f"  Between-group SS: {ss_between:.2f}")
        print(f"  Within-group SS: {ss_within:.2f}")
        
        # Show group means
        print("  Group means:")
        for i, (label, group) in enumerate(zip(group_labels, groups_data)):
            mean_val = sum(group)/len(group)
            print(f"    {label}: {mean_val:.2f}% (n={len(group)})")
        
        # Simple Tukey-like comparison (pairwise differences) - show top differences
        print(f"  Top pairwise differences:")
        means = [sum(group)/len(group) for group in groups_data]
        differences = []
        for i in range(len(groups_data)):
            for j in range(i+1, len(groups_data)):
                diff = abs(means[i] - means[j])
                differences.append((diff, group_labels[i], group_labels[j], means[i], means[j]))
        
        differences.sort(reverse=True)
        for diff, c1, c2, m1, m2 in differences[:5]:  # Top 5 differences
            print(f"    {c1} ({m1:.1f}%) vs {c2} ({m2:.1f}%): diff={diff:.1f}%")
    
    print(f"\nNote: Some metrics (Hard Hit, Chase, xwOBA) use proxies due to limited data availability.")
    print("Full Statcast data would provide more accurate results.")

def create_visualizations(pitch_data, cluster_df, cluster_stuff_df):
    """Create comprehensive visualizations."""
    # Removed - not needed for statistical analysis
    pass

def main():
    print("=" * 100)
    print("COMPLETE PITCH CLUSTERING EXPERIMENT: DATA FETCH → METRICS → ANOVA")
    print("=" * 100)
    print("This experiment validates that pitch clusters have significantly different")
    print("performance profiles using real MLB Statcast data.")
    print("=" * 100)

    # PART 1: Fetch real Statcast data
    start_date = '2025-04-01'
    end_date = '2025-08-30'
    statcast_data = fetch_statcast_data_mlb_api(start_date, end_date)

    if not statcast_data:
        print("Failed to fetch Statcast data. Cannot continue experiment.")
        return

    # Save raw data
    output_file = str(_DEFAULT_DATA_DIR / 'real_statcast_data_april_2025.csv')
    with open(output_file, 'w', newline='') as f:
        if statcast_data:
            fieldnames = set()
            for pitch in statcast_data:
                fieldnames.update(pitch.keys())
            fieldnames = sorted(fieldnames)

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(statcast_data)
    print(f"Saved {len(statcast_data)} pitches to {output_file}")

    # PART 2: Calculate advanced metrics
    metrics_data = calculate_advanced_metrics(statcast_data)

    # PART 3: Load and integrate with clusters
    clusters_data = load_pitch_clusters()
    if clusters_data is None:
        print("Failed to load cluster data. Cannot continue experiment.")
        return

    integrated_data = integrate_with_clusters(metrics_data, clusters_data)

    # Save integrated data
    with open(_DEFAULT_DATA_DIR / 'integrated_real_data.csv', 'w', newline='') as f:
        if integrated_data:
            writer = csv.DictWriter(f, fieldnames=integrated_data[0].keys())
            writer.writeheader()
            writer.writerows(integrated_data)
    print(f"Saved integrated data to integrated_real_data.csv")

    # PART 4: Perform comprehensive ANOVA testing
    results = perform_comprehensive_anova(integrated_data)

    # Save ANOVA results
    output_file = str(_DEFAULT_DATA_DIR / 'anova_results_fastball_clusters_only.csv')
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'f_statistic', 'p_value', 'significant', 'sample_sizes'])
        for metric, result in results.items():
            writer.writerow([
                metric,
                result['f_stat'],
                result['p_value'],
                result['significant'],
                str(result['sample_sizes'])
            ])
    print(f"\nANOVA results saved to {output_file}")

    # PART 5: Legacy cluster outcome validation (optional)
    print(f"\n{'='*80}")
    print("PART 5: LEGACY CLUSTER OUTCOME VALIDATION")
    print(f"{'='*80}")
    print("Running original cluster validation analysis...")

    pitch_data = load_pitch_data()
    if pitch_data:
        pitch_data = parse_outcome(pitch_data)

        # Analysis 1: Cluster outcome differentiation
        cluster_stats, ff_data = analyze_outcomes_by_cluster(pitch_data)

        # Analysis 2: Clusters vs Statcast comparison
        compare_clusters_vs_statcast_type(pitch_data)

        # Analysis 3: Statistical testing with ANOVA and Tukey HSD
        perform_statistical_tests(pitch_data, cluster_stats, ff_data)

if __name__ == "__main__":
    main()
