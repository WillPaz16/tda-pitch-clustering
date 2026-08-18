#!/usr/bin/env python3
"""
ANOVA Testing on Real Statcast Data: Test for significant differences between pitch clusters
using advanced metrics (Whiff%, Chase%, GB%, FB%, xwOBA, etc.)
"""

import csv
import math
from pathlib import Path
from collections import defaultdict
from scipy import stats

_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

def load_integrated_data():
    """Load the integrated real data."""
    print("Loading integrated real data...")
    data = []
    try:
        with open(_DEFAULT_DATA_DIR / 'integrated_real_data.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        print(f"Loaded {len(data)} integrated data points")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_anova_test(groups_data, group_labels, metric_name):
    """Perform ANOVA test on grouped data using manual calculation."""
    if len(groups_data) < 2:
        return None

    # Flatten data for ANOVA
    all_data = []
    group_indices = []

    for i, group in enumerate(groups_data):
        all_data.extend(group)
        group_indices.extend([i] * len(group))

    if len(set(all_data)) <= 1:  # No variation
        return {'f_stat': 0, 'p_value': 1.0, 'significant': False}

    # Manual ANOVA calculation
    return manual_anova(groups_data, all_data)

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
        p_value = stats.f.sf(f_stat, df_between, df_within)
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
    anova_result = perform_anova_test(groups_data, group_labels, metric_name)

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
    """Perform ANOVA tests on all key metrics."""
    print("=" * 80)
    print("COMPREHENSIVE ANOVA TESTING ON REAL STATCAST DATA")
    print("=" * 80)
    print(f"Dataset: {len(data)} observations from April 2023")
    print("Testing for significant differences between pitch clusters")

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
        result = analyze_metric_by_cluster(data, metric)
        if result:
            results[metric] = result

    # Summary
    print("\n" + "=" * 80)
    print("ANOVA SUMMARY RESULTS")
    print("=" * 80)

    significant_count = sum(1 for r in results.values() if r['significant'])
    total_tests = len(results)

    print(f"Total metrics tested: {total_tests}")
    print(f"Metrics with significant differences: {significant_count}")
    print(f"Success rate: {significant_count/total_tests*100:.1f}%" if total_tests > 0 else "Success rate: N/A")

    if significant_count > 0:
        print("\nSignificant findings:")
        for metric, result in results.items():
            if result['significant']:
                print(f"  ✓ {metric}: F={result['f_stat']:.2f}, p={result['p_value']:.4f}")

    return results

def main():
    print("=== REAL STATCAST DATA ANOVA ANALYSIS ===")

    # Load data
    data = load_integrated_data()
    if not data:
        print("Failed to load data")
        return

    # Perform comprehensive ANOVA testing
    results = perform_comprehensive_anova(data)

    # Save results
    output_file = str(_DEFAULT_DATA_DIR / 'anova_results_real_data.csv')
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

    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()