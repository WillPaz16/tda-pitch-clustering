#!/usr/bin/env python3
"""
Variance Analysis: Demonstrate that Stuff+ ratings have greater variance
across statcast pitch types than within granular clusters.

This shows that clusters capture meaningful mechanical differences that
statcast types miss, making clusters more predictive of pitcher success.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _ROOT / 'data'
_MODELS_DIR = _ROOT / 'models'
_RESULTS_DIR = _ROOT / 'results'

def load_data():
    """Load pitch data and Stuff+ ratings."""
    # Load pitches
    pitch_files = glob.glob(str(_DATA_DIR / 'classified_pitches_*.csv'))
    dfs = []
    for file in pitch_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except:
            continue
    pitch_data = pd.concat(dfs, ignore_index=True)

    # Load Stuff+ data
    stuff_df = pd.read_csv(_MODELS_DIR / 'stuff_plus_pitcher_level.csv')
    stuff_df['pitcher_id'] = stuff_df['pitcher'].astype(str)

    # Merge Stuff+ with pitch data
    pitch_data['pitcher_id'] = pitch_data['pitcher'].astype(str)
    merged_data = pd.merge(pitch_data, stuff_df[['pitcher_id', 'stuff_plus_overall']],
                          on='pitcher_id', how='left')

    return merged_data.dropna(subset=['stuff_plus_overall'])

def analyze_variance_by_statcast_type(data):
    """Analyze Stuff+ variance across statcast pitch types."""
    print("\n" + "="*60)
    print("STATCAST PITCH TYPE VARIANCE ANALYSIS")
    print("="*60)

    results = []
    for ptype in ['FF', 'SI', 'FC', 'CU', 'SL', 'CH']:
        type_data = data[data['pitch_type'] == ptype]
        if len(type_data) < 10:  # Skip if too few samples
            continue

        stuff_ratings = type_data['stuff_plus_overall'].values
        variance = np.var(stuff_ratings, ddof=1)  # Sample variance
        std_dev = np.std(stuff_ratings, ddof=1)
        mean_rating = np.mean(stuff_ratings)
        n_samples = len(stuff_ratings)

        results.append({
            'pitch_type': ptype,
            'mean_stuff_plus': mean_rating,
            'variance': variance,
            'std_dev': std_dev,
            'n_samples': n_samples,
            'cv': std_dev / mean_rating if mean_rating != 0 else 0  # Coefficient of variation
        })

        print(f"\n{ptype} (n={n_samples}):")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")

    results_df = pd.DataFrame(results)

    # Overall statistics
    overall_variance = np.var(data['stuff_plus_overall'].values, ddof=1)
    overall_std = np.std(data['stuff_plus_overall'].values, ddof=1)
    overall_mean = np.mean(data['stuff_plus_overall'].values)

    print(f"\n{'OVERALL (all pitches)':<20}: Mean={overall_mean:.2f}, Std={overall_std:.2f}, Var={overall_variance:.2f}")

    return results_df

def analyze_variance_by_cluster(data):
    """Analyze Stuff+ variance within granular clusters."""
    print("\n" + "="*60)
    print("CLUSTER-BASED VARIANCE ANALYSIS")
    print("="*60)

    # Get top clusters by frequency
    cluster_counts = data['cluster_id'].value_counts()
    top_clusters = cluster_counts.nlargest(20).index.tolist()  # Top 20 clusters

    results = []
    for cluster_id in top_clusters:
        cluster_data = data[data['cluster_id'] == cluster_id]
        if len(cluster_data) < 10:  # Skip if too few samples
            continue

        stuff_ratings = cluster_data['stuff_plus_overall'].values
        variance = np.var(stuff_ratings, ddof=1)
        std_dev = np.std(stuff_ratings, ddof=1)
        mean_rating = np.mean(stuff_ratings)
        n_samples = len(stuff_ratings)

        results.append({
            'cluster_id': cluster_id,
            'mean_stuff_plus': mean_rating,
            'variance': variance,
            'std_dev': std_dev,
            'n_samples': n_samples,
            'cv': std_dev / mean_rating if mean_rating != 0 else 0
        })

        print(f"\nCluster {cluster_id} (n={n_samples}):")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")

    results_df = pd.DataFrame(results)

    # Overall statistics for clusters
    cluster_variance = np.var(data['stuff_plus_overall'].values, ddof=1)
    cluster_std = np.std(data['stuff_plus_overall'].values, ddof=1)
    cluster_mean = np.mean(data['stuff_plus_overall'].values)

    print(f"\n{'OVERALL (all clusters)':<25}: Mean={cluster_mean:.2f}, Std={cluster_std:.2f}, Var={cluster_variance:.2f}")

    return results_df

def compare_variances(statcast_results, cluster_results):
    """Compare variance between statcast types and clusters."""
    print("\n" + "="*80)
    print("VARIANCE COMPARISON: Statcast Types vs Clusters")
    print("="*80)

    # Calculate average variances
    avg_statcast_var = statcast_results['variance'].mean()
    avg_cluster_var = cluster_results['variance'].mean()

    avg_statcast_std = statcast_results['std_dev'].mean()
    avg_cluster_std = cluster_results['std_dev'].mean()

    avg_statcast_cv = statcast_results['cv'].mean()
    avg_cluster_cv = cluster_results['cv'].mean()

    print("\nAVERAGE VARIANCE METRICS:")
    print(f"  Statcast Types:  Var={avg_statcast_var:.2f}, Std={avg_statcast_std:.2f}, CV={avg_statcast_cv:.3f}")
    print(f"  Clusters:        Var={avg_cluster_var:.2f}, Std={avg_cluster_std:.2f}, CV={avg_cluster_cv:.3f}")

    variance_ratio = avg_statcast_var / avg_cluster_var
    std_ratio = avg_statcast_std / avg_cluster_std
    cv_ratio = avg_statcast_cv / avg_cluster_cv

    print("\nRATIOS (Statcast / Cluster):")
    print(f"  Variance:     {variance_ratio:.2f}")
    print(f"  Std Dev:      {std_ratio:.2f}")
    print(f"  CV:           {cv_ratio:.3f}")

    # Statistical test
    statcast_vars = statcast_results['variance'].values
    cluster_vars = cluster_results['variance'].values

    # Mann-Whitney U test (non-parametric test for difference in variances)
    try:
        u_stat, p_value = stats.mannwhitneyu(statcast_vars, cluster_vars, alternative='greater')
        print("\nSTATISTICAL TEST (Mann-Whitney U):")
        print(f"  U-statistic: {u_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  ✓ SIGNIFICANT: Statcast types have greater variance than clusters")
        else:
            print("  ✗ NOT SIGNIFICANT: No difference in variance")
    except:
        print("  Could not perform statistical test")

    return {
        'variance_ratio': variance_ratio,
        'std_ratio': std_ratio,
        'cv_ratio': cv_ratio,
        'p_value': p_value if 'p_value' in locals() else None
    }

def create_visualization(statcast_results, cluster_results):
    """Create visualization comparing variances."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Variance comparison
    ax1.bar(['Statcast Types', 'Clusters'],
            [statcast_results['variance'].mean(), cluster_results['variance'].mean()],
            color=['red', 'blue'], alpha=0.7)
    ax1.set_title('Average Variance in Stuff+ Ratings')
    ax1.set_ylabel('Variance')
    ax1.grid(True, alpha=0.3)

    # Standard deviation comparison
    ax2.bar(['Statcast Types', 'Clusters'],
            [statcast_results['std_dev'].mean(), cluster_results['std_dev'].mean()],
            color=['red', 'blue'], alpha=0.7)
    ax2.set_title('Average Standard Deviation in Stuff+ Ratings')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(True, alpha=0.3)

    # Coefficient of variation
    ax3.bar(['Statcast Types', 'Clusters'],
            [statcast_results['cv'].mean(), cluster_results['cv'].mean()],
            color=['red', 'blue'], alpha=0.7)
    ax3.set_title('Average Coefficient of Variation in Stuff+ Ratings')
    ax3.set_ylabel('CV (Std/Mean)')
    ax3.grid(True, alpha=0.3)

    # Distribution of variances
    ax4.hist(statcast_results['variance'], alpha=0.7, label='Statcast Types', color='red', bins=10)
    ax4.hist(cluster_results['variance'], alpha=0.7, label='Clusters', color='blue', bins=10)
    ax4.set_title('Distribution of Variances')
    ax4.set_xlabel('Variance')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(_RESULTS_DIR / 'variance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'variance_comparison.png'")

def main():
    print("="*80)
    print("VARIANCE ANALYSIS: Stuff+ Distribution Across Statcast Types vs Clusters")
    print("="*80)

    print("\n1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} pitches with Stuff+ ratings")
    print(f"   {data['pitcher_id'].nunique()} pitchers, {data['cluster_id'].nunique()} clusters")

    print("\n2. Analyzing variance by statcast pitch type...")
    statcast_results = analyze_variance_by_statcast_type(data)

    print("\n3. Analyzing variance by cluster...")
    cluster_results = analyze_variance_by_cluster(data)

    print("\n4. Comparing variances...")
    comparison = compare_variances(statcast_results, cluster_results)

    print("\n5. Creating visualization...")
    create_visualization(statcast_results, cluster_results)
    

if __name__ == "__main__":
    main()